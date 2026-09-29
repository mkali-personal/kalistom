#!/usr/bin/env python3
"""
A one-hidden-layer head over the same context stack, with the hidden layer shared across taps.

The obvious MLP puts a hidden layer on the 10,250-wide stacked vector: 10,250 x 64 is 656k weights,
sixty times the logistic regression, fitted to 85 ad breaks. That is capacity pointed the wrong way.
This one applies a single small layer to each 1025-d base frame, and only then looks across time:

    h(j)     = relu(W1 . z(j) + b1)                    one H-vector per base frame
    score(i) = b2 + sum over taps t of  W2[t] . h(i - lag_t)

At H=32 that is 33k weights, of the same order as the linear head, and it keeps the property the
linear head's fast path rests on: every base frame is transformed once, and the context is applied
by indexing already-computed rows rather than by copying 1025-wide features ten times. On the phone
the same holds - one 1025 x H product per new frame, plus a 10H dot product over a ring of hidden
vectors - which is less arithmetic than the linear head's 10,250 multiply-adds.

Everything the linear head gets right carries over unchanged: the same ContextDesign supplies the
base matrix and the recording boundaries, context clamps to each recording's first frame rather than
reaching into the previous one, standardisation is fitted on training rows only, and the loss is
class-balanced the same way. Rows marked DROP carry no loss but still serve as context.

Plain numpy with the gradient written out, like context_lr.py. Two layers do not need a framework,
and the trainer's requirements stay free of one.

It trains for a fixed number of epochs. Stopping early on the held-out fold would pick the epoch
that happens to suit the recording being tested, which is a leak by another name.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from context_lr import ContextDesign


def context_index(design: ContextDesign) -> np.ndarray:
    """(n, taps) row numbers, oldest tap first, clamped at each recording's start.

    The same convention as ContextDesign.gather_into and train.stack_context, written as indices
    because the hidden layer is not linear and cannot be shifted as scalars after the fact.
    """
    idx = np.empty((design.n, design.taps), dtype=np.int64)
    lags = np.arange(design.taps)[::-1]
    for a, b in design.bounds:
        rows = np.arange(a, b)[:, None] - lags[None, :]
        idx[a:b] = np.maximum(rows, a)
    return idx


@dataclass
class Head:
    w1: np.ndarray      # (hidden, d0)
    b1: np.ndarray      # (hidden,)
    w2: np.ndarray      # (taps, hidden), oldest tap first
    b2: float

    @classmethod
    def init(cls, d0: int, taps: int, hidden: int, rng) -> "Head":
        # He initialisation for the ReLU layer; the output layer starts small so the first
        # predictions sit near 0.5 rather than saturated one way or the other.
        return cls(w1=(rng.normal(size=(hidden, d0)) * np.sqrt(2.0 / d0)).astype(np.float32),
                   b1=np.zeros(hidden, np.float32),
                   w2=(rng.normal(size=(taps, hidden)) * 0.01).astype(np.float32),
                   b2=0.0)

    def params(self) -> list[np.ndarray]:
        return [self.w1, self.b1, self.w2]


def _sigmoid(s: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))


def fit(design: ContextDesign, train: np.ndarray, y: np.ndarray, mu, sd, hidden: int,
        weight_decay: float, dropout: float, epochs: int, batch: int = 512,
        lr: float = 1e-3, seed: int = 0) -> Head:
    """AdamW on class-balanced log loss. Dropout is on the standardised inputs, where the
    redundancy is: 1025 correlated embedding dimensions, of which any one should not matter."""
    rng = np.random.default_rng(seed)
    idx = context_index(design)
    T, d0 = design.taps, design.d0
    inv_sd = (1.0 / sd).astype(np.float32)

    rows = np.flatnonzero(train)
    pos = max(int((y[rows] == 1).sum()), 1)
    neg = max(int((y[rows] == 0).sum()), 1)
    # Same balance as the linear head: each class carries half the total weight.
    w_row = (np.where(y == 1, 0.5 / pos, 0.5 / neg) * len(rows)).astype(np.float32)

    m = Head.init(d0, T, hidden, rng)
    # Adam state for w1, b1, w2 and b2 (the last as a 1-element array so it updates in place).
    b2 = np.zeros(1, np.float32)
    params = m.params() + [b2]
    m1 = [np.zeros_like(p) for p in params]
    m2 = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    last = 0.0
    for _ in range(epochs):
        rng.shuffle(rows)
        total = 0.0
        for k in range(0, len(rows), batch):
            b = rows[k:k + batch]
            B = len(b)
            z = (design.E[idx[b]] - mu) * inv_sd                    # (B, T, d0)
            if dropout:
                z *= (rng.random(z.shape, dtype=np.float32) >= dropout) / (1.0 - dropout)
            a = z @ m.w1.T + m.b1                                   # (B, T, H)
            h = np.maximum(a, 0.0)
            s = np.einsum("bth,th->b", h, m.w2) + b2[0]
            p = _sigmoid(s)
            yt, wt = y[b], w_row[b]
            total += float(np.sum(wt * (np.logaddexp(0, s) - yt * s)))

            ds = (wt * (p - yt) / B).astype(np.float32)             # (B,)
            g_w2 = np.einsum("b,bth->th", ds, h)
            g_b2 = np.array([ds.sum()], np.float32)
            da = (ds[:, None, None] * m.w2[None]) * (a > 0)         # (B, T, H)
            g_w1 = da.reshape(-1, hidden).T @ z.reshape(-1, d0)
            g_b1 = da.sum((0, 1))

            step += 1
            c1, c2 = 1 - beta1 ** step, 1 - beta2 ** step
            for i, (prm, g) in enumerate(zip(params, (g_w1, g_b1, g_w2, g_b2))):
                m1[i] = beta1 * m1[i] + (1 - beta1) * g
                m2[i] = beta2 * m2[i] + (1 - beta2) * g * g
                if prm.ndim == 2:                                   # decoupled decay, weights only
                    prm *= 1 - lr * weight_decay
                prm -= lr * (m1[i] / c1) / (np.sqrt(m2[i] / c2) + eps)
        last = total / len(rows)
    m.b2 = float(b2[0])
    fit.last = f"{epochs} epochs, final train loss {last:.4f}"
    return m


def predict(design: ContextDesign, m: Head, mu, sd, batch: int = 8192) -> np.ndarray:
    """Hidden vectors once per base frame, then the taps read them by index - the phone's order."""
    inv_sd = (1.0 / sd).astype(np.float32)
    H = np.empty((design.n, len(m.b1)), np.float32)
    for k in range(0, design.n, batch):
        H[k:k + batch] = np.maximum((design.E[k:k + batch] - mu) * inv_sd @ m.w1.T + m.b1, 0.0)
    idx = context_index(design)
    out = np.empty(design.n, np.float32)
    for k in range(0, design.n, batch):
        out[k:k + batch] = _sigmoid(np.einsum("bth,th->b", H[idx[k:k + batch]], m.w2) + m.b2)
    return out


def export(m: Head, mu, sd) -> dict:
    """Weights in the layout Detector.kt would read: w1 row-major (hidden, d0), w2 oldest tap
    first as (taps, hidden). Statistics are per BASE column, not tiled - the MLP standardises a
    frame once, before the hidden layer, not once per tap."""
    return {
        "hidden": int(len(m.b1)),
        "mean": np.asarray(mu, np.float64).tolist(),
        "scale": np.asarray(sd, np.float64).tolist(),
        "w1": m.w1.astype(np.float64).ravel().tolist(),
        "b1": m.b1.astype(np.float64).tolist(),
        "w2": m.w2.astype(np.float64).ravel().tolist(),
        "b2": m.b2,
    }
