#!/usr/bin/env python3
"""
Logistic regression over a context stack, without ever building the context stack.

The head sees ten consecutive frames, so the obvious design matrix is 10 x 1025 columns wide -
and every embedding appears in ten of its rows. At 33 recordings that is 128,366 x 10,250, which
is 2.6 GB of tenfold-redundant data that has to be read twice per optimiser iteration. Thirty-one
folds of five hundred iterations came to seventy-five hours.

None of it is necessary, because the model is linear:

    score(i) = b + sum over taps t of  z(i - lag_t) . w_t

where z is one standardised *base* frame. So the whole forward pass is one product against the
base matrix, E @ W', giving a score per (row, tap); the context is then applied by *shifting those
scalars*, not by copying the features. The backward pass mirrors it: the residual is scattered
back along the same shifts and one product E' @ R gives every tap's gradient.

Two reads of 526 MB per iteration instead of two of 2.6 GB, for identical arithmetic.

The shifts are contiguous slices rather than gathers. Within a recording, "the frame `lag` back"
is just the base matrix offset by `lag` rows; only the first `lag` rows of each recording clamp,
because context must not reach across a recording boundary into unrelated audio.

Standardisation is folded into the weights rather than applied to the data - (E - mu)/sd . w is
(E . w/sd) - (mu/sd . w) - so the base matrix is never copied or rewritten either. One simplifying
choice: the statistics are computed per base column and shared across taps, rather than separately
for each of the 10,250 stacked columns. A tap is the same column shifted, so its values are the
same multiset apart from `lag` rows at each recording's edge - a fraction of a percent of a
half-hour recording.
"""
from __future__ import annotations

import numpy as np


class ContextDesign:
    """A base feature matrix plus the recording boundaries context must not cross."""

    def __init__(self, E: np.ndarray, bounds: list[tuple[int, int]], taps: int):
        self.E = E                       # (n, d0) float32
        self.bounds = bounds             # [(start, stop)] per recording, in row order
        self.taps = taps                 # lags are taps-1 .. 0, oldest first
        self.n, self.d0 = E.shape

    # ---- the two shift operations, both contiguous ----

    def gather_into(self, P: np.ndarray, out: np.ndarray) -> None:
        """out[i] += sum_t P[i - lag_t, t], clamped at each recording's start."""
        for t in range(self.taps):
            lag = self.taps - 1 - t
            col = P[:, t]
            for a, b in self.bounds:
                if b - a <= lag:
                    out[a:b] += col[a]
                    continue
                out[a + lag:b] += col[a:b - lag]
                if lag:
                    out[a:a + lag] += col[a]

    def scatter_from(self, r: np.ndarray, R: np.ndarray) -> None:
        """R[j, t] = sum of r[i] over rows i whose tap t reads row j. The transpose of gather."""
        R.fill(0.0)
        for t in range(self.taps):
            lag = self.taps - 1 - t
            col = R[:, t]
            for a, b in self.bounds:
                if b - a <= lag:
                    col[a] += r[a:b].sum()
                    continue
                col[a:b - lag] += r[a + lag:b]
                if lag:
                    col[a] += r[a:a + lag].sum()

    # ---- statistics over a subset of rows, on the base matrix only ----

    def mean_std(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = self.E[mask]
        mu = rows.mean(0, dtype=np.float64)
        sd = rows.std(0, dtype=np.float64) + 1e-6
        return mu.astype(np.float32), sd.astype(np.float32)

    def scores(self, W: np.ndarray, bias: float, mu, sd) -> np.ndarray:
        """Score every row. W is (taps, d0)."""
        Wn = (W / sd).astype(np.float32)
        P = self.E @ Wn.T                                  # (n, taps) - one pass over E
        s = np.full(self.n, bias - float((Wn @ mu).sum()), np.float32)
        self.gather_into(P, s)
        return s

    def grad(self, r: np.ndarray, sd, mu) -> np.ndarray:
        """d/dW of sum_i r_i * score_i, as (taps, d0)."""
        R = np.empty((self.n, self.taps), np.float32)
        self.scatter_from(r, R)
        G = (self.E.T @ R).T                               # (taps, d0) - one pass over E
        return (G - np.outer(R.sum(0), mu)) / sd


def fit(design: ContextDesign, train: np.ndarray, y: np.ndarray, mu, sd,
        l2: float, iters: int) -> np.ndarray:
    """Class-balanced logistic regression. Returns weights flattened as (taps*d0 + 1,)."""
    from scipy.optimize import minimize

    T, d0 = design.taps, design.d0
    n = int(train.sum())
    pos = max(int(((y == 1) & train).sum()), 1)
    neg = max(int(((y == 0) & train).sum()), 1)
    sw = np.where(y == 1, 0.5 / pos, 0.5 / neg) * n
    sw = np.where(train, sw, 0.0)                          # rows outside the fold contribute 0

    def loss_grad(w):
        W = w[:T * d0].reshape(T, d0).astype(np.float32)
        s = design.scores(W, float(w[-1]), mu, sd).astype(np.float64)
        np.clip(s, -30, 30, out=s)
        ll = float(np.sum(sw * (np.logaddexp(0, s) - y * s))) / n
        r = (sw * (1.0 / (1.0 + np.exp(-s)) - y) / n).astype(np.float32)
        g = np.empty(T * d0 + 1)
        g[:T * d0] = design.grad(r, sd, mu).ravel()
        g[-1] = float(r.sum())
        g[:T * d0] += l2 * w[:T * d0]
        return ll + 0.5 * l2 * float(w[:T * d0] @ w[:T * d0]), g

    res = minimize(loss_grad, np.zeros(T * d0 + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": iters, "maxcor": 20})
    # Worth knowing which of the two happened: a fit that stopped because it converged is done,
    # one that stopped at the cap is whatever the optimiser had reached when the budget ran out.
    fit.last = f"{res.nit} iters, {'converged' if res.nit < iters else 'hit --iters cap'}"
    return res.x.astype(np.float32)


def predict(design: ContextDesign, w: np.ndarray, mu, sd) -> np.ndarray:
    T, d0 = design.taps, design.d0
    s = design.scores(w[:T * d0].reshape(T, d0), float(w[-1]), mu, sd)
    return (1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))).astype(np.float32)
