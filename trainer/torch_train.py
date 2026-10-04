#!/usr/bin/env python3
"""
The same heads as train.py, fitted with PyTorch - on a GPU when there is one, otherwise the CPU.

train.py remains the reference. It builds the pool (embeddings, loudness, labels, join guards,
same-broadcast folding) and fits the heads in plain numpy; this file only replaces the fitting, so
it can run somewhere faster. It reads the pool that `train.py --export` writes, trains one model
per held-out fold, and writes out-of-fold predictions in exactly the format `train.py --save-oof`
does - smooth.py, sweep.py and the label editor read either without knowing which made it.

It needs nothing from this repository but numpy and torch, so the whole file can be shipped to a
cloud machine as it is (kaggle_run.py does that).

    python trainer/train.py --dir captures/stitched --dir captures/sessions --export pool.npz
    python trainer/torch_train.py pool.npz --head lr --l2 1e-3 --out oof_lr.npz
    python trainer/torch_train.py pool.npz --head mlp --hidden 16 --wd 1e-2 --out oof_mlp.npz

THE HEADS, as in context_lr.py and context_mlp.py:

    lr    logistic regression over the stack of `context` frames. L-BFGS on the class-balanced log
          loss plus 0.5 * l2 * |W|^2, like scipy's L-BFGS-B in train.py.
    mlp   one hidden layer applied to each frame separately and shared across taps, then linear
          over time. AdamW with dropout on the inputs, a fixed number of epochs - never stopped
          early on the held-out fold, which would pick the epoch that suits the test recording.

Both see standardised inputs whose mean and spread come from the training rows of each fold only.
Rows labelled DROP carry no loss but still serve as context, and context never reaches across a
recording boundary: a frame near a recording's start repeats that recording's first frame.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

POSITIVE, NEGATIVE, DROP = 1, 0, -1          # as in trainer/labels.py


def context_index(bounds, taps: int, n: int) -> np.ndarray:
    """(n, taps) row numbers, oldest tap first, clamped at each recording's start."""
    idx = np.empty((n, taps), dtype=np.int64)
    lags = np.arange(taps)[::-1]
    for a, b in bounds:
        idx[a:b] = np.maximum(np.arange(a, b)[:, None] - lags[None, :], a)
    return idx


def balanced_weights(y: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
    """Each class carries half the total weight; rows outside the fold carry none."""
    n = int(train.sum())
    pos = max(int(((y == 1) & train).sum()), 1)
    neg = max(int(((y == 0) & train).sum()), 1)
    w = torch.where(y == 1, 0.5 / pos, 0.5 / neg) * n
    return torch.where(train, w, torch.zeros_like(w))


# ---------------------------------------------------------------- the two heads

def fit_lr(Z, idx, y, train, l2: float, iters: int, **_):
    T, d0 = idx.shape[1], Z.shape[1]
    W = torch.zeros(T, d0, device=Z.device, requires_grad=True)
    b = torch.zeros(1, device=Z.device, requires_grad=True)
    sw = balanced_weights(y, train)
    n = float(train.sum())
    taps = torch.arange(T, device=Z.device)

    def scores():
        P = Z @ W.T                                          # (n, taps): one pass over the frames
        return P[idx, taps].sum(1) + b                       # each row reads its own lags

    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20,
                            tolerance_grad=1e-5, tolerance_change=1e-10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        s = scores()
        ll = (sw * (torch.nn.functional.softplus(s) - y * s)).sum() / n
        loss = ll + 0.5 * l2 * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    steps = opt.state[opt._params[0]].get("n_iter", 0)
    with torch.no_grad():
        return torch.sigmoid(scores()), f"{steps} L-BFGS iterations"


def fit_mlp(Z, idx, y, train, hidden: int, wd: float, dropout: float, epochs: int,
            batch: int, lr: float, seed: int, **_):
    g = torch.Generator(device="cpu").manual_seed(seed)
    T, d0, dev = idx.shape[1], Z.shape[1], Z.device
    w1 = (torch.randn(hidden, d0, generator=g) * (2.0 / d0) ** 0.5).to(dev).requires_grad_()
    b1 = torch.zeros(hidden, device=dev, requires_grad=True)
    w2 = (torch.randn(T, hidden, generator=g) * 0.01).to(dev).requires_grad_()
    b2 = torch.zeros(1, device=dev, requires_grad=True)
    # Decoupled decay on the two weight matrices only, as in context_mlp.py.
    opt = torch.optim.AdamW([{"params": [w1, w2], "weight_decay": wd},
                             {"params": [b1, b2], "weight_decay": 0.0}], lr=lr)
    sw = balanced_weights(y, train)
    rows = torch.nonzero(train).squeeze(1).cpu()
    last = 0.0
    for _ in range(epochs):
        order = rows[torch.randperm(len(rows), generator=g)].to(dev)
        total = 0.0
        for k in range(0, len(order), batch):
            bi = order[k:k + batch]
            z = Z[idx[bi]]                                  # (B, taps, d0)
            if dropout:
                keep = (torch.rand(z.shape, device=dev) >= dropout).float()
                z = z * keep / (1.0 - dropout)
            h = torch.relu(z @ w1.T + b1)                     # (B, taps, hidden)
            s = (h * w2).sum((1, 2)) + b2
            loss_rows = sw[bi] * torch.nn.functional.binary_cross_entropy_with_logits(
                s, y[bi], reduction="none")
            loss = loss_rows.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss_rows.detach().sum())
        last = total / len(rows)
    with torch.no_grad():
        H = torch.relu(Z @ w1.T + b1)                         # hidden vectors once per frame
        s = (H[idx] * w2).sum((1, 2)) + b2                  # then the taps read them by index
        return torch.sigmoid(s), f"{epochs} epochs, final train loss {last:.4f}"


def trailing_index(bounds, n: int, width: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """For each row: where its trailing window of `width` frames starts, and how many it holds.

    The window stops at the recording's start rather than reaching into the previous one, so the
    first frames of a recording average over fewer frames - the same way the phone's ring buffer
    fills up."""
    lo = np.empty(n, np.int64)
    for a, b in bounds:
        lo[a:b] = np.maximum(np.arange(a, b) - width + 1, a)
    count = np.arange(n) + 1 - lo
    return (torch.as_tensor(lo, device=device),
            torch.as_tensor(count, dtype=torch.float32, device=device))


def fit_pool(Z, idx, y, train, hidden: int, dense: int, windows: str, wd: float, dropout: float,
             epochs: int, batch: int, lr: float, seed: int, bounds=None, **_):
    """Multiple-timescale pooling: the current frame plus what the last few seconds, the last
    quarter-minute and the last minute have sounded like on average.

        h(t)  = relu(W1 . z(t) + b1)                        per frame, as in the mlp head
        x(t)  = [h(t), mean h over last w1, w2, w3 frames]  causal, clamped at recording start
        score = w3 . relu(W2 . x(t) + b2) + b3

    A minute of average is the kind of context that says "we are inside a break" when the current
    second is ambiguous, which the ten-frame heads cannot see. The trailing means make every row
    depend on up to a minute of its neighbours, so each step recomputes h for ALL frames (cheap on a
    GPU: one 172k x 1025 x H product) and takes the loss on a minibatch of rows. Dropout is applied
    to h rather than to the 1025 inputs, which would cost a random number per input per step.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev, d0, n = Z.device, Z.shape[1], Z.shape[0]
    ws = [int(w) for w in str(windows).split(",")]
    trail = [trailing_index(bounds, n, w, dev) for w in ws]
    feat = hidden * (1 + len(ws))
    w1 = (torch.randn(hidden, d0, generator=g) * (2.0 / d0) ** 0.5).to(dev).requires_grad_()
    b1 = torch.zeros(hidden, device=dev, requires_grad=True)
    w2 = (torch.randn(dense, feat, generator=g) * (2.0 / feat) ** 0.5).to(dev).requires_grad_()
    b2 = torch.zeros(dense, device=dev, requires_grad=True)
    w3 = (torch.randn(1, dense, generator=g) * 0.01).to(dev).requires_grad_()
    b3 = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.AdamW([{"params": [w1, w2, w3], "weight_decay": wd},
                             {"params": [b1, b2, b3], "weight_decay": 0.0}], lr=lr)
    sw = balanced_weights(y, train)

    def features(train_mode: bool) -> torch.Tensor:
        h = torch.relu(Z @ w1.T + b1)                                      # (n, hidden)
        if train_mode and dropout:
            h = h * (torch.rand(h.shape, device=dev) >= dropout).float() / (1.0 - dropout)
        c = torch.cat([torch.zeros(1, hidden, device=dev, dtype=torch.float64),
                       torch.cumsum(h.double(), 0)])
        parts = [h]
        for lo, cnt in trail:
            parts.append(((c[1:] - c[lo]) / cnt[:, None]).float())       # causal trailing mean
        return torch.cat(parts, 1)

    def score_rows(x: torch.Tensor) -> torch.Tensor:
        return (torch.relu(x @ w2.T + b2) @ w3.T).squeeze(1) + b3

    rows = torch.nonzero(train).squeeze(1).cpu()
    last = 0.0
    for _ in range(epochs):
        order = rows[torch.randperm(len(rows), generator=g)].to(dev)
        total = 0.0
        for k in range(0, len(order), batch):
            bi = order[k:k + batch]
            x = features(True)[bi]
            loss_rows = sw[bi] * torch.nn.functional.binary_cross_entropy_with_logits(
                score_rows(x), y[bi], reduction="none")
            loss = loss_rows.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss_rows.detach().sum())
        last = total / len(rows)
    with torch.no_grad():
        return (torch.sigmoid(score_rows(features(False))),
                f"{epochs} epochs, final train loss {last:.4f}")


HEADS = {"lr": fit_lr, "mlp": fit_mlp, "pool": fit_pool}
DEFAULTS = {"lr": {"l2": 1e-3, "iters": 3000},
            "mlp": {"hidden": 16, "wd": 1e-2, "dropout": 0.2, "epochs": 8, "batch": 512,
                    "lr": 1e-3, "seed": 0},
            # 10, 31 and 125 frames: about 5 s, 15 s and 60 s of trailing context.
            "pool": {"hidden": 16, "dense": 32, "windows": "10,31,125", "wd": 1e-2,
                     "dropout": 0.2, "epochs": 8, "batch": 512, "lr": 1e-3, "seed": 0}}


# ---------------------------------------------------------------- running a pool

def pool_fingerprint(d) -> str:
    """Identifies a pool by its labels and folds. kaggle_run.py computes the same thing locally
    (kept identical by hand, because importing this file there would need torch)."""
    import hashlib
    h = hashlib.sha1()
    for k in ("y", "fold"):
        h.update(np.ascontiguousarray(d[k]).tobytes())
    return h.hexdigest()[:16]


def run(pool: str, head: str, out: str, device: str = "auto", max_folds: int = 0,
        log=print, expect_pool: str = "", **params) -> dict:
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else ("cpu" if device == "auto" else device))
    cfg = {**DEFAULTS[head], **{k: v for k, v in params.items() if v is not None}}
    d = np.load(pool, allow_pickle=True)
    if expect_pool and pool_fingerprint(d) != expect_pool:
        # Kaggle can mount the previous version of a dataset that was re-uploaded moments ago.
        # Training on it would quietly report on the old data under the new run's name.
        raise RuntimeError(f"STALE POOL: mounted {pool_fingerprint(d)}, expected {expect_pool}")
    y_np, fold = d["y"].astype(np.int8), d["fold"]
    n, taps = len(y_np), int(d["context"])
    idx = torch.as_tensor(context_index(d["rec_bounds"], taps, n), device=dev)
    E = torch.as_tensor(d["E"].astype(np.float32), device=dev)
    y = torch.as_tensor((y_np == POSITIVE).astype(np.float32), device=dev)
    keep = torch.as_tensor(y_np != DROP, device=dev)
    fold_t = torch.as_tensor(fold, device=dev)
    # A fold below zero is training-only (train.py --test-newest-hours): never held out.
    folds = sorted(set(int(f) for f in fold if f >= 0))
    if max_folds:
        folds = folds[:max_folds]
    log(f"{head} {json.dumps(cfg)} on {dev}"
        + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else "")
        + f": {n} rows, {len(folds)} folds, {taps} taps")

    oof = np.zeros(n, np.float32)
    scored = np.zeros(n, bool)
    t_all = time.time()
    for f in folds:
        t0 = time.time()
        tr, te = keep & (fold_t != f), fold_t == f
        if not bool((y[tr] == 1).any()) or not bool((y[tr] == 0).any()):
            log(f"  fold {f}: one class missing in training - skipped")
            continue
        rows = E[tr]
        mu = rows.double().mean(0).float()
        sd = rows.double().std(0, unbiased=False).float() + 1e-6
        Z = (E - mu) / sd
        p, how = HEADS[head](Z, idx, y, tr, bounds=d["rec_bounds"], **cfg)
        te_np = te.cpu().numpy()
        oof[te_np] = p[te].float().cpu().numpy()
        scored |= te_np
        del Z, rows
        # Pools exported before day folds have no fold_names; there each fold was one group.
        fnames = d["fold_names"] if "fold_names" in d.files else d["names"]
        name = str(fnames[f]) if f < len(fnames) else f"fold {f}"
        log(f"  held out {name} [{time.time() - t0:.0f}s, {how}]")

    k = (y_np != DROP) & scored                   # judged on the folds actually run
    pred, truth = oof[k] >= 0.5, y_np[k] == POSITIVE
    tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
    summary = {"head": head, "params": cfg, "device": str(dev), "folds": len(folds),
               "seconds": round(time.time() - t_all, 1),
               "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
    log(f"POOLED  precision {100 * summary['precision']:.1f}%  recall "
        f"{100 * summary['recall']:.1f}%  in {summary['seconds']:.0f}s")
    np.savez(out, y=y_np, p=oof, groups=d["groups"], names=d["names"],
             rec_names=d["rec_names"], rec_bounds=d["rec_bounds"], fold=fold,
             fold_names=d["fold_names"] if "fold_names" in d.files else d["names"],
             params=json.dumps(summary))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("pool", help=".npz written by train.py --export")
    ap.add_argument("--head", choices=sorted(HEADS), required=True)
    ap.add_argument("--out", required=True, help="out-of-fold predictions, as train.py writes")
    ap.add_argument("--device", default="auto", help="auto, cuda or cpu")
    ap.add_argument("--max-folds", type=int, default=0, help="smoke test: stop after this many")
    for k in ("l2", "wd", "dropout", "lr"):
        ap.add_argument(f"--{k}", type=float)
    for k in ("iters", "hidden", "dense", "epochs", "batch", "seed"):
        ap.add_argument(f"--{k}", type=int)
    ap.add_argument("--windows", help="pool: trailing windows in frames, e.g. 10,31,125")
    a = vars(ap.parse_args())
    run(a.pop("pool"), a.pop("head"), a.pop("out"), a.pop("device"), a.pop("max_folds"), **a)
    return 0


# ---- entry point (kaggle_run.py replaces everything below this line with its own runner) ----
if __name__ == "__main__":
    raise SystemExit(main())
