"""ContextDesign must agree with the stacked matrix it replaces, to float32 precision.

The fast path shifts scalars where the old code copied features, so an off-by-one in a lag or a
missed clamp at a recording boundary would not crash - it would quietly train on the wrong frames.
This builds the stack explicitly with train.stack_context and compares both the scores and the
gradient. Recording lengths shorter than the context are included on purpose: those clamp every tap.

    python trainer/test_context_lr.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_lr import ContextDesign                                  # noqa: E402
from train import stack_context                                       # noqa: E402

rng = np.random.default_rng(0)
T, d0 = 10, 37
lens = [53, 7, 4, 120, 1]          # includes recordings shorter than the context
E = rng.normal(size=(sum(lens), d0)).astype(np.float32)

bounds, a = [], 0
for L in lens:
    bounds.append((a, a + L)); a += L

S = np.vstack([stack_context(E[a:b], T) for a, b in bounds])
assert S.shape == (len(E), T * d0)

mu = E.mean(0).astype(np.float32); sd = (E.std(0) + 1e-6).astype(np.float32)
mu_s, sd_s = np.tile(mu, T), np.tile(sd, T)
Z = (S - mu_s) / sd_s

W = (rng.normal(size=(T, d0)) * 0.1).astype(np.float32)
b = 0.37
des = ContextDesign(E, bounds, T)

s_ref = Z @ W.ravel() + b
s_new = des.scores(W, b, mu, sd)
ds = np.abs(s_ref - s_new).max()
print("scores   max abs diff : %.3e   (scale %.3e)" % (ds, np.abs(s_ref).max()))
assert ds < 1e-4, "scores disagree with the stacked reference"

r = rng.normal(size=len(E)).astype(np.float32)
g_ref = (Z.T @ r).reshape(T, d0)
g_new = des.grad(r, sd, mu)
dg = np.abs(g_ref - g_new).max()
print("gradient max abs diff : %.3e   (scale %.3e)" % (dg, np.abs(g_ref).max()))
assert dg < 1e-3, "gradient disagrees with the stacked reference"
print("OK - fast path matches the stacked definition")
