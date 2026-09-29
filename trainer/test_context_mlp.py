"""The MLP head must agree with the stacked definition of the context, and its gradient with the loss.

context_index replaces the shifts the linear head uses, and a wrong lag or a missed clamp would not
crash; it would quietly score each row against the wrong frames. This builds the stack explicitly
with train.stack_context, runs the same network over it frame by frame, and compares. The gradient
is written by hand, so it is checked too: on a tiny problem with random labels, the optimiser has
to be able to memorise them, which a gradient with a wrong sign or a missing term cannot do.

    python trainer/test_context_mlp.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_lr import ContextDesign                                  # noqa: E402
import context_mlp                                                    # noqa: E402
from context_mlp import Head, predict                                 # noqa: E402
from train import stack_context                                       # noqa: E402

rng = np.random.default_rng(0)
T, d0, H = 10, 37, 8
lens = [53, 7, 4, 120, 1]          # includes recordings shorter than the context
E = rng.normal(size=(sum(lens), d0)).astype(np.float32)

bounds, a = [], 0
for L in lens:
    bounds.append((a, a + L)); a += L
des = ContextDesign(E, bounds, T)

S = np.vstack([stack_context(E[a:b], T) for a, b in bounds])
mu = E.mean(0).astype(np.float32); sd = (E.std(0) + 1e-6).astype(np.float32)
Z = (S.reshape(len(E), T, d0) - mu) / sd

m = Head.init(d0, T, H, rng)
m.w2 = rng.normal(size=(T, H)).astype(np.float32)
m.b2 = 0.3
h_ref = np.maximum(Z @ m.w1.T + m.b1, 0.0)
p_ref = 1.0 / (1.0 + np.exp(-(np.einsum("bth,th->b", h_ref, m.w2) + m.b2)))
p_new = predict(des, m, mu, sd, batch=16)
d = np.abs(p_ref - p_new).max()
print("probabilities max abs diff : %.3e" % d)
assert d < 1e-5, "indexed forward pass disagrees with the stacked reference"

# Gradient: 180 rows of random labels against a few hundred weights. A correct gradient memorises
# them and more than halves the loss; a wrong one wanders.
y = (rng.random(len(E)) < 0.3).astype(np.float64)
train = np.ones(len(E), bool)


def loss(head):
    p = predict(des, head, mu, sd).astype(np.float64)
    pos, neg = y.sum(), (1 - y).sum()
    w = np.where(y == 1, 0.5 / pos, 0.5 / neg) * len(y)
    return float(np.mean(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))))


before = loss(Head.init(d0, T, H, np.random.default_rng(1)))
fitted = context_mlp.fit(des, train, y, mu, sd, hidden=H, weight_decay=0.0, dropout=0.0,
                         epochs=200, batch=len(E), lr=1e-2, seed=1)
after = loss(fitted)
print("loss before %.4f, after 200 full-batch steps %.4f" % (before, after))
assert after < 0.5 * before, "hand-written gradient does not descend the loss"
print("OK - indexed context matches the stacked definition, and the gradient descends")
