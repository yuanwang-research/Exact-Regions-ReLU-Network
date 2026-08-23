#!/usr/bin/env python3
"""2-D analogue of the hourglass example: input 2, hidden (2, 3, 1).

Layer 1 is fixed to relu(t1), relu(t2) (allowed: the counts do not depend on the
first layer when input dim >= its width).  Layer 2 = relu(M h + m) is the old
first layer composed with an affine map R^2 -> R^3, layer 3 = relu(-s1-s2+s3-1).
In the open positive quadrant the non-constant locus is the quadrangle
Q = {t : M t + m in C}; outside it the picture is projected onto the axes.

Configurations:
  ring   : Q inside the open first quadrant, no axis meets Q  -> constant set is a ring
  axis   : the t1-axis crosses Q (t2-axis does not, origin outside Q)
Writes examples/quad_ring.json and examples/quad_axis.json.
"""
import json
import random
from fractions import Fraction as F

from exact_regions import ReLUNetwork, RegionCounter
from exact_regions.lp import open_polyhedron_nonempty, closed_polyhedron_nonempty
from exact_regions.numbers import fmt

CONE = [((F(0), F(0), F(1)), F(-1)), ((F(-1), F(0), F(1)), F(-1)),
        ((F(0), F(-1), F(1)), F(-1)), ((F(-1), F(-1), F(1)), F(-1))]


def quad(M, m):
    """Q = {t in R^2 : M t + m in C} as (w, c) with w.t + c >= 0."""
    out = []
    for (a, c0) in CONE:
        w = tuple(sum(a[i] * M[i][j] for i in range(3)) for j in range(2))
        c = sum(a[i] * m[i] for i in range(3)) + c0
        out.append((w, c))
    return out


def bounded(Q):
    rec = [(w, F(0)) for (w, _) in Q]
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        norm = (F(d[0]), F(d[1]))
        if closed_polyhedron_nonempty(2, rec + [(norm, F(-1))]):
            return False
    return True


def nonempty(Q, extra=(), eq=()):
    return open_polyhedron_nonempty(2, list(Q) + list(extra), list(eq))


def find(kind, seed=0):
    rng = random.Random(seed)
    vals = [F(k, 4) for k in range(-12, 13) if k != 0]
    while True:
        M = [[rng.choice(vals) for _ in range(2)] for _ in range(3)]
        m = [rng.choice(vals) for _ in range(3)]
        Q = quad(M, m)
        if not nonempty(Q) or not bounded(Q):
            continue
        X = ((F(1), F(0)), F(0))      # t1 > 0
        Y = ((F(0), F(1)), F(0))      # t2 > 0
        negX = ((F(-1), F(0)), F(0))
        negY = ((F(0), F(-1)), F(0))
        closedQ = Q
        if kind == "ring":
            # Q entirely in the open first quadrant: closure(Q) meets {t1<=0} or {t2<=0} nowhere
            if closed_polyhedron_nonempty(2, closedQ + [negX]) or closed_polyhedron_nonempty(2, closedQ + [negY]):
                continue
            return M, m, Q
        if kind == "axis":
            # t1-axis crosses the interior of Q at t1>0 ; t2-axis misses closure(Q); origin outside
            if not nonempty(Q, [X], [((F(0), F(1)), F(0))]):
                continue
            if closed_polyhedron_nonempty(2, closedQ + [negX]):
                continue
            if closed_polyhedron_nonempty(2, closedQ + [((F(1), F(0)), F(0)), ((F(-1), F(0)), F(0))]):
                continue   # closure(Q) meets the t2-axis
            return M, m, Q


def network(M, m, perturb_seed=None, eps=F(1, 10 ** 7)):
    I = [[F(1), F(0)], [F(0), F(1)]]
    W = [I, M, [[F(-1), F(-1), F(1)]], [[F(1)]]]
    b = [[F(0), F(0)], list(m), [F(-1)], [F(1, 3)]]
    if perturb_seed is not None:      # perturb everything except the fixed first layer
        rng = random.Random(perturb_seed)
        W = [W[0]] + [[[v * (1 + eps * rng.randint(-9, 9)) for v in r] for r in Wk] for Wk in W[1:]]
        b = [b[0]] + [[v * (1 + eps * rng.randint(-9, 9)) for v in bk] for bk in b[1:]]
    return ReLUNetwork(W, b)


def summarize(net, label):
    c = RegionCounter(net)
    acts = c.activation_regions()
    lins = c.linear_regions()
    const = [L for L in lins if all(v == 0 for v in L[0].output[0])]
    print("%s: activation regions %d, linear regions %d, constant linear regions %d"
          % (label, len(acts), len(lins), len(const)))


if __name__ == "__main__":
    for kind in ("ring", "axis"):
        M, m, Q = find(kind)
        print("== %s:  M = %s  m = %s" % (kind, [[fmt(v) for v in r] for r in M], [fmt(v) for v in m]))
        print("   quadrangle edges (w.t + c >= 0):", [([fmt(v) for v in w], fmt(c)) for (w, c) in Q])
        summarize(network(M, m), "   exact construction")
        net = network(M, m, perturb_seed=1)
        summarize(net, "   perturbed 1e-7")
        summarize(network(M, m, perturb_seed=2, eps=F(1, 10 ** 4)), "   perturbed 1e-4")
        with open("examples/quad_%s.json" % kind, "w") as f:
            json.dump(net.to_dict(), f)
