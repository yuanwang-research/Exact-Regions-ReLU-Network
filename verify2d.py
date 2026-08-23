#!/usr/bin/env python3
"""Independent (LP-free) region counter for 2-D inputs, used to cross-check RegionCounter.

* every activation pattern (all 2^#neurons of them) is tested by clipping a large box with
  the pattern's half-planes (exact Sutherland-Hodgman); the pattern is an activation region
  iff the resulting polygon has positive area (full dimension 2);
* two activation regions are merged iff they have the same affine output map and their
  closed polygons share an edge segment of positive length on a genuine constraint line
  (dimension 1 contact).
The box is chosen to contain every intersection point of every constraint line, so the
adjacency structure inside the box equals the true one.
"""
import itertools
import sys
from fractions import Fraction as F

from exact_regions import ReLUNetwork, RegionCounter


def pattern_data(net, bits):
    """Constraints (w, c): w.t + c > 0, and output (a, c0) for a full pattern; None if inconsistent."""
    n = 2
    M = [tuple(F(int(i == j)) for j in range(n)) for i in range(n)]
    v = tuple(F(0) for _ in range(n))
    cons = []
    for k in range(net.num_hidden_layers):
        W, b = net.weights[k], net.biases[k]
        A = [tuple(sum(row[j] * M[j][t] for j in range(len(row))) for t in range(n)) for row in W]
        d = [sum(row[j] * v[j] for j in range(len(row))) + bk for row, bk in zip(W, b)]
        for i, s in enumerate(bits[k]):
            if all(x == 0 for x in A[i]):
                if (d[i] > 0) != bool(s):
                    return None
            elif s:
                cons.append((A[i], d[i]))
            else:
                cons.append((tuple(-x for x in A[i]), -d[i]))
        M = [A[i] if bits[k][i] else tuple(F(0) for _ in range(n)) for i in range(len(A))]
        v = tuple(d[i] if bits[k][i] else F(0) for i in range(len(A)))
    W, b = net.weights[-1], net.biases[-1]
    a = tuple(sum(W[0][j] * M[j][t] for j in range(len(W[0]))) for t in range(n))
    c0 = sum(W[0][j] * v[j] for j in range(len(W[0]))) + b[0]
    return cons, (a, c0)


def clip(poly, w, c):
    """Keep the part of polygon with w.t + c >= 0."""
    out = []
    m = len(poly)
    for i in range(m):
        P, Q = poly[i], poly[(i + 1) % m]
        fp = w[0] * P[0] + w[1] * P[1] + c
        fq = w[0] * Q[0] + w[1] * Q[1] + c
        if fp >= 0:
            out.append(P)
        if (fp > 0 and fq < 0) or (fp < 0 and fq > 0):
            t = fp / (fp - fq)
            out.append((P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])))
    # remove duplicate consecutive vertices
    res = []
    for p in out:
        if not res or res[-1] != p:
            res.append(p)
    if len(res) > 1 and res[0] == res[-1]:
        res.pop()
    return res


def area2(poly):
    return sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
               for i in range(len(poly)))


def line_key(P, Q):
    """Normalised key of the line through P, Q."""
    a, b = Q[1] - P[1], P[0] - Q[0]
    c = -(a * P[0] + b * P[1])
    for v in (a, b):
        if v != 0:
            return (a / v, b / v, c / v)
    raise ValueError


def count_2d(net):
    sizes = net.hidden_widths
    pats = {}
    lines = set()
    for bits in itertools.product(*[itertools.product([0, 1], repeat=w) for w in sizes]):
        data = pattern_data(net, bits)
        if data is None:
            continue
        pats[bits] = data
        for (w, c) in data[0]:
            lines.add(line_key((0, 0), (0, 0)) if False else (w, c))
    # box containing all pairwise intersections of constraint lines
    B = F(1)
    L = list(lines)
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            (w1, c1), (w2, c2) = L[i], L[j]
            det = w1[0] * w2[1] - w1[1] * w2[0]
            if det != 0:
                x = (-c1 * w2[1] + c2 * w1[1]) / det
                y = (-w1[0] * c2 + w2[0] * c1) / det
                B = max(B, abs(x) + 1, abs(y) + 1)
        # also axis intercepts, so lines parallel to the box sides are well inside
        w, c = L[i]
        if w[0] != 0:
            B = max(B, abs(c / w[0]) + 1)
        if w[1] != 0:
            B = max(B, abs(c / w[1]) + 1)
    B = B + F(7, 3)
    box = [(-B, -B), (B, -B), (B, B), (-B, B)]
    regions = {}
    for bits, (cons, out) in pats.items():
        poly = box
        for (w, c) in cons:
            poly = clip(poly, w, c)
            if len(poly) < 3:
                break
        if len(poly) >= 3 and area2(poly) > 0:
            regions[bits] = (poly, out)
    # adjacency: shared edge of positive length on a non-box line, same output map
    keys = list(regions)
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def edges(poly):
        out = []
        m = len(poly)
        for i in range(m):
            P, Q = poly[i], poly[(i + 1) % m]
            key = line_key(P, Q)
            onbox = (P[0] == Q[0] and abs(P[0]) == B) or (P[1] == Q[1] and abs(P[1]) == B)
            if not onbox:
                out.append((key, P, Q))
        return out

    E = {k: edges(regions[k][0]) for k in keys}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ka, kb = keys[i], keys[j]
            if regions[ka][1] != regions[kb][1]:
                continue
            found = False
            for (key1, P1, Q1) in E[ka]:
                for (key2, P2, Q2) in E[kb]:
                    if key1 != key2:
                        continue
                    # parametrise along the line by the coordinate with larger direction component
                    d = (Q1[0] - P1[0], Q1[1] - P1[1])
                    ax = 0 if abs(d[0]) >= abs(d[1]) else 1
                    lo1, hi1 = sorted((P1[ax], Q1[ax]))
                    lo2, hi2 = sorted((P2[ax], Q2[ax]))
                    if min(hi1, hi2) - max(lo1, lo2) > 0:
                        found = True
                        break
                if found:
                    break
            if found:
                ra, rb = find(ka), find(kb)
                if ra != rb:
                    parent[ra] = rb
    n_lin = len({find(k) for k in keys})
    return len(regions), n_lin, regions


if __name__ == "__main__":
    net = ReLUNetwork.from_json(sys.argv[1])
    a, l, _ = count_2d(net)
    c = RegionCounter(net)
    print("independent: activation %d linear %d ; RegionCounter: %s" % (a, l, c.count()))
