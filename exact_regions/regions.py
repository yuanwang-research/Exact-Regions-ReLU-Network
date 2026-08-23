"""Exact enumeration of activation regions and linear regions.

Definitions used here
---------------------
* **Activation pattern**: for every hidden neuron, whether it is active
  (pre-activation > 0) or not (pre-activation <= 0).
* **Activation region**: the set of inputs with a given activation pattern.
  It is an (open, convex) polyhedron.  We count the patterns whose region has
  non-empty interior (full dimension n); patterns that only occur on
  measure-zero sets (exactly on a boundary hyperplane) are not regions.
  A neuron whose pre-activation is *constant* on a region (e.g. all its inputs
  are dead there) contributes no boundary; its state is the sign of the constant.
* **Linear region**: a maximal connected set on which the network is given by
  one affine function.  Formally: the connected components of the set of
  points at which the network is locally affine (Hanin & Rolnick).  Two
  activation regions belong to the same linear region iff they are linked by a
  chain of activation regions that all have the same output affine function and
  consecutive ones share an (n-1)-dimensional face.  This is option
  ``adjacency="facet"`` (default).  With ``adjacency="touch"`` it suffices that
  the closures of consecutive regions intersect (a weaker, set-theoretic
  notion of "connected region").

Everything is computed with exact rational arithmetic; every non-emptiness
question is an exact LP (see ``lp.py``).
"""
import sys
from collections import defaultdict
from fractions import Fraction
from .lp import open_polyhedron_nonempty, closed_polyhedron_nonempty
from .numbers import check, fmt

ZERO = Fraction(0)


class TooManyRegionsError(RuntimeError):
    pass


class ActivationRegion:
    __slots__ = ("index", "pattern", "constraints", "output", "constraint_index")

    def __init__(self, index, pattern, constraints, output, constraint_index=None):
        self.index = index
        self.pattern = pattern            # tuple of tuples of 0/1
        self.constraints = constraints    # list of (w, c): region = {w.x + c > 0 for all}
        self.output = output              # (a, c): network = a.x + c on this region
        # (layer, neuron) -> position in ``constraints`` (absent for constant neurons)
        self.constraint_index = constraint_index if constraint_index is not None else {}

    def __repr__(self):
        a, c = self.output
        return "ActivationRegion(%d, pattern=%s, f(x)=%s)" % (
            self.index, self.pattern, " + ".join(["%s*x%d" % (fmt(v), i) for i, v in enumerate(a)] + [fmt(c)]))


def _hyperplane_key(w, c):
    """Canonical key of the hyperplane {w.x + c = 0}, independent of scaling/orientation."""
    for v in w:
        if v != 0:
            inv = Fraction(1) / v
            return tuple(check(u * inv) for u in w) + (check(c * inv),)
    raise ValueError("zero normal vector")


class RegionCounter:
    def __init__(self, network, max_regions=None, verbose=False):
        self.net = network
        self.max_regions = max_regions
        self.verbose = verbose
        self.regions = None
        self.lp_calls = 0

    # ------------------------------------------------------------------
    def activation_regions(self):
        """Enumerate all full-dimensional activation regions (cached)."""
        if self.regions is not None:
            return self.regions
        n = self.net.input_dim
        depth_needed = sum(self.net.hidden_widths) + self.net.num_hidden_layers + 50
        if sys.getrecursionlimit() < depth_needed:
            sys.setrecursionlimit(depth_needed)
        self.regions = []
        identity = [tuple(Fraction(1 if i == j else 0) for j in range(n)) for i in range(n)]
        zero = tuple([ZERO] * n)
        self._rec_layer(0, identity, zero, [], ())
        return self.regions

    def _affine_layer(self, k, M, v):
        """Pre-activations of layer k as affine functions of x, given h = M x + v."""
        W, b = self.net.weights[k], self.net.biases[k]
        n = self.net.input_dim
        A, d = [], []
        for row, bk in zip(W, b):
            a = [ZERO] * n
            const = bk
            for wij, Mj, vj in zip(row, M, v):
                if wij != 0:
                    for t in range(n):
                        a[t] += wij * Mj[t]
                    const += wij * vj
            A.append(tuple(check(x) for x in a))
            d.append(check(const))
        return A, d

    def _rec_layer(self, k, M, v, constraints, pattern):
        A, d = self._affine_layer(k, M, v)
        if k == self.net.num_hidden_layers:          # output layer
            if self.max_regions is not None and len(self.regions) >= self.max_regions:
                raise TooManyRegionsError("more than %d activation regions" % self.max_regions)
            reg = ActivationRegion(len(self.regions), pattern, [c for _, c in constraints], (A[0], d[0]),
                                   {key: j for j, (key, _) in enumerate(constraints) if key is not None})
            self.regions.append(reg)
            if self.verbose and len(self.regions) % 1000 == 0:
                print("  ... %d activation regions so far" % len(self.regions), file=sys.stderr)
            return
        self._rec_neuron(k, 0, A, d, [], constraints, pattern)

    def _rec_neuron(self, k, i, A, d, signs, constraints, pattern):
        n = self.net.input_dim
        if i == len(A):
            zero = tuple([ZERO] * n)
            M = [A[j] if signs[j] else zero for j in range(len(A))]
            v = tuple(d[j] if signs[j] else ZERO for j in range(len(A)))
            self._rec_layer(k + 1, M, v, constraints, pattern + (tuple(signs),))
            return
        w, c = A[i], d[i]
        if all(x == 0 for x in w):
            # constant pre-activation on the current region: no boundary here
            self._rec_neuron(k, i + 1, A, d, signs + [1 if c > 0 else 0], constraints, pattern)
            return
        pos = (w, c)
        neg = (tuple(-x for x in w), -c)
        plain = [c for _, c in constraints]
        self.lp_calls += 1
        pos_ok = open_polyhedron_nonempty(n, plain + [pos])
        if pos_ok:
            self._rec_neuron(k, i + 1, A, d, signs + [1], constraints + [((k, i), pos)], pattern)
            self.lp_calls += 1
            neg_ok = open_polyhedron_nonempty(n, plain + [neg])
        else:
            neg_ok = True    # current region is non-empty, so the other side must be
        if neg_ok:
            self._rec_neuron(k, i + 1, A, d, signs + [0], constraints + [((k, i), neg)], pattern)

    # ------------------------------------------------------------------
    def linear_regions(self, adjacency="facet", generic=False):
        """Return the linear regions as a list of lists of ActivationRegion.

        ``generic=True`` (only with facet adjacency) assumes the network is generic:
        at a generic point of a shared facet exactly one neuron has pre-activation 0,
        so facet-adjacent activation regions differ in exactly one pattern bit.  Only
        such pairs are then tested (each still with an exact LP), which is much faster.
        The default tests every pair sharing a defining hyperplane and is exact for
        arbitrary (possibly degenerate) parameters.
        """
        regions = self.activation_regions()
        n = self.net.input_dim
        parent = list(range(len(regions)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        groups = defaultdict(list)       # same output affine function
        for r in regions:
            groups[r.output].append(r)

        for group in groups.values():
            if len(group) < 2:
                continue
            if adjacency == "touch":
                for a_idx in range(len(group)):
                    for b_idx in range(a_idx + 1, len(group)):
                        ra, rb = group[a_idx], group[b_idx]
                        if find(ra.index) == find(rb.index):
                            continue
                        self.lp_calls += 1
                        if closed_polyhedron_nonempty(n, ra.constraints + rb.constraints):
                            union(ra.index, rb.index)
            elif adjacency == "facet" and generic:
                by_pattern = {r.pattern: r for r in group}
                flat = {r.index: [b for layer in r.pattern for b in layer] for r in group}
                for ra in group:
                    bits = flat[ra.index]
                    pos = 0
                    for k, layer in enumerate(ra.pattern):
                        for i in range(len(layer)):
                            if bits[pos]:          # test each unordered pair once: from the active side
                                flipped = list(ra.pattern)
                                lay = list(layer); lay[i] = 0
                                flipped[k] = tuple(lay)
                                rb = by_pattern.get(tuple(flipped))
                                if rb is not None and find(ra.index) != find(rb.index):
                                    # hyperplane of neuron (k, i): same affine map in both regions
                                    hyp = self._neuron_constraint(ra, k, i)
                                    if hyp is not None:
                                        key = _hyperplane_key(*hyp)
                                        strict = [(w, c) for r in (ra, rb) for (w, c) in r.constraints
                                                  if _hyperplane_key(w, c) != key]
                                        self.lp_calls += 1
                                        if open_polyhedron_nonempty(n, strict, [hyp]):
                                            union(ra.index, rb.index)
                            pos += 1
            elif adjacency == "facet":
                # Regions sharing an (n-1)-face must both have that face's hyperplane among
                # their defining hyperplanes, so only pairs sharing a hyperplane are candidates.
                by_plane = defaultdict(list)
                keys = {}
                for r in group:
                    seen = set()
                    for (w, c) in r.constraints:
                        key = _hyperplane_key(w, c)
                        keys[(r.index, w, c)] = key
                        if key not in seen:
                            seen.add(key)
                            by_plane[key].append(r)
                for key, members in by_plane.items():
                    if len(members) < 2:
                        continue
                    hyper = (key[:-1], key[-1])          # representative equation w.x + c = 0
                    for a_idx in range(len(members)):
                        for b_idx in range(a_idx + 1, len(members)):
                            ra, rb = members[a_idx], members[b_idx]
                            if find(ra.index) == find(rb.index):
                                continue
                            strict = [(w, c) for r in (ra, rb) for (w, c) in r.constraints
                                      if keys[(r.index, w, c)] != key]
                            self.lp_calls += 1
                            if open_polyhedron_nonempty(n, strict, [hyper]):
                                union(ra.index, rb.index)
            else:
                raise ValueError("adjacency must be 'facet' or 'touch'")

        comps = defaultdict(list)
        for r in regions:
            comps[find(r.index)].append(r)
        return [comps[k] for k in sorted(comps)]

    def _neuron_constraint(self, region, k, i):
        """The signed constraint of neuron i of hidden layer k in ``region`` (None if constant there)."""
        idx = region.constraint_index.get((k, i))
        return None if idx is None else region.constraints[idx]

    def count(self, adjacency="facet", generic=False):
        """Return (number of activation regions, number of linear regions)."""
        acts = self.activation_regions()
        lins = self.linear_regions(adjacency, generic)
        return len(acts), len(lins)
