#!/usr/bin/env python3
"""Exact local-stability certificates for the sign-flip experiments, any architecture.

Setting.  Fix a fully connected ReLU network with rational parameters, input
dimension d and one output.  Every hidden preactivation, restricted to a prefix
activation pattern of the earlier layers, is an affine form  a_1 x_1 + ... +
a_d x_d + c  whose coefficients are polynomials in the parameters.  Collect the
coefficient vectors (a_1, ..., a_d, c) of all these forms, canonical "probe"
vectors for any missing coordinate directions, and the hyperplane at infinity
(0, ..., 0, 1), into one rank-(d+1) vector configuration.

The oriented matroid of that configuration (the signs of all (d+1)x(d+1)
determinants) determines every predicate the exact region counter evaluates:
which activation patterns are realised by full-dimensional regions, and which
pairs share a facet.  Determinants that vanish *identically* in the parameters
are structural and persist under perturbation; all others must be nonzero for
the counts to be locally constant.  Output affine maps are compared through
their symbolic classes; distinct classes must not coincide on two feasible
regions.  Passing all checks proves that one open box around the magnitude
vector preserves both region counts -- for a single signed network
(``certify_single``) or simultaneously for every sign flip (``certify``), in
which case the sign-flip averages are locally constant.

When the input dimension is at least the first hidden width the first layer is
normalised to ``[I | 0]`` and not flipped (an invertible affine change of input
coordinates; region counts do not depend on such a first layer).  Otherwise its
parameters are flipped too, modulo input reflections.  The flipped-parameter
set is exactly ``exact_regions.signflip.flip_positions``, the one used by the
sign-flip experiments.

SymPy decides which determinants are structural; per sign pattern a vectorised
modular screen (prime 1000003) is used and every modular zero is rechecked with
exact ``Fraction`` arithmetic.

Usage:

    python verify_stability.py examples/quad_ring.json            # all sign flips
    python verify_stability.py --single some_network.json         # just that network
"""
import argparse
import hashlib
import itertools
import math
import sys
import time
from fractions import Fraction as F

try:
    import numpy as np
    import sympy as sp
except ImportError as exc:  # pragma: no cover - environment-dependent message
    raise SystemExit("verify_stability.py requires sympy and numpy") from exc

from exact_regions import ReLUNetwork, RegionCounter
from exact_regions.signflip import flip_positions

PRIME = 1_000_003


class StabilityFailure(RuntimeError):
    """A non-structural degeneracy was found at a signed parameter vector."""


class BudgetExceeded(RuntimeError):
    """The projected amount of work exceeds the configured budget."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fraction_mod(v, prime=PRIME):
    v = F(v)
    den = v.denominator % prime
    if den == 0:
        raise ValueError("denominator divisible by the modular screening prime %d" % prime)
    return (v.numerator % prime) * pow(den, -1, prime) % prime


def _first_layer_fixable(net):
    return net.input_dim >= net.hidden_widths[0]


def _validate_structure(net, fix_first_layer):
    if len(net.weights[-1]) != 1:
        raise ValueError("scalar output required")
    if not fix_first_layer and any(v == 0 for v in net.weights[0][0]):
        # The input-reflection quotient anchors the signs of row 0 of W_1; a
        # zero anchor entry has no sign, and the family would bake in an exact
        # zero whose perturbation is not covered by the certificate.
        raise ValueError("first-layer row 0 (the input-reflection anchor) must have no zero entries")


def _validate_rank(net, signed):
    """Full row rank of the first layer: of W_1 itself for a single signed
    network, of |W_1| when certifying all sign flips of the magnitudes."""
    rows = net.weights[0] if signed else [[abs(v) for v in row] for row in net.weights[0]]
    W0 = sp.Matrix([[sp.Rational(F(v)) for v in row] for row in rows])
    if W0.rank() < net.hidden_widths[0]:
        raise ValueError("the %sfirst layer must have full row rank"
                         % ("" if signed else "entrywise-magnitude "))


def _layers_from_values(net, positions, values, fix_first_layer, fixed=lambda v: v):
    """All layer parameters with ``values`` substituted at the flip positions.

    Non-flipped positions keep the entrywise magnitude of ``net``, passed
    through ``fixed`` (identity, exact residue, ...); the first layer is
    replaced by [I | 0], 0 when it is fixed.
    """
    lookup = dict(zip(positions, values))
    d = net.input_dim

    def get(kind, k, i, j, raw):
        try:
            return lookup[(kind, k, i, j)]
        except KeyError:
            return fixed(abs(raw))

    W, b = [], []
    for k in range(len(net.weights)):
        if fix_first_layer and k == 0:
            W.append([[1 if i == j else 0 for j in range(d)] for i in range(net.hidden_widths[0])])
            b.append([0] * net.hidden_widths[0])
            continue
        W.append([[get("w", k, i, j, v) for j, v in enumerate(row)]
                  for i, row in enumerate(net.weights[k])])
        b.append([get("b", k, i, 0, v) for i, v in enumerate(net.biases[k])])
    return W, b


def fixed_fingerprint(net, fix_first_layer=None):
    """The non-flipped parameter magnitudes baked into a symbolic family."""
    if fix_first_layer is None:
        fix_first_layer = _first_layer_fixable(net)
    flipped = set(flip_positions(net, fix_first_layer))
    out = []
    for k in range(len(net.weights)):
        if fix_first_layer and k == 0:
            continue
        for i, row in enumerate(net.weights[k]):
            for j, v in enumerate(row):
                if ("w", k, i, j) not in flipped:
                    out.append((("w", k, i, j), abs(F(v))))
    return tuple(out)


def _forms_and_outputs(d, W, b, modulus=None):
    """Every prefix preactivation form and every full-pattern output map.

    Returns (records, outputs): records are (layer, prefix, neuron, form) in a
    deterministic order; outputs are (pattern, output affine map), the map being
    W_out . activated_last_layer + b_out.
    """
    zero = tuple([0] * (d + 1))
    identity = tuple(tuple(1 if t == i else 0 for t in range(d + 1)) for i in range(d))
    states = [((), identity)]
    records = []
    n_hidden = len(W) - 1
    for k in range(n_hidden):
        next_states = []
        for prefix, inputs in states:
            pre = []
            for row, bias in zip(W[k], b[k]):
                coeffs = []
                for t in range(d + 1):
                    value = sum(row[j] * inputs[j][t] for j in range(len(row)))
                    if t == d:
                        value += bias
                    if modulus is not None:
                        value %= modulus
                    coeffs.append(value)
                pre.append(tuple(coeffs))
            for i, form in enumerate(pre):
                records.append((k, prefix, i, form))
            for bits in itertools.product((0, 1), repeat=len(pre)):
                activated = tuple(pre[i] if bits[i] else zero for i in range(len(pre)))
                next_states.append((prefix + (tuple(bits),), activated))
        states = next_states
    outputs = []
    for pattern, inputs in states:
        coeffs = []
        for t in range(d + 1):
            value = sum(W[n_hidden][0][j] * inputs[j][t] for j in range(len(inputs)))
            if t == d:
                value += b[n_hidden][0]
            if modulus is not None:
                value %= modulus
            coeffs.append(value)
        outputs.append((pattern, tuple(coeffs)))
    return records, outputs


def _probes(d, records):
    """Canonical vectors missing from the configuration, plus the one at infinity."""
    present = {tuple(form) for (k, _, _, form) in records if k == 0}
    probes = []
    for i in range(d):
        e = tuple(1 if t == i else 0 for t in range(d + 1))
        if e not in present:
            probes.append(e)
    probes.append(tuple([0] * d + [1]))          # line/hyperplane at infinity
    return probes


# ---------------------------------------------------------------------------
# symbolic family: everything that depends only on the architecture
# ---------------------------------------------------------------------------

def symbolic_family(net, fix_first_layer=None):
    if fix_first_layer is None:
        fix_first_layer = _first_layer_fixable(net)
    _validate_structure(net, fix_first_layer)
    d = net.input_dim
    positions = flip_positions(net, fix_first_layer)
    symbols = sp.symbols("p0:%d" % len(positions))
    W, b = _layers_from_values(net, positions, symbols, fix_first_layer,
                               fixed=lambda v: sp.Rational(F(v)))
    records, outputs = _forms_and_outputs(d, W, b)
    vectors = ([tuple(sp.expand(v) for v in form) for (_, _, _, form) in records]
               + _probes(d, records))

    triples = []
    digest = hashlib.sha256()
    for subset in itertools.combinations(range(len(vectors)), d + 1):
        det = sp.Matrix([vectors[q] for q in subset]).det(method="berkowitz")
        poly = sp.Poly(sp.expand(det), *symbols, domain=sp.QQ)
        digest.update(repr(poly.terms()).encode("ascii"))
        digest.update(b"\n")
        if not poly.is_zero:
            triples.append(subset)

    def poly_key(expr):
        return tuple(sp.Poly(sp.expand(expr), *symbols, domain=sp.QQ).terms())

    class_by_key = {}
    pattern_class = {}
    representatives = []
    for index, (pattern, form) in enumerate(outputs):
        key = tuple(poly_key(v) for v in form)
        if key not in class_by_key:
            class_by_key[key] = len(representatives)
            representatives.append(index)
        pattern_class[pattern] = class_by_key[key]

    return {
        "d": d,
        "fix_first_layer": fix_first_layer,
        "positions": positions,
        "symbols": symbols,
        "records": records,
        "num_vectors": len(vectors),
        "num_probes": len(vectors) - len(records),
        "subsets": triples,
        "pattern_class": pattern_class,
        "output_representatives": representatives,
        "digest": digest.hexdigest(),
        "all_determinants": math.comb(len(vectors), d + 1),
        "fingerprint": fixed_fingerprint(net, fix_first_layer),
    }


def _require_family_match(net, family):
    """A supplied family must have been built for this network's architecture,
    flipped-parameter set, and fixed magnitudes -- otherwise its symbolic
    classification does not apply."""
    fix = family["fix_first_layer"]
    _validate_structure(net, fix)
    if family["positions"] != flip_positions(net, fix):
        raise ValueError("family does not match this network's architecture")
    if family["fingerprint"] != fixed_fingerprint(net, fix):
        raise ValueError("family was built for different fixed (non-flipped) parameter magnitudes")


# ---------------------------------------------------------------------------
# per-sign-pattern checks
# ---------------------------------------------------------------------------

def _magnitude_values(net, positions, config):
    values = []
    for j, (kind, k, i, col) in enumerate(positions):
        v = abs(net.weights[k][i][col] if kind == "w" else net.biases[k][i])
        if v == 0:
            raise ValueError("flipped parameter %r is zero" % ((kind, k, i, col),))
        values.append(-v if config & (1 << j) else v)
    return values


def _describe(config, positions):
    neg = ["%s%d[%d,%d]" % (kind, k, i, col)
           for j, (kind, k, i, col) in enumerate(positions) if config & (1 << j)]
    return "negative={%s}" % ", ".join(neg)


def _signature(perm):
    inversions = sum(1 for a in range(len(perm)) for b in range(a + 1, len(perm))
                     if perm[a] > perm[b])
    return -1 if inversions % 2 else 1


def _signed_permutations(k):
    return [(_signature(p), p) for p in itertools.permutations(range(k))]


def _det_batches(family):
    subsets = family["subsets"]
    k = family["d"] + 1
    IDX = np.asarray(subsets, dtype=np.int64) if subsets else np.zeros((0, k), np.int64)
    return IDX, _signed_permutations(k)


def _modular_dets(lines, IDX, perms, prime=PRIME):
    arr = np.asarray(lines, dtype=np.int64)
    rows = arr[IDX]                              # (subsets, k, k)
    total = np.zeros(len(IDX), dtype=np.int64)
    for sign, p in perms:
        prod = np.ones(len(IDX), dtype=np.int64)
        for i, col in enumerate(p):
            prod = (prod * rows[:, i, col]) % prime
        total = (total + sign * prod) % prime
    return total


def _det_exact(vectors):
    total = F(0)
    for sign, perm in _signed_permutations(len(vectors)):
        prod = F(1)
        for i, col in enumerate(perm):
            prod *= F(vectors[i][col])
        total += sign * prod
    return total


class _ConfigChecker:
    """Checks one signed parameter vector against the symbolic family."""

    def __init__(self, net, family):
        self.net = net
        self.family = family
        self.IDX, self.perms = _det_batches(family)
        reps = family["output_representatives"]
        pairs = list(itertools.combinations(range(len(reps)), 2))
        self.RP = np.asarray([p[0] for p in pairs], dtype=np.int64)
        self.RQ = np.asarray([p[1] for p in pairs], dtype=np.int64)
        self.modular_fallbacks = 0
        self.output_fallbacks = 0

    def _forms(self, config, converter, modulus):
        values = [converter(v) for v in
                  _magnitude_values(self.net, self.family["positions"], config)]
        W, b = _layers_from_values(self.net, self.family["positions"], values,
                                   self.family["fix_first_layer"], fixed=converter)
        d = self.family["d"]
        records, outputs = _forms_and_outputs(d, W, b, modulus)
        lines = [form for (_, _, _, form) in records] + _probes(d, records)
        return records, outputs, lines

    def check(self, config):
        family = self.family
        mod_records, mod_outputs, mod_lines = self._forms(config, _fraction_mod, PRIME)
        residues = _modular_dets(mod_lines, self.IDX, self.perms)
        suspects = np.flatnonzero(residues == 0)
        if len(suspects):
            self.modular_fallbacks += len(suspects)
            ex_records, _, ex_lines = self._forms(config, F, None)
            for index in suspects:
                subset = family["subsets"][int(index)]
                if _det_exact([ex_lines[q] for q in subset]) == 0:
                    labels = ["probe/infinity" if q >= len(ex_records) else ex_records[q][:3]
                              for q in subset]
                    raise StabilityFailure(
                        "non-structural determinant vanishes for %s; forms=%r"
                        % (_describe(config, family["positions"]), labels))

        reps = family["output_representatives"]
        if len(reps) > 1:
            mod_out = np.asarray([mod_outputs[q][1] for q in reps], dtype=np.int64)
            if np.any(np.all(mod_out[self.RP] == mod_out[self.RQ], axis=1)):
                self.output_fallbacks += 1
                values = _magnitude_values(self.net, family["positions"], config)
                W, b = _layers_from_values(self.net, family["positions"], values,
                                           family["fix_first_layer"], fixed=lambda v: F(v))
                signed = ReLUNetwork(W, b)
                groups = {}
                for region in RegionCounter(signed).activation_regions():
                    cls = family["pattern_class"][region.pattern]
                    previous = groups.setdefault(region.output, cls)
                    if previous != cls:
                        raise StabilityFailure(
                            "symbolically distinct output maps coincide on feasible regions "
                            "for %s; classes %d and %d"
                            % (_describe(config, family["positions"]), previous, cls))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def estimate_work(family):
    return (1 << len(family["positions"])) * len(family["subsets"])


def certify(net, family=None, fix_first_layer=None, budget=None, progress=False):
    """Certify a common stable neighborhood for every sign flip of |net|.

    Only the entrywise magnitudes of ``net`` are used; correspondingly the
    (fixed) first layer must have full row rank *as a magnitude matrix* when it
    is normalised away.
    """
    if family is None:
        family = symbolic_family(net, fix_first_layer)
    _require_family_match(net, family)
    if family["fix_first_layer"]:
        _validate_rank(net, signed=False)
    work = estimate_work(family)
    if budget is not None and work > budget:
        raise BudgetExceeded(
            "projected work %.3g determinant evaluations exceeds budget %.3g "
            "(2^%d sign patterns x %d determinants)"
            % (work, budget, len(family["positions"]), len(family["subsets"])))
    checker = _ConfigChecker(net, family)
    num_configs = 1 << len(family["positions"])
    t0 = time.time()
    for config in range(num_configs):
        checker.check(config)
        if progress and ((config + 1) % 1024 == 0 or config + 1 == num_configs):
            print("  checked %d/%d sign patterns (%.1fs)"
                  % (config + 1, num_configs, time.time() - t0), file=sys.stderr)
    return _stats(family, checker, num_configs, time.time() - t0)


def certify_single(net, family=None, fix_first_layer=None):
    """Certify local stability of the one signed network given (signs taken from it)."""
    if family is None:
        family = symbolic_family(net, fix_first_layer)
    _require_family_match(net, family)
    if family["fix_first_layer"]:
        _validate_rank(net, signed=True)
        colsign = None
    else:
        # Map the network to its input-reflection representative: reflecting
        # x_j -> -x_j flips *column* j of W_1 (biases and deeper layers are
        # unchanged), so normalising the anchor row to positive signs must
        # carry the rest of each column along.
        colsign = [1 if v > 0 else -1 for v in net.weights[0][0]]
    config = 0
    for j, (kind, k, i, col) in enumerate(family["positions"]):
        v = net.weights[k][i][col] if kind == "w" else net.biases[k][i]
        if colsign is not None and kind == "w" and k == 0:
            v = v * colsign[col]
        if v < 0:
            config |= 1 << j
    checker = _ConfigChecker(net, family)
    t0 = time.time()
    checker.check(config)
    return _stats(family, checker, 1, time.time() - t0)


def _stats(family, checker, num_configs, seconds):
    return {
        "flipped_parameters": len(family["positions"]),
        "sign_patterns": num_configs,
        "affine_forms": len(family["records"]),
        "probes": family["num_probes"],
        "all_determinants": family["all_determinants"],
        "structural_zero_determinants": family["all_determinants"] - len(family["subsets"]),
        "nonstructural_determinants": len(family["subsets"]),
        "output_symbolic_classes": len(family["output_representatives"]),
        "modular_exact_fallbacks": checker.modular_fallbacks,
        "output_feasibility_fallbacks": checker.output_fallbacks,
        "symbolic_digest": family["digest"],
        "seconds": seconds,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="network JSON files")
    parser.add_argument("--single", action="store_true",
                        help="certify each file as one signed network instead of all sign flips")
    parser.add_argument("--budget", type=float, default=2e10,
                        help="refuse if projected determinant evaluations exceed this (default 2e10)")
    parser.add_argument("--quiet", action="store_true", help="suppress progress updates")
    args = parser.parse_args(argv)
    failed = False
    families = {}
    for path in args.files:
        print("certifying %s%s" % (path, " (single network)" if args.single else ""))
        try:
            net = ReLUNetwork.from_json(path)
            key = (net.input_dim, tuple(net.hidden_widths), fixed_fingerprint(net))
            if key not in families:
                families[key] = symbolic_family(net)
            family = families[key]
            if args.single:
                result = certify_single(net, family=family)
            else:
                result = certify(net, family=family, budget=args.budget,
                                 progress=not args.quiet)
        except (StabilityFailure, BudgetExceeded, ValueError) as exc:
            failed = True
            print("  FAIL: %s" % exc)
            continue
        print("  PASS: %d sign pattern(s) of %d parameters locally stable"
              % (result["sign_patterns"], result["flipped_parameters"]))
        print("  %d affine forms (+%d probe vectors); %d determinants structural, %d checked nonzero"
              % (result["affine_forms"], result["probes"],
                 result["structural_zero_determinants"], result["nonstructural_determinants"]))
        print("  %d symbolic output-map classes; digest %s"
              % (result["output_symbolic_classes"], result["symbolic_digest"][:16]))
        print("  exact fallbacks: determinants %d, output feasibility %d; %.1fs"
              % (result["modular_exact_fallbacks"], result["output_feasibility_fallbacks"],
                 result["seconds"]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
