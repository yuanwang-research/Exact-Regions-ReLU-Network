#!/usr/bin/env python3
"""Exact local-stability certificate for the 2-(2,3,1)-1 experiment.

The first hidden affine layer is normalised to ``x -> x``.  This loses no
region-count information: every full-rank 2-by-2 first affine layer is an
invertible affine change of input coordinates.  The parameters of the two
remaining hidden layers are treated as indeterminates.

For every prefix activation pattern, each hidden preactivation is an affine
form in the input,

    a(theta) * x + b(theta) * y + c(theta).

Put the coefficient triples of *all* these forms, together with the line at
infinity (0, 0, 1), in one rank-three vector configuration.  Its oriented
matroid determines all strict-half-plane feasibility and relative-facet
feasibility questions used to count activation and facet-connected linear
regions.  The oriented matroid can change only when a 3-by-3 determinant of
coefficient triples changes sign.  Determinants that vanish identically are
structural incidences and persist under perturbation.

This script checks, exactly, that every non-identically-zero determinant is
nonzero at every sign flip of the supplied magnitude vector.  It also checks
that two symbolically distinct output affine forms never coincide on two
simultaneously feasible activation regions.  Because there are finitely many
polynomials and sign patterns, passing proves that one common open box around
the magnitude vector preserves every activation region, output-map equality,
facet adjacency, and hence both region counts for every sign configuration.

SymPy is used only to decide polynomial identities.  Values at the supplied
rational magnitudes are checked exactly.  A modular vectorised pass screens
the determinants; every modular zero is rechecked with ``Fraction`` arithmetic.

Usage:

    python verify_stability2d.py examples/quad_random0.json examples/quad_random1.json
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
    raise SystemExit("verify_stability2d.py requires sympy and numpy") from exc

from exact_regions import ReLUNetwork, RegionCounter


ARCHITECTURE = (2, (2, 3, 1), 1)
PRIME = 1_000_003
INF = (0, 0, 1)
ZERO_AFFINE = (0, 0, 0)


class StabilityFailure(RuntimeError):
    """A non-structural degeneracy was found at a signed magnitude vector."""


def _det3(u, v, w):
    return (u[0] * (v[1] * w[2] - v[2] * w[1])
            - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0]))


def _fraction_mod(v, prime=PRIME):
    v = F(v)
    den = v.denominator % prime
    if den == 0:
        raise ValueError("denominator is divisible by modular screening prime %d" % prime)
    return (v.numerator % prime) * pow(den, -1, prime) % prime


def _sp_rational(v):
    v = F(v)
    return sp.Rational(v.numerator, v.denominator)


def _affine_layer(W, b, inputs, modulus=None):
    """Apply an affine layer to affine scalar inputs (ax, ay, c)."""
    out = []
    for row, bias in zip(W, b):
        coeffs = []
        for t in range(3):
            value = sum(row[j] * inputs[j][t] for j in range(len(row)))
            if t == 2:
                value += bias
            if modulus is not None:
                value %= modulus
            coeffs.append(value)
        out.append(tuple(coeffs))
    return out


def _all_affine_forms(hidden_weights, hidden_biases, modulus=None):
    """Return every prefix preactivation and every full-pattern last activation.

    Records have deterministic order.  ``line_records`` entries are
    ``(layer, prefix_pattern, neuron, affine_form)`` and ``outputs`` entries are
    ``(full_pattern, affine_form)``.
    """
    states = [((), ((1, 0, 0), (0, 1, 0)))]
    line_records = []
    for k, (W, b) in enumerate(zip(hidden_weights, hidden_biases)):
        next_states = []
        for prefix, inputs in states:
            pre = _affine_layer(W, b, inputs, modulus)
            for i, form in enumerate(pre):
                line_records.append((k, prefix, i, form))
            for bits in itertools.product((0, 1), repeat=len(pre)):
                activated = tuple(pre[i] if bits[i] else ZERO_AFFINE for i in range(len(pre)))
                next_states.append((prefix + (tuple(bits),), activated))
        states = next_states
    if any(len(values) != 1 for _, values in states):
        raise ValueError("stability verifier requires final hidden width 1")
    return line_records, [(pattern, values[0]) for pattern, values in states]


def _validate_architecture(net):
    got = (net.input_dim, tuple(net.hidden_widths), len(net.weights[-1]))
    if got != ARCHITECTURE:
        raise ValueError("expected architecture input 2, hidden (2,3,1), scalar output; got %r" % (got,))
    W0 = [[abs(v) for v in row] for row in net.weights[0]]
    if W0[0][0] * W0[1][1] - W0[0][1] * W0[1][0] == 0:
        raise ValueError("the entrywise-magnitude first layer must have full rank")
    if net.weights[-1][0][0] == 0:
        raise ValueError("output weight must be nonzero")


def hidden_parameter_positions(net):
    """The 13 sign-varying parameters that affect activation geometry."""
    pos = []
    for k in range(1, net.num_hidden_layers):
        for i, row in enumerate(net.weights[k]):
            for j in range(len(row)):
                pos.append(("w", k, i, j))
        for i in range(len(net.biases[k])):
            pos.append(("b", k, i, 0))
    return pos


def _normalised_hidden_layers(net, values):
    """Build hidden layers with identity first layer and supplied later values."""
    positions = hidden_parameter_positions(net)
    if len(values) != len(positions):
        raise ValueError("wrong number of hidden parameter values")
    lookup = {p: values[i] for i, p in enumerate(positions)}
    W = [((1, 0), (0, 1))]
    b = [(0, 0)]
    for k in range(1, net.num_hidden_layers):
        W.append(tuple(tuple(lookup[("w", k, i, j)] for j in range(len(row)))
                       for i, row in enumerate(net.weights[k])))
        b.append(tuple(lookup[("b", k, i, 0)] for i in range(len(net.biases[k]))))
    return W, b


def _symbolic_family(net):
    positions = hidden_parameter_positions(net)
    symbols = sp.symbols("p0:%d" % len(positions))
    W, b = _normalised_hidden_layers(net, symbols)
    records, outputs = _all_affine_forms(W, b)
    vectors = [tuple(sp.expand(v) for v in rec[3]) for rec in records] + [INF]

    triples = []
    digest = hashlib.sha256()
    for i, j, k in itertools.combinations(range(len(vectors)), 3):
        expr = sp.Poly(sp.expand(_det3(vectors[i], vectors[j], vectors[k])), *symbols, domain=sp.QQ)
        digest.update(repr(expr.terms()).encode("ascii"))
        digest.update(b"\n")
        if not expr.is_zero:
            triples.append((i, j, k))

    # Symbolic equality classes of the final hidden activation's affine form.
    def poly_key(expr):
        return tuple(sp.Poly(sp.expand(expr), *symbols, domain=sp.QQ).terms())

    class_by_key = {}
    pattern_class = {}
    representatives = []
    for out_index, (pattern, form) in enumerate(outputs):
        key = tuple(poly_key(v) for v in form)
        if key not in class_by_key:
            class_by_key[key] = len(representatives)
            representatives.append(out_index)
        pattern_class[pattern] = class_by_key[key]
    return {
        "positions": positions,
        "symbols": symbols,
        "records": records,
        "outputs": outputs,
        "vectors": vectors,
        "triples": triples,
        "pattern_class": pattern_class,
        "output_representatives": representatives,
        "digest": digest.hexdigest(),
    }


def _signed_values(net, config, converter=lambda x: x):
    values = []
    for j, (kind, k, i, col) in enumerate(hidden_parameter_positions(net)):
        value = net.weights[k][i][col] if kind == "w" else net.biases[k][i]
        value = abs(value)
        if value == 0:
            raise ValueError("hidden sign-flipped parameter %r is zero" % ((kind, k, i, col),))
        if config & (1 << j):
            value = -value
        values.append(converter(value))
    return values


def _numeric_forms(net, config, converter=lambda x: x, modulus=None):
    values = _signed_values(net, config, converter)
    W, b = _normalised_hidden_layers(net, values)
    return _all_affine_forms(W, b, modulus)


def _signed_network(net, config):
    values = _signed_values(net, config)
    hidden_W, hidden_b = _normalised_hidden_layers(net, values)
    out_weight = abs(net.weights[-1][0][0])
    out_bias = abs(net.biases[-1][0])
    weights = [[list(row) for row in W] for W in hidden_W] + [[[out_weight]]]
    biases = [list(row) for row in hidden_b] + [[out_bias]]
    return ReLUNetwork(weights, biases)


def _describe_config(config, positions):
    neg = []
    for j, (kind, k, i, col) in enumerate(positions):
        if config & (1 << j):
            neg.append("%s%d[%d,%d]" % (kind, k, i, col))
    return "negative={%s}" % ", ".join(neg)


def _modular_determinants(lines, I, J, K, prime=PRIME):
    arr = np.asarray(lines, dtype=np.int64)
    u, v, w = arr[I], arr[J], arr[K]
    m0 = (v[:, 1] * w[:, 2] - v[:, 2] * w[:, 1]) % prime
    m1 = (v[:, 0] * w[:, 2] - v[:, 2] * w[:, 0]) % prime
    m2 = (v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]) % prime
    return (u[:, 0] * m0 - u[:, 1] * m1 + u[:, 2] * m2) % prime


def certify(net, progress=False, family=None):
    """Certify a common stable neighborhood; return certificate statistics."""
    _validate_architecture(net)
    family = _symbolic_family(net) if family is None else family
    triples = family["triples"]
    I = np.asarray([t[0] for t in triples], dtype=np.int64)
    J = np.asarray([t[1] for t in triples], dtype=np.int64)
    K = np.asarray([t[2] for t in triples], dtype=np.int64)
    reps = family["output_representatives"]
    rep_pairs = list(itertools.combinations(range(len(reps)), 2))
    RP = np.asarray([p[0] for p in rep_pairs], dtype=np.int64)
    RQ = np.asarray([p[1] for p in rep_pairs], dtype=np.int64)
    num_configs = 1 << len(family["positions"])
    modular_fallbacks = 0
    output_fallbacks = 0
    t0 = time.time()

    for config in range(num_configs):
        mod_records, mod_outputs = _numeric_forms(
            net, config, converter=lambda v: _fraction_mod(v, PRIME), modulus=PRIME)
        mod_lines = [rec[3] for rec in mod_records] + [INF]
        residues = _modular_determinants(mod_lines, I, J, K)
        suspects = np.flatnonzero(residues == 0)
        exact_records = exact_outputs = None
        if len(suspects):
            modular_fallbacks += len(suspects)
            exact_records, exact_outputs = _numeric_forms(net, config)
            exact_lines = [rec[3] for rec in exact_records] + [INF]
            for index in suspects:
                triple = triples[int(index)]
                if _det3(*(exact_lines[q] for q in triple)) == 0:
                    labels = []
                    for q in triple:
                        labels.append("infinity" if q == len(exact_records) else exact_records[q][:3])
                    raise StabilityFailure(
                        "non-structural line determinant vanishes for %s; triple=%r"
                        % (_describe_config(config, family["positions"]), labels))

        # A modular equality between distinct symbolic output classes is rare;
        # when it occurs, ask RegionCounter whether two feasible classes really
        # have the same exact output map.
        if len(rep_pairs):
            mod_out = np.asarray([mod_outputs[q][1] for q in reps], dtype=np.int64)
            collisions = np.all(mod_out[RP] == mod_out[RQ], axis=1)
            if np.any(collisions):
                output_fallbacks += 1
                signed_net = _signed_network(net, config)
                groups = {}
                for region in RegionCounter(signed_net).activation_regions():
                    cls = family["pattern_class"][region.pattern]
                    previous = groups.setdefault(region.output, cls)
                    if previous != cls:
                        raise StabilityFailure(
                            "symbolically distinct output maps coincide on feasible regions for %s; "
                            "classes %d and %d" %
                            (_describe_config(config, family["positions"]), previous, cls))

        if progress and ((config + 1) % 1024 == 0 or config + 1 == num_configs):
            print("  checked %d/%d geometry sign patterns (%.1fs)" %
                  (config + 1, num_configs, time.time() - t0), file=sys.stderr)

    return {
        "geometry_parameters": len(family["positions"]),
        "geometry_sign_patterns": num_configs,
        "affine_forms": len(family["vectors"]) - 1,
        "all_determinants": math.comb(len(family["vectors"]), 3),
        "structural_zero_determinants": math.comb(len(family["vectors"]), 3) - len(triples),
        "nonstructural_determinants": len(triples),
        "output_symbolic_classes": len(reps),
        "modular_exact_fallbacks": modular_fallbacks,
        "output_feasibility_fallbacks": output_fallbacks,
        "symbolic_digest": family["digest"],
        "seconds": time.time() - t0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="magnitude-vector network JSON files")
    parser.add_argument("--quiet", action="store_true", help="suppress progress updates")
    args = parser.parse_args(argv)
    failed = False
    family = None
    for path in args.files:
        print("certifying %s" % path)
        try:
            net = ReLUNetwork.from_json(path)
            _validate_architecture(net)
            if family is None:
                family = _symbolic_family(net)
            result = certify(net, progress=not args.quiet, family=family)
        except (StabilityFailure, ValueError) as exc:
            failed = True
            print("  FAIL: %s" % exc)
            continue
        print("  PASS: all %d geometry sign patterns are locally stable" %
              result["geometry_sign_patterns"])
        print("  (the output-weight sign gives all %d experiment sign patterns)" %
              (2 * result["geometry_sign_patterns"]))
        print("  %d affine boundary forms; %d determinant identities structural, %d checked nonzero"
              % (result["affine_forms"], result["structural_zero_determinants"],
                 result["nonstructural_determinants"]))
        print("  %d symbolic output-map classes; digest %s"
              % (result["output_symbolic_classes"], result["symbolic_digest"][:16]))
        print("  exact fallbacks: determinants %d, output feasibility %d; %.1fs"
              % (result["modular_exact_fallbacks"], result["output_feasibility_fallbacks"],
                 result["seconds"]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
