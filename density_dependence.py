#!/usr/bin/env python3
"""Replicate the density-dependence example: input 2, hidden widths (2, 3, 1), output 1.

For each magnitude file the exact sign-flip average of several region statistics
is computed by enumerating all 2^14 sign patterns of layers 2..4 (the first
layer is fixed to the identity, which is allowed because input dim >= width 1).
Different magnitude files giving different averages of "linear" prove that the
expected number of linear regions depends on the parameter densities, even
though the averages of "activation" and "nonconst_linear" coincide.

    python density_dependence.py --jobs 6            # ~5 min per file
    python density_dependence.py --jobs 6 --verify   # + LP-free cross-check of every network (2-D only)

Output: a markdown table on stdout and in results/density_dependence.md.
"""
import argparse
import sys
import time
from fractions import Fraction

from exact_regions import ReLUNetwork, sign_flip_averages
from exact_regions.numbers import fmt

DEFAULT_FILES = ["examples/quad_ring.json", "examples/quad_axis.json",
                 "examples/quad_random0.json", "examples/quad_random1.json", "examples/quad_random2.json"]
COLUMNS = ["activation", "nonconst_linear", "const_linear", "linear", "const_activation", "distinct_const_maps"]


def verify_extra(net):
    from verify2d import count_2d
    a, l, _ = count_2d(net)
    return {"verify_activation": a, "verify_linear": l}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", default=DEFAULT_FILES)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--verify", action="store_true", help="cross-check every network with verify2d (input dim 2 only)")
    p.add_argument("--out", default="results/density_dependence.md")
    a = p.parse_args()

    rows = []
    for f in a.files:
        net = ReLUNetwork.from_json(f)
        t0 = time.time()
        print("enumerating %s ..." % f, file=sys.stderr)
        avg, res, P = sign_flip_averages(net, fix_first_layer=True, jobs=a.jobs,
                                         extra=verify_extra if a.verify else None)
        row = {"file": f, "P": P, "N": len(res), "time": time.time() - t0}
        row.update(avg)
        if a.verify:
            row["mismatches"] = sum(1 for _, s in res
                                    if (s["activation"], s["linear"]) != (s["verify_activation"], s["verify_linear"]))
        rows.append(row)
        print("  done in %.0fs: linear = %s" % (row["time"], fmt(avg["linear"])), file=sys.stderr)

    net0 = ReLUNetwork.from_json(a.files[0])
    lines = ["# Sign-flip averages, input %d, hidden %s, first layer fixed to the identity"
             % (net0.input_dim, net0.hidden_widths), "",
             "Exact averages over all 2^P sign patterns of layers 2..%d (P = %d)." % (len(net0.weights), rows[0]["P"]), ""]
    head = ["magnitudes"] + COLUMNS + (["mismatches"] if a.verify else [])
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for r in rows:
        cells = [r["file"]] + ["%s (%.6f)" % (fmt(r[c]), float(r[c])) for c in COLUMNS]
        if a.verify:
            cells.append(str(r["mismatches"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Columns: activation = activation regions; nonconst_linear / const_linear = linear regions on which the",
              "network is non-constant / constant; linear = their sum; const_activation = activation regions with constant map;",
              "distinct_const_maps = distinct constant values realised." + (
                  "  mismatches = networks where the independent polygon counter disagrees." if a.verify else "")]
    text = "\n".join(lines)
    print(text)
    with open(a.out, "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
