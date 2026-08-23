#!/usr/bin/env python3
"""Exact sign-flip averages of region counts (see exact_regions/signflip.py).

Random magnitudes:
    python sign_flip_experiment.py -d 1 --widths 2,2 --draws 3 --seed 0 --jobs 6
Magnitudes from a file, first layer fixed (valid when d >= first width):
    python sign_flip_experiment.py --from-file examples/quad_ring.json --fix-first-layer --jobs 6
"""
import argparse
import random
import sys
import time
from fractions import Fraction

from exact_regions import ReLUNetwork, sign_flip_averages
from exact_regions.numbers import fmt


def draw_params(rng, d, widths, denominator):
    """Random non-zero magnitudes in (0, 1] with the given denominator."""
    def r():
        return Fraction(rng.randint(1, denominator), denominator)
    dims = [d] + widths + [1]
    W = [[[r() for _ in range(dims[k])] for _ in range(dims[k + 1])] for k in range(len(dims) - 1)]
    b = [[r() for _ in range(dims[k + 1])] for k in range(len(dims) - 1)]
    return W, b


def identity_first_layer(W, b, d):
    W[0] = [[Fraction(int(i == j)) for j in range(d)] for i in range(len(W[0]))]
    b[0] = [Fraction(0)] * len(W[0])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", type=int, default=None, help="input dimension")
    p.add_argument("--widths", default=None, help="hidden widths, e.g. 2,2")
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--denominator", type=int, default=1000)
    p.add_argument("--from-file", default=None, help="take magnitudes from this network JSON (one draw)")
    p.add_argument("--fix-first-layer", action="store_true",
                   help="do not flip the first hidden layer (valid when d >= first width)")
    p.add_argument("--adjacency", default="facet", choices=["facet", "touch"])
    p.add_argument("--generic", action="store_true", help="use the fast generic facet test")
    p.add_argument("--jobs", type=int, default=1)
    a = p.parse_args()
    if a.from_file:
        nets = [ReLUNetwork.from_json(a.from_file)]
        print("magnitudes from %s" % a.from_file)
    else:
        if a.d is None or a.widths is None:
            p.error("give -d and --widths, or --from-file")
        widths = [int(w) for w in a.widths.split(",")]
        rng = random.Random(a.seed)
        nets = []
        for _ in range(a.draws):
            W, b = draw_params(rng, a.d, widths, a.denominator)
            if a.fix_first_layer:
                identity_first_layer(W, b, a.d)
            nets.append(ReLUNetwork(W, b))
    net0 = nets[0]
    print("architecture: input %d, hidden %s, output 1%s"
          % (net0.input_dim, net0.hidden_widths, " (first layer fixed)" if a.fix_first_layer else ""))
    for t, net in enumerate(nets):
        t0 = time.time()
        avg, res, P = sign_flip_averages(net, a.fix_first_layer, a.adjacency, a.generic, a.jobs)
        print("draw %d: %d sign patterns enumerated (2^%d after symmetry): mean activation regions = %s (%.6f), "
              "mean linear regions = %s (%.6f)   [%.1fs]"
              % (t, len(res), P, fmt(avg["activation"]), float(avg["activation"]),
                 fmt(avg["linear"]), float(avg["linear"]), time.time() - t0))
        print("        non-constant linear = %s, constant linear = %s, constant activation = %s"
              % (fmt(avg["nonconst_linear"]), fmt(avg["const_linear"]), fmt(avg["const_activation"])))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
