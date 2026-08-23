#!/usr/bin/env python3
"""Count activation regions and linear regions of a ReLU network, exactly.

Usage:
    python count_regions.py network.json [--adjacency facet|touch] [--list] [--max-regions N] [--max-bits B]

network.json has the form
    {"weights": [W_1, ..., W_l, W_out], "biases": [b_1, ..., b_l, b_out]}
where W_k is a list of rows and every entry is an integer, a decimal such as
0.25 (kept exact), or a string fraction such as "-1/3".
"""
import argparse
import sys
import time

from exact_regions import ReLUNetwork, RegionCounter, set_max_bits, PrecisionOverflowError, TooManyRegionsError
from exact_regions.numbers import fmt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("network", help="JSON file with weights and biases")
    p.add_argument("--adjacency", choices=["facet", "touch"], default="facet",
                   help="how activation regions with equal affine map are glued into linear regions (default facet)")
    p.add_argument("--list", action="store_true", help="print every activation region and linear region")
    p.add_argument("--generic", action="store_true",
                   help="assume generic parameters: only test pattern pairs differing in one neuron for facet adjacency (faster)")
    p.add_argument("--max-regions", type=int, default=None, help="abort if more activation regions than this")
    p.add_argument("--max-bits", type=int, default=None,
                   help="abort if any exact number exceeds this many bits (default 100000)")
    args = p.parse_args(argv)

    if args.max_bits is not None:
        set_max_bits(args.max_bits)
    net = ReLUNetwork.from_json(args.network)
    print("input dimension n = %d, hidden widths = %s, hidden layers l = %d"
          % (net.input_dim, net.hidden_widths, net.num_hidden_layers))
    counter = RegionCounter(net, max_regions=args.max_regions, verbose=True)
    t0 = time.time()
    try:
        acts = counter.activation_regions()
        lins = counter.linear_regions(args.adjacency, args.generic)
    except PrecisionOverflowError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    except TooManyRegionsError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 3
    dt = time.time() - t0
    if args.list:
        print("\nActivation regions:")
        for r in acts:
            a, c = r.output
            f = " + ".join(["%s*x%d" % (fmt(v), i + 1) for i, v in enumerate(a)] + [fmt(c)])
            print("  #%d pattern=%s  f(x) = %s" % (r.index, r.pattern, f))
        print("\nLinear regions (as unions of activation regions):")
        for i, comp in enumerate(lins):
            print("  L%d: %s" % (i, [r.index for r in comp]))
        print()
    print("activation regions: %d" % len(acts))
    print("linear regions (%s adjacency): %d" % (args.adjacency, len(lins)))
    print("(%d exact LPs solved, %.2fs)" % (counter.lp_calls, dt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
