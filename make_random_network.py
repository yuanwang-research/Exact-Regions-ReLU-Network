#!/usr/bin/env python3
"""Generate a random ReLU network with exact rational parameters as JSON.

    python make_random_network.py -n 2 -l 2 --seed 1 --denominator 100 -o examples/rand.json
    python make_random_network.py -n 2 --widths 4,3 --seed 1 -o examples/rand.json

Entries are uniform random fractions k/denominator with |k| <= denominator
(written as decimal strings when denominator is a power of ten).
"""
import argparse
import json
import random
from fractions import Fraction
from exact_regions.numbers import fmt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, required=True, help="input dimension")
    p.add_argument("-l", type=int, default=None, help="number of hidden layers (each of width n) -- or use --widths")
    p.add_argument("--widths", type=str, default=None,
                   help="comma separated hidden widths, e.g. 4,3,5 (overrides -l)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--denominator", type=int, default=100)
    p.add_argument("-o", "--output", default=None)
    a = p.parse_args()
    rng = random.Random(a.seed)
    D = a.denominator

    def r():
        return fmt(Fraction(rng.randint(-D, D), D))

    if a.widths:
        hidden = [int(w) for w in a.widths.split(",")]
    elif a.l is not None:
        hidden = [a.n] * a.l
    else:
        p.error("give -l or --widths")
    widths = hidden + [1]
    prev = a.n
    weights, biases = [], []
    for w in widths:
        weights.append([[r() for _ in range(prev)] for _ in range(w)])
        biases.append([r() for _ in range(w)])
        prev = w
    s = json.dumps({"weights": weights, "biases": biases})
    if a.output:
        with open(a.output, "w") as f:
            f.write(s + "\n")
    else:
        print(s)


if __name__ == "__main__":
    main()
