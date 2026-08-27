"""Full sign-flip enumeration vs the theoretical formulas of num_regions_theoretical.

For each small architecture: draw fixed rational magnitudes, enumerate every
sign pattern, count the activation regions of each signed network exactly, and
compare the exact average with E_act (and, where the theoretical value applies,
the linear-region average with E_lin).
"""
import os
import random
import unittest
from fractions import Fraction

import num_regions_theoretical as theory
from exact_regions import ReLUNetwork, sign_flip_averages


def magnitudes(ln, seed, denominator=50):
    """Random non-zero magnitudes for architecture ln = (n_0, ..., n_l), output 1."""
    rng = random.Random(seed)

    def r():
        return Fraction(rng.randint(1, denominator), denominator)

    dims = list(ln) + [1]
    W = [[[r() for _ in range(dims[k])] for _ in range(dims[k + 1])] for k in range(len(dims) - 1)]
    b = [[r() for _ in range(dims[k + 1])] for k in range(len(dims) - 1)]
    return ReLUNetwork(W, b)


class TestSignFlipVsTheory(unittest.TestCase):
    # (ln, fix_first_layer, seeds); every case enumerates 2^P networks, so keep P
    # small.  fix_first_layer=True is valid whenever n_0 >= n_1 (the counts do not
    # depend on a full-rank first layer) and shrinks the enumeration a lot.
    CASES = [
        ([1, 2], False, (0, 1)),      # 2^5  patterns per draw
        ([1, 3], False, (0, 1)),      # 2^8
        ([2, 2], False, (0, 1)),      # 2^6
        ([3, 3], True, (0, 1)),       # 2^3
        ([1, 1, 2], False, (0, 1)),   # 2^7
        ([1, 2, 1], False, (0, 1)),   # 2^7
        ([1, 2, 2], False, (0,)),     # 2^11
        ([2, 1, 2], False, (0, 1)),   # 2^7
        ([2, 2, 1], False, (0, 1)),   # 2^8
        ([2, 2, 2], True, (0, 1)),    # 2^8
        ([3, 2, 2], True, (0,)),      # 2^8
    ]

    def test_activation_average_matches_E_act(self):
        for ln, fix, seeds in self.CASES:
            expected = theory.E_act(list(ln))
            for seed in seeds:
                avg, _, _ = sign_flip_averages(magnitudes(ln, seed), fix_first_layer=fix)
                self.assertAlmostEqual(float(avg["activation"]), expected, places=9,
                                       msg="E_act mismatch for ln=%s seed=%d" % (ln, seed))

    def test_linear_average_matches_E_lin_one_hidden_layer(self):
        # with one hidden layer E_lin = E_act and the sign-flip average agrees
        for ln in ([1, 2], [1, 3], [2, 2]):
            expected = theory.E_lin(list(ln))
            self.assertAlmostEqual(theory.E_act(list(ln)), expected, places=12)
            avg, _, _ = sign_flip_averages(magnitudes(ln, 0))
            self.assertAlmostEqual(float(avg["linear"]), expected, places=9,
                                   msg="E_lin mismatch for ln=%s" % (ln,))

    def test_linear_average_matches_E_lin_two_hidden_layers(self):
        # includes cases with input dimension >= first hidden width, where the
        # expected number of linear regions is fixed and equal to E_lin
        for ln, fix in ([[1, 1, 2], False], [[1, 2, 1], False], [[1, 2, 2], False],
                        [[2, 1, 2], False], [[2, 2, 1], False], [[2, 2, 2], True], [[3, 2, 2], True]):
            expected = theory.E_lin(list(ln))
            avg, _, _ = sign_flip_averages(magnitudes(ln, 0), fix_first_layer=fix)
            self.assertAlmostEqual(float(avg["linear"]), expected, places=9,
                                   msg="E_lin mismatch for ln=%s" % (ln,))

    @unittest.skipUnless(os.environ.get("RUN_SLOW_TESTS"), "set RUN_SLOW_TESTS=1 to enable (~30s)")
    def test_linear_average_matches_E_lin_2_2_3(self):
        avg, _, _ = sign_flip_averages(magnitudes([2, 2, 3], 0), fix_first_layer=True)
        self.assertAlmostEqual(float(avg["activation"]), theory.E_act([2, 2, 3]), places=9)
        self.assertAlmostEqual(float(avg["linear"]), theory.E_lin([2, 2, 3]), places=9)


if __name__ == "__main__":
    unittest.main()
