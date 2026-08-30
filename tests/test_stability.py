"""Tests for verify_stability on synthetic networks (independent of examples/)."""
import importlib.util
import random
import unittest
from fractions import Fraction as F

from exact_regions import ReLUNetwork

HAS_OPTIONAL_DEPS = (importlib.util.find_spec("numpy") is not None and
                     importlib.util.find_spec("sympy") is not None)
if HAS_OPTIONAL_DEPS:
    import verify_stability as stability


def make_net(d, widths, W_hidden, b_hidden, W_out, b_out=F(1, 3), W1=None):
    if W1 is None:
        W1 = [[1 if i == j else 0 for j in range(d)] for i in range(widths[0])]
        b1 = [0] * widths[0]
        return ReLUNetwork([W1] + W_hidden + [W_out], [b1] + b_hidden + [[b_out]])
    return ReLUNetwork([W1] + W_hidden + [W_out], b_hidden + [[b_out]])


def random_net(d, widths, seed, denominator=50):
    rng = random.Random(seed)

    def r():
        return F(rng.randint(1, denominator), denominator)

    dims = [d] + list(widths) + [1]
    return ReLUNetwork([[[r() for _ in range(dims[k])] for _ in range(dims[k + 1])]
                        for k in range(len(dims) - 1)],
                       [[r() for _ in range(dims[k + 1])] for k in range(len(dims) - 1)])


@unittest.skipUnless(HAS_OPTIONAL_DEPS, "stability certificate requires sympy and numpy")
class TestFlagshipArchitecture(unittest.TestCase):
    """Input 2, hidden (2, 3, 1): the architecture of the density-dependence example."""

    @classmethod
    def setUpClass(cls):
        cls.net = random_net(2, (2, 3, 1), 0)
        cls.family = stability.symbolic_family(cls.net)

    def test_family_invariants(self):
        self.assertTrue(self.family["fix_first_layer"])
        self.assertEqual(len(self.family["positions"]), 14)
        self.assertEqual(len(self.family["records"]), 46)
        self.assertEqual(self.family["num_probes"], 1)          # only infinity
        self.assertEqual(len(self.family["subsets"]), 10346)
        self.assertEqual(len(self.family["output_representatives"]), 30)

    def test_random_magnitudes_certify_all_flips(self):
        result = stability.certify(self.net, family=self.family)
        self.assertEqual(result["sign_patterns"], 16384)

    def test_certify_single_uses_the_given_signs(self):
        W = [[list(r) for r in Wk] for Wk in self.net.weights]
        b = [list(bk) for bk in self.net.biases]
        W[1][0][0] = -W[1][0][0]
        b[2][0] = -b[2][0]
        result = stability.certify_single(ReLUNetwork(W, b), family=self.family)
        self.assertEqual(result["sign_patterns"], 1)

    def test_concurrent_lines_fail(self):
        # Second-layer lines x+y-2, x+2y-3, 2x+y-3 pass through (1, 1) at the
        # sign pattern with all three biases negative.
        net = make_net(2, (2, 3, 1), [[[1, 1], [1, 2], [2, 1]], [[1, 1, 1]]],
                       [[2, 3, 3], [1]], [[1]])
        with self.assertRaises(stability.StabilityFailure):
            stability.certify(net, family=self.family)

    def test_parallel_lines_fail(self):
        # Two second-layer lines with equal normals; the vanishing determinant
        # involves the hyperplane at infinity.
        net = make_net(2, (2, 3, 1), [[[1, 1], [1, 1], [1, 2]], [[1, 1, 1]]],
                       [[1, 2, 1], [1]], [[1]])
        with self.assertRaises(stability.StabilityFailure):
            stability.certify(net, family=self.family)

    def test_engineered_coincidence_is_nonstructural(self):
        # With every flipped parameter equal to 1 some symbolically nonzero
        # determinant evaluates to zero: what certify() must detect.
        values = [F(1)] * len(self.family["positions"])
        W, b = stability._layers_from_values(self.net, self.family["positions"], values,
                                             True, fixed=lambda v: F(v))
        records, _ = stability._forms_and_outputs(2, W, b)
        lines = [form for (_, _, _, form) in records] + stability._probes(2, records)
        vanished = [t for t in self.family["subsets"]
                    if stability._det_exact([lines[q] for q in t]) == 0]
        self.assertTrue(vanished)

    def test_budget_guard(self):
        with self.assertRaises(stability.BudgetExceeded):
            stability.certify(self.net, family=self.family, budget=10)

    def test_first_layer_rank_is_validated_per_mode(self):
        # |W1| singular, signed W1 full rank: certify (which uses magnitudes)
        # must reject it; certify_single (which uses the signs) must accept it.
        def with_W1(W1):
            return make_net(2, (2, 3, 1), [[[1, 1], [1, 2], [2, 1]], [[1, 1, 1]]],
                            [[0, 0], [1, 1, 1], [1]], [[1]], W1=W1)
        mixed = with_W1([[1, 1], [1, -1]])
        with self.assertRaises(ValueError):
            stability.certify(mixed, family=self.family)
        with self.assertRaises(ValueError):        # singular either way
            stability.certify_single(with_W1([[1, 1], [1, 1]]), family=self.family)

    def test_supplied_family_cannot_bypass_validation(self):
        bad = make_net(2, (2, 3, 1), [[[1, 1], [1, 2], [2, 1]], [[1, 1, 1]]],
                       [[0, 0], [1, 1, 1], [1]], [[1]], W1=[[1, 1], [1, 1]])
        with self.assertRaises(ValueError):
            stability.certify(bad, family=self.family)
        other_arch = random_net(2, (2, 2), 0)
        with self.assertRaises(ValueError):
            stability.certify(other_arch, family=self.family)

    def test_zero_flipped_parameter_is_rejected(self):
        net = make_net(2, (2, 3, 1), [[[1, 0], [1, 2], [2, 1]], [[1, 1, 1]]],
                       [[1, 1, 1], [1]], [[1]])
        with self.assertRaises(ValueError):
            stability.certify(net, family=self.family)


@unittest.skipUnless(HAS_OPTIONAL_DEPS, "stability certificate requires sympy and numpy")
class TestOtherArchitectures(unittest.TestCase):
    def test_3d_random_certifies(self):
        net = random_net(3, (2, 2), 7)
        result = stability.certify(net)
        self.assertEqual(result["sign_patterns"], 256)
        self.assertEqual(result["probes"], 2)      # e3 and infinity

    def test_3d_concurrent_planes_fail(self):
        # Planes x=0, y=0 and the two second-layer planes x+y+z±1, x+2y+z±1
        # meet at (0, 0, -+1): degenerate at every bias sign.
        net = ReLUNetwork([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 1, 1], [1, 2, 1]], [[1, 1]]],
                          [[0, 0, 0], [1, 1], [1]])
        with self.assertRaises(stability.StabilityFailure):
            stability.certify(net)

    def test_1d_reflection_normalisation_in_certify_single(self):
        # W1 = (-1, 2), b1 = (1, -2): both breakpoints at x = 1 -- degenerate.
        # Normalising the anchor row must reflect whole columns, otherwise this
        # coincidence is invisible.
        from fractions import Fraction as FF
        W2, b2, Wo, bo = [[FF(3, 7), FF(2, 5)]], [FF(1, 3)], [[FF(4, 9)]], FF(1, 3)
        net = ReLUNetwork([[[-1], [2]], W2, Wo], [[1, -2], b2, [bo]])
        with self.assertRaises(stability.StabilityFailure):
            stability.certify_single(net)
        # a reflection-equivalent pair must certify identically
        netA = ReLUNetwork([[[-1], [2]], W2, Wo], [[1, FF(3, 7)], b2, [bo]])
        netB = ReLUNetwork([[[1], [-2]], W2, Wo], [[1, FF(3, 7)], b2, [bo]])
        family = stability.symbolic_family(netA)
        self.assertEqual(stability.certify_single(netA, family=family)["sign_patterns"], 1)
        self.assertEqual(stability.certify_single(netB, family=family)["sign_patterns"], 1)

    def test_zero_reflection_anchor_rejected(self):
        from fractions import Fraction as FF
        net = ReLUNetwork([[[0], [1]], [[FF(3, 7), FF(2, 5)]], [[FF(4, 9)]]],
                          [[FF(1, 3), FF(2, 7)], [FF(1, 3)], [FF(1, 3)]])
        with self.assertRaises(ValueError):
            stability.certify(net)
        with self.assertRaises(ValueError):
            stability.symbolic_family(net)

    def test_1d_first_layer_flipped(self):
        net = random_net(1, (2, 2), 3)
        family = stability.symbolic_family(net)
        self.assertFalse(family["fix_first_layer"])
        result = stability.certify(net, family=family)
        self.assertEqual(result["sign_patterns"], 2048)


if __name__ == "__main__":
    unittest.main()
