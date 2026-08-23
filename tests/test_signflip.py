import unittest
from fractions import Fraction as F

from exact_regions import ReLUNetwork, region_summary, sign_flip_averages
from exact_regions.signflip import flip_positions


class TestStats(unittest.TestCase):
    def test_summary_fields(self):
        # f = relu(x) + 0*relu(x-1): 3 activation regions, 2 linear (one constant)
        net = ReLUNetwork([[[1], [1]], [[1, 0]]], [[0, -1], [0]])
        s = region_summary(net)
        self.assertEqual(s["activation"], 3)
        self.assertEqual(s["linear"], 2)
        self.assertEqual(s["const_linear"], 1)
        self.assertEqual(s["nonconst_linear"], 1)
        self.assertEqual(s["const_activation"], 1)
        self.assertEqual(s["distinct_maps"], 2)
        self.assertEqual(s["distinct_const_maps"], 1)


class TestSignFlip(unittest.TestCase):
    def test_positions(self):
        net = ReLUNetwork([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10]]], [[1, 1], [1, 1], [1]])
        # total 4+2 + 4+2 + 2+1 = 15 ; minus 2 (input reflections) minus 1 (output bias) = 12
        self.assertEqual(len(flip_positions(net, False)), 12)
        # first layer fixed: 4+2 + 2 = 8
        self.assertEqual(len(flip_positions(net, True)), 8)

    def test_known_1d(self):
        # 1 hidden layer of width 2 on R: every sign pattern gives 3 activation and 3 linear regions
        net = ReLUNetwork([[[F(1, 3)], [F(2, 5)]], [[F(1, 7), F(3, 4)]]], [[F(1, 2), F(2, 3)], [F(1, 9)]])
        avg, res, P = sign_flip_averages(net)
        self.assertEqual(P, 5)
        self.assertEqual(len(res), 32)
        self.assertEqual(avg["activation"], 3)
        self.assertEqual(avg["linear"], 3)

    def test_fixed_first_layer_invariance(self):
        # with d >= n1 the averages must not depend on the first layer at all
        import random
        rng = random.Random(3)
        def r():
            return F(rng.randint(1, 50), 50)
        W2 = [[r(), r()], [r(), r()]]; b2 = [r(), r()]; W3 = [[r(), r()]]; b3 = [r()]
        nets = [ReLUNetwork([[[r(), r()], [r(), r()]], W2, W3], [[r(), r()], b2, b3]) for _ in range(2)]
        avgs = [sign_flip_averages(n, fix_first_layer=True)[0] for n in nets]
        self.assertEqual(avgs[0], avgs[1])

    def test_zero_parameter_rejected(self):
        net = ReLUNetwork([[[1], [1]], [[1, 0]]], [[0, -1], [0]])
        with self.assertRaises(ValueError):
            sign_flip_averages(net)


if __name__ == "__main__":
    unittest.main()
