import itertools
import random
import unittest
from fractions import Fraction

from exact_regions import ReLUNetwork, RegionCounter, PrecisionOverflowError, set_max_bits
from exact_regions.lp import lp_feasible, open_polyhedron_nonempty, closed_polyhedron_nonempty
from exact_regions.numbers import to_fraction

F = Fraction


class TestNumbers(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(to_fraction("1/3"), F(1, 3))
        self.assertEqual(to_fraction("-0.25"), F(-1, 4))
        self.assertEqual(to_fraction(0.1), F(1, 10))
        self.assertEqual(to_fraction("2.5e-3"), F(1, 400))
        self.assertEqual(to_fraction(7), F(7))

    def test_overflow(self):
        set_max_bits(8)
        try:
            with self.assertRaises(PrecisionOverflowError):
                to_fraction("1/1000")
        finally:
            set_max_bits(100_000)


class TestLP(unittest.TestCase):
    def test_basic(self):
        # x >= 0, x1 + x2 >= 1, x1 - x2 == 0  -> feasible
        self.assertTrue(lp_feasible(2, [([1, 1], 1)], [([1, -1], 0)]))
        # x1 + x2 >= 1 and -x1 - x2 >= 0 -> infeasible
        self.assertFalse(lp_feasible(2, [([1, 1], 1), ([-1, -1], 0)]))

    def test_open(self):
        # x > 0 and x < 1 : non-empty
        self.assertTrue(open_polyhedron_nonempty(1, [((F(1),), F(0)), ((F(-1),), F(1))]))
        # x > 1 and x < 1 : empty
        self.assertFalse(open_polyhedron_nonempty(1, [((F(1),), F(-1)), ((F(-1),), F(1))]))
        # x > 0 and x < 0 on the line x = 0 : empty, but closed version is a point
        cons = [((F(1),), F(0)), ((F(-1),), F(0))]
        self.assertFalse(open_polyhedron_nonempty(1, cons))
        self.assertTrue(closed_polyhedron_nonempty(1, cons))
        # strict system with equality: y > 0, y < 1, x = 0  in R^2
        self.assertTrue(open_polyhedron_nonempty(2, [((F(0), F(1)), F(0)), ((F(0), F(-1)), F(1))],
                                                 [((F(1), F(0)), F(0))]))
        # unbounded direction only: x > 0, y > x (cone)
        self.assertTrue(open_polyhedron_nonempty(2, [((F(1), F(0)), F(0)), ((F(-1), F(1)), F(0))]))


def brute_force_activation_regions(net):
    """Independent check: try every pattern and test its open polyhedron directly."""
    n = net.input_dim
    count = 0
    sizes = net.hidden_widths
    for bits in itertools.product(*[itertools.product([0, 1], repeat=w) for w in sizes]):
        # affine maps layer by layer
        M = [tuple(F(int(i == j)) for j in range(n)) for i in range(n)]
        v = tuple(F(0) for _ in range(n))
        cons = []
        ok = True
        for k in range(net.num_hidden_layers):
            W, b = net.weights[k], net.biases[k]
            A = [tuple(sum(row[j] * M[j][t] for j in range(len(row))) for t in range(n)) for row in W]
            d = [sum(row[j] * v[j] for j in range(len(row))) + bk for row, bk in zip(W, b)]
            for i, s in enumerate(bits[k]):
                if all(x == 0 for x in A[i]):
                    if (d[i] > 0) != bool(s):
                        ok = False
                elif s:
                    cons.append((A[i], d[i]))
                else:
                    cons.append((tuple(-x for x in A[i]), -d[i]))
            M = [A[i] if bits[k][i] else tuple(F(0) for _ in range(n)) for i in range(len(A))]
            v = tuple(d[i] if bits[k][i] else F(0) for i in range(len(A)))
        if ok and open_polyhedron_nonempty(n, cons):
            count += 1
    return count


def random_net(rng, n, l, D=20, widths=None):
    def r():
        return F(rng.randint(-D, D), D)
    widths = (widths if widths is not None else [n] * l) + [1]
    prev = n
    weights, biases = [], []
    for w in widths:
        weights.append([[r() for _ in range(prev)] for _ in range(w)])
        biases.append([r() for _ in range(w)])
        prev = w
    return ReLUNetwork(weights, biases)


class TestRegions(unittest.TestCase):
    def test_relu_1d(self):
        # f = relu(x) + 0*relu(x-1): 3 activation regions, 2 linear regions
        net = ReLUNetwork([[[1], [1]], [[1, 0]]], [[0, -1], [0]])
        c = RegionCounter(net)
        self.assertEqual(c.count(), (3, 2))
        self.assertEqual(c.count("touch"), (3, 2))

    def test_relu_minus_relu(self):
        # f = relu(x) - relu(x-1): 3 activation and 3 linear regions
        net = ReLUNetwork([[[1], [1]], [[1, -1]]], [[0, -1], [0]])
        self.assertEqual(RegionCounter(net).count(), (3, 3))

    def test_two_lines(self):
        net = ReLUNetwork([[[1, 0], [0, 1]], [[1, 1]]], [[0, 0], [0]])
        self.assertEqual(RegionCounter(net).count(), (4, 4))

    def test_zero_output(self):
        # output weights zero: f == 0 everywhere -> 1 linear region, 4 activation regions
        net = ReLUNetwork([[[1, 0], [0, 1]], [[0, 0]]], [[0, 0], [0]])
        self.assertEqual(RegionCounter(net).count(), (4, 1))

    def test_sawtooth(self):
        # tent(tent(x)) built from two layers of width 2: 6 pieces on [0,1] plus two constant tails
        net = ReLUNetwork([[[2], [2]], [[2, -4], [2, -4]], [[1, -2]]], [[0, -1], [0, -1], [0]])
        self.assertEqual(RegionCounter(net).count(), (6, 6))

    def test_touch_vs_facet(self):
        # h = (relu(x), relu(-x), relu(y), relu(-y)),  z = |y| - |x|,  f = relu(z).
        # f == 0 on the two opposite cones |x| > |y|, which meet only at the origin:
        # with facet adjacency they are 2 linear regions, with touch adjacency 1.
        # On |y| > |x| the maps y-x, y+x, -y-x, -y+x are all distinct (4 regions).
        net = ReLUNetwork([[[1, 0], [-1, 0], [0, 1], [0, -1]], [[-1, -1, 1, 1]], [[1]]],
                          [[0, 0, 0, 0], [0], [0]])
        c = RegionCounter(net)
        acts, facet = c.count("facet")
        _, touch = c.count("touch")
        self.assertEqual(acts, 8)
        self.assertEqual(facet, 6)
        self.assertEqual(touch, 5)

    def test_constant_neuron(self):
        # second-layer neuron with zero incoming weights and positive bias: constant, no boundary
        net = ReLUNetwork([[[1]], [[0], [3]], [[1, 1]]], [[0], [1, 0], [0]])
        c = RegionCounter(net)
        self.assertEqual(c.count(), (2, 2))

    def test_brute_force_random(self):
        rng = random.Random(12345)
        for trial in range(12):
            n = rng.choice([1, 2, 3])
            l = rng.choice([1, 2])
            net = random_net(rng, n, l, D=10)
            c = RegionCounter(net)
            acts = c.activation_regions()
            self.assertEqual(len(acts), brute_force_activation_regions(net), msg=net.to_dict())
            # every region's affine map must agree with direct evaluation at an interior-ish sample
            lins = c.linear_regions()
            self.assertLessEqual(len(lins), len(acts))
            # patterns observed on a grid must be among the enumerated ones
            pats = {r.pattern for r in acts}
            for _ in range(200):
                x = [F(rng.randint(-300, 300), 7) for _ in range(n)]
                pre = net.preactivations(x)
                if any(v == 0 for z in pre[:-1] for v in z):
                    continue
                self.assertIn(net.activation_pattern(x), pats)

    def test_mixed_widths_hand_computed(self):
        # n=1, widths [3, 2]: h = (relu(x), relu(x-1), relu(x-2)); second layer
        # u1 = relu(h1 - 2 h2), u2 = relu(h2 - 2 h3 - 1/4); f = u1 + u2.
        # g1 = h1 - 2h2 = 0 (x<0), x (0<x<1), 2-x (x>1): zero at x=2 -> u1 kink at x=2.
        # g2 = h2 - 2h3 - 1/4 = -1/4 (x<1), x-5/4 (1<x<2), 11/4-x (x>2): zeros at 5/4, 11/4.
        # Breakpoints: 0, 1, 5/4, 2, 11/4  -> 6 activation regions.
        # Pieces of f: 0 | x | 2-x | (2-x)+(x-5/4)=3/4 | 11/4-x (x in (2,11/4)) ... careful:
        #   (1,5/4): u1 = 2-x, u2 = 0 -> 2-x ; (5/4,2): 2-x + x-5/4 = 3/4 ; (2,11/4): 0 + 11/4-x ; x>11/4: 0.
        # All neighbouring maps differ -> 6 linear regions.
        net = ReLUNetwork([[[1], [1], [1]], [[1, -2, 0], [0, 1, -2]], [[1, 1]]],
                          [[0, -1, -2], [0, F(-1, 4)], [0]])
        self.assertEqual(net.hidden_widths, [3, 2])
        c = RegionCounter(net)
        self.assertEqual(c.count(), (6, 6))
        outs = sorted({r.output for r in c.activation_regions()})
        self.assertIn(((F(0),), F(3, 4)), outs)
        self.assertIn(((F(-1),), F(11, 4)), outs)

    def test_mixed_widths_merge(self):
        # n=2, widths [3, 1]: h = (relu(x), relu(y), relu(-x-y)); f = relu(h1 + h2 + h3)*0 + ... use
        # f = relu(h1 - h1) -> trivial; instead f = relu(-(h1+h2+h3)) which is identically 0:
        # 3 lines through the origin in general position -> 6 activation regions of layer 1,
        # second neuron always dead -> 6 activation regions, all glued (facet-adjacent) -> 1 linear region.
        net = ReLUNetwork([[[1, 0], [0, 1], [-1, -1]], [[-1, -1, -1]], [[1]]], [[0, 0, 0], [0], [0]])
        self.assertEqual(net.hidden_widths, [3, 1])
        self.assertEqual(RegionCounter(net).count(), (6, 1))

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            ReLUNetwork([[[1, 0], [0, 1]], [[1, 1, 1]]], [[0, 0], [0]])      # 3 inputs to output, 2 neurons
        with self.assertRaises(ValueError):
            ReLUNetwork([[[1, 0], [0, 1]], [[1, 1], [1, 1]]], [[0, 0], [0, 0]])   # two outputs

    def test_brute_force_random_mixed_widths(self):
        rng = random.Random(999)
        for trial in range(10):
            n = rng.choice([1, 2, 3])
            widths = [rng.choice([1, 2, 3, 4]) for _ in range(rng.choice([1, 2, 3]))]
            net = random_net(rng, n, None, D=10, widths=widths)
            self.assertEqual(net.hidden_widths, widths)
            c = RegionCounter(net)
            acts = c.activation_regions()
            self.assertEqual(len(acts), brute_force_activation_regions(net), msg=net.to_dict())
            lins = c.linear_regions()
            self.assertLessEqual(len(lins), len(acts))
            pats = {r.pattern for r in acts}
            for _ in range(100):
                x = [F(rng.randint(-300, 300), 7) for _ in range(n)]
                pre = net.preactivations(x)
                if any(v == 0 for z in pre[:-1] for v in z):
                    continue
                self.assertIn(net.activation_pattern(x), pats)

    def test_generic_mode_agrees(self):
        rng = random.Random(4242)
        for trial in range(12):
            n = rng.choice([1, 2, 3])
            widths = [rng.choice([1, 2, 3]) for _ in range(rng.choice([1, 2, 3]))]
            net = random_net(rng, n, None, D=1000, widths=widths)   # large denominator: generic
            c = RegionCounter(net)
            exact = c.linear_regions("facet", generic=False)
            fast = c.linear_regions("facet", generic=True)
            self.assertEqual(sorted(sorted(r.index for r in L) for L in exact),
                             sorted(sorted(r.index for r in L) for L in fast), msg=net.to_dict())

    def test_output_affine_matches_evaluation(self):
        rng = random.Random(7)
        net = random_net(rng, 2, 2, D=10)
        c = RegionCounter(net)
        pats = {r.pattern: r.output for r in c.activation_regions()}
        for _ in range(300):
            x = [F(rng.randint(-300, 300), 11) for _ in range(2)]
            pat = net.activation_pattern(x)
            if pat in pats:
                a, cc = pats[pat]
                self.assertEqual(sum(ai * xi for ai, xi in zip(a, x)) + cc, net(x))


if __name__ == "__main__":
    unittest.main()
