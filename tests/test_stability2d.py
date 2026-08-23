import unittest
import importlib.util
from fractions import Fraction as F

from exact_regions import ReLUNetwork

HAS_OPTIONAL_DEPS = (importlib.util.find_spec("numpy") is not None and
                     importlib.util.find_spec("sympy") is not None)
if HAS_OPTIONAL_DEPS:
    import verify_stability2d as stability


@unittest.skipUnless(HAS_OPTIONAL_DEPS, "stability certificate requires sympy and numpy")
class TestStabilityCertificate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = ReLUNetwork.from_json("examples/quad_random0.json")
        cls.family = stability._symbolic_family(cls.net)

    def test_symbolic_family(self):
        self.assertEqual(len(self.family["positions"]), 13)
        self.assertEqual(len(self.family["vectors"]) - 1, 46)
        self.assertEqual(len(self.family["outputs"]), 64)
        self.assertEqual(len(self.family["triples"]), 10346)
        self.assertEqual(len(self.family["output_representatives"]), 30)

    def test_nonstructural_degeneracy_is_detectable(self):
        # Making all later hidden parameters equal creates an accidental
        # coincidence.  The associated determinant is symbolically nonzero
        # but evaluates to zero, exactly the failure checked by certify().
        values = [F(1)] * len(self.family["positions"])
        W, b = stability._normalised_hidden_layers(self.net, values)
        records, _ = stability._all_affine_forms(W, b)
        lines = [record[3] for record in records] + [stability.INF]
        triple = (0, 5, 6)
        self.assertIn(triple, self.family["triples"])
        self.assertEqual(stability._det3(*(lines[i] for i in triple)), 0)

    def test_first_layer_rank_is_validated(self):
        W = [[list(row) for row in layer] for layer in self.net.weights]
        b = [list(row) for row in self.net.biases]
        W[0] = [[1, 1], [1, 1]]
        with self.assertRaises(ValueError):
            stability._validate_architecture(ReLUNetwork(W, b))


if __name__ == "__main__":
    unittest.main()
