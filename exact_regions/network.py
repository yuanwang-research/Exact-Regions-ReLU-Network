"""Fully connected ReLU network with exact rational parameters."""
import json
from fractions import Fraction
from .numbers import to_fraction, fmt


class ReLUNetwork:
    """A feed-forward ReLU network  x -> W_out relu(... relu(W_1 x + b_1) ...) + b_out.

    ``weights[k]`` is the matrix of layer k (rows = output neurons), ``biases[k]``
    its bias vector.  The last layer is the (linear) output layer and must have
    a single output.  Hidden layers may have any widths, although the intended
    use is ``l`` hidden layers of width ``n`` on an ``n``-dimensional input.
    """

    def __init__(self, weights, biases):
        if len(weights) != len(biases) or len(weights) < 1:
            raise ValueError("need the same (positive) number of weight matrices and bias vectors")
        self.weights = []
        self.biases = []
        prev = None
        for k, (W, b) in enumerate(zip(weights, biases)):
            Wf = [tuple(to_fraction(v) for v in row) for row in W]
            bf = tuple(to_fraction(v) for v in b)
            if not Wf:
                raise ValueError("layer %d has no neurons" % k)
            width = len(Wf[0])
            if any(len(row) != width for row in Wf):
                raise ValueError("layer %d: ragged weight matrix" % k)
            if len(bf) != len(Wf):
                raise ValueError("layer %d: %d rows but %d biases" % (k, len(Wf), len(bf)))
            if prev is not None and width != prev:
                raise ValueError("layer %d expects %d inputs but previous layer has %d neurons"
                                 % (k, width, prev))
            prev = len(Wf)
            self.weights.append(Wf)
            self.biases.append(bf)
        if prev != 1:
            raise ValueError("output layer must have exactly one neuron, got %d" % prev)
        self.input_dim = len(self.weights[0][0])
        self.hidden_widths = [len(W) for W in self.weights[:-1]]
        self.num_hidden_layers = len(self.weights) - 1

    # -- construction helpers --------------------------------------------
    @classmethod
    def from_dict(cls, d):
        return cls(d["weights"], d["biases"])

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            # parse_float=Fraction keeps decimals such as 0.1 exact (1/10).
            d = json.load(f, parse_float=Fraction)
        return cls.from_dict(d)

    def to_dict(self):
        return {"weights": [[[fmt(v) for v in row] for row in W] for W in self.weights],
                "biases": [[fmt(v) for v in b] for b in self.biases]}

    # -- evaluation ------------------------------------------------------
    def preactivations(self, x):
        """Exact pre-activations of every hidden layer and the output, at rational x."""
        h = [to_fraction(v) for v in x]
        if len(h) != self.input_dim:
            raise ValueError("input has wrong dimension")
        pres = []
        for k, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = [sum(wij * hj for wij, hj in zip(row, h)) + bk for row, bk in zip(W, b)]
            pres.append(z)
            if k < self.num_hidden_layers:
                h = [v if v > 0 else Fraction(0) for v in z]
        return pres

    def __call__(self, x):
        return self.preactivations(x)[-1][0]

    def activation_pattern(self, x):
        """Pattern (tuple per hidden layer of 0/1) at x; a neuron is 'active' iff its pre-activation is > 0."""
        pres = self.preactivations(x)
        return tuple(tuple(1 if v > 0 else 0 for v in z) for z in pres[:-1])
