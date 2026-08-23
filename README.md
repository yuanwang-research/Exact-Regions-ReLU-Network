# Exact counting of activation regions and linear regions of ReLU networks

Pure-Python (3.8+, standard library only) code that counts, **exactly**, the
activation regions and linear regions of a fully connected ReLU network
`R^n -> R` with `l` hidden layers (any widths; intended use is width `n`).
All arithmetic is done with `fractions.Fraction`; nothing is ever rounded.

## Usage

```bash
python count_regions.py examples/quad_ring.json --list
```

Options: `--adjacency facet|touch` (see definitions), `--list` (print every
region), `--max-regions N`, `--max-bits B`.

Network file format (`W_k` = list of rows, last layer is the 1-output linear layer):

```json
{"weights": [W_1, ..., W_l, W_out], "biases": [b_1, ..., b_l, b_out]}
```

Entries may be integers, decimals (`0.25` – kept exact, `0.1` means 1/10) or
fraction strings (`"-1/3"`).  Hidden layers may have arbitrary (different) widths.
Random rational networks: `python make_random_network.py -n 3 -l 3 --seed 1 -o net.json`
or with explicit widths `python make_random_network.py -n 2 --widths 4,3,2 --seed 1 -o net.json`.

From Python:

```python
from exact_regions import ReLUNetwork, RegionCounter
net = ReLUNetwork([[[1], [1]], [[1, 0]]], [[0, -1], [0]])   # f(x) = relu(x) + 0*relu(x-1)
c = RegionCounter(net)
n_act, n_lin = c.count()            # (3, 2)
for r in c.activation_regions():   # r.pattern, r.constraints, r.output = (a, c): f(x) = a.x + c
    print(r)
```

Run the tests with `python -m unittest discover -s tests`.

## Definitions

* **Activation region** – the set of inputs with a given activation pattern
  (for every hidden neuron: pre-activation `> 0` or `<= 0`).  It is an open
  convex polyhedron; we count the patterns whose region is non-empty, i.e. has
  non-empty interior.  Patterns realised only on measure-zero sets (exactly on
  a boundary hyperplane) are not counted.  A neuron whose pre-activation is
  constant on a region (e.g. all of its inputs are dead there) creates no
  boundary – its state is simply the sign of that constant.
* **Linear region** – a maximal connected set on which the network is one
  affine function; i.e. a connected component of the set of points at which
  the network is locally affine (Hanin & Rolnick's definition).  Two activation
  regions lie in the same linear region iff they are joined by a chain of
  activation regions, all with the *same* output affine map, in which
  consecutive regions share an `(n-1)`-dimensional face (`--adjacency facet`,
  default).  `--adjacency touch` instead only requires the closures of
  consecutive regions to intersect (so two regions with the same affine map
  that meet at a single vertex are glued as well).

## How it works

1. **Enumeration** (`exact_regions/regions.py`).  Depth-first search layer by
   layer.  Given the pattern of the previous layers, every pre-activation of
   the current layer is an explicit affine function `w.x + c` of the input.
   For each neuron we branch on its sign and keep a branch only if the open
   polyhedron `{x : all chosen signed pre-activations > 0}` is non-empty.
   Each surviving leaf is an activation region with its affine output map.
2. **Exact LP** (`exact_regions/lp.py`).  Non-emptiness of an open polyhedron
   `{w_j.x + c_j > 0}` is equivalent to feasibility of the closed homogenised
   system `w_j.y + c_j s >= 1, s >= 0` (proof in the module docstring), which
   is decided by a Phase-I simplex method with Bland's rule over the rationals
   – exact and guaranteed to terminate.
3. **Linear regions.**  Activation regions are grouped by their (exact) output
   affine map.  Within a group, two regions can share a facet only if that
   facet's hyperplane is a defining hyperplane of both, so candidate pairs are
   found through a hyperplane index; each candidate is checked with one exact
   LP (relative interior of the common face on that hyperplane non-empty).
   Union–find then yields the connected components.

## Precision guard

Numbers never lose precision, but to make "too large" fail loudly rather than
silently grind, every number produced by a pivot or parsed from input is
checked against a bit-length limit (default 100 000 bits; change with
`--max-bits` or `exact_regions.set_max_bits`).  Exceeding it raises
`PrecisionOverflowError` (CLI exit code 2).  `--max-regions` raises
`TooManyRegionsError` (exit code 3) when the enumeration grows beyond a bound.

## Performance

The work is roughly 2 exact LPs per node of the search tree.  Random
`n = 3, l = 3` takes about a second; `n = 5, l = 3` (≈ 900 regions) about
25 s.  The method is exponential in the number of regions by nature.

## Generic mode for linear regions

`--generic` (API: `linear_regions(generic=True)` / `count(generic=True)`) assumes
the parameters are generic: at a generic point of a facet shared by two
activation regions exactly one neuron has pre-activation 0, so facet-adjacent
activation regions differ in exactly one pattern bit.  Only such pairs are LP-tested,
which is typically 20–50× faster for deep networks.  The default tests all pairs that
share a defining hyperplane and is exact for arbitrary parameters.

## Sign-flip experiments (expected number of regions)

Research question: for a ReLU network whose weights and biases are independent
continuous random variables with *even* densities, is the expected number of
linear regions independent of the densities?

**Test.** Fix a vector of magnitudes |θ| (all non-zero) and average the exact
region count over *all* 2^P sign patterns ε·|θ|.  If the expectation were
density-independent, this average would be the same for almost every |θ|
(a density concentrated near |θ| has expectation equal to this average).
Conversely, two magnitude vectors with different sign-flip averages prove
density dependence.  Everything is exact: each count is an exact LP-based
enumeration and the average is an exact rational number.

Library: `exact_regions.sign_flip_averages(net, fix_first_layer, ...)` returns the
exact averages of the statistics of `exact_regions.region_summary` (activation regions,
linear regions, constant / non-constant linear regions, …).  Two symmetries are
quotiented out (negating the output layer; reflecting an input coordinate), and with
`fix_first_layer=True` the first hidden layer is not flipped at all — valid whenever
input dim ≥ first width, because an affine change of input coordinates turns any
full-rank first layer into `[I | 0], 0`, so no region count depends on the first layer.

Scripts:

* `sign_flip_experiment.py` – exact averages for random magnitudes or a magnitude file.
* `density_dependence.py` – the replication script for the example below.
* `verify2d.py` – an independent, LP-free counter for 2-D inputs (exact polygon clipping,
  areas for full dimension, shared edges of positive length for adjacency), used to
  cross-check the LP-based counter network by network.
* `quadrangle_example.py` – construction of the example.

### Density-independent cases

For all small architectures we tried with input dimension 1 (up to four hidden layers of
width ≤ 3) and for input 2 with hidden widths (2,2) and (2,2,2), the sign-flip averages
were identical for every magnitude draw.  The following example shows that this is not
true in general.

### Result: a density-DEPENDENT example (input 2, hidden (2, 3, 1))

**Construction** (`quadrangle_example.py`).  Start from the network on R³ with hidden
widths (3, 1): `s = relu(u)`, `c = relu(-s₁ - s₂ + s₃ - 1)`, output `c + 1/3`.  Its
non-constant locus is the cone `C = {w ≥ 1 + max(0, u, v, u+v)}` and it is constant
(zero) elsewhere.  Restrict to a plane via an affine map `u = M t + m`, `t ∈ R²` (this
*is* a network on R² whose first layer is `(M, m)`): the non-constant locus becomes the
quadrangle `Q = {t : M t + m ∈ C}`, bounded exactly when the plane's slope lies in the
open unit square, and the constant locus `R² \ Q` is a ring.  Finally add a hidden layer
of width 2 in front, fixed to `relu(t₁), relu(t₂)`; outside the positive quadrant the
picture is projected onto the axes.  Two configurations are provided:

* `examples/quad_ring.json` – `Q` inside the open first quadrant (constant locus is a ring);
* `examples/quad_axis.json` – the `t₁`-axis crosses `Q` (the constant locus is cut open).

All parameters are generic (non-zero except the fixed first layer; counts stable under
random relative perturbations of 1e‑4 … 1e‑7).  For comparison,
`examples/quad_random{0,1,2}.json` hold random magnitudes in (0, 1] for the same
architecture (first layer fixed to the identity).

**Exact sign-flip averages** over all 2^14 sign patterns of layers 2–4
(`results/density_dependence.md`, produced by `density_dependence.py`):

| magnitudes | activation | non-constant linear | constant linear | **linear** |
|---|---|---|---|---|
| quad_ring    | 387/32 | 319/64 | 3105/2048 | **13/2 + 1/2048** |
| quad_axis    | 387/32 | 319/64 | 3105/2048 | **13/2 + 1/2048** |
| quad_random0 | 387/32 | 319/64 | 3107/2048 | **13/2 + 3/2048** |
| quad_random1 | 387/32 | 319/64 | 97/64 = 3104/2048 | **13/2** |
| quad_random2 | 387/32 | 319/64 | 97/64 = 3104/2048 | **13/2** |

So the expected number of **activation regions** (387/32) and of **non-constant linear
regions** (319/64) coincide for all magnitudes, but the expected number of **constant
linear regions** – hence of linear regions – differs between magnitude vectors.  The
expected number of linear regions therefore depends on the parameter densities for this
architecture.  The mechanism is the gluing of constant activation regions: their number
(455/64 on average) is the same for all magnitudes, but how many connected components
they form is not.  Recall that two activation regions with the same affine map that are
not linked by a chain of (n−1)-dimensional contacts count as distinct linear regions.

**Verification.**  `python density_dependence.py --verify` recomputes every one of the
5 × 16 384 networks with `verify2d.py` (exact polygon clipping, no LP) and reports the
number of disagreements – it was 0 for all of them.  The same 2^14 enumeration was also
run with the default (non-generic) facet test, so no genericity assumption enters the
counts.

**To replicate** (about 5 min per magnitude file with 6 processes, double with `--verify`):

```bash
python density_dependence.py --jobs 6 --verify
```

or a single file:

```bash
python sign_flip_experiment.py --from-file examples/quad_ring.json --fix-first-layer --jobs 6
```

To rebuild the examples from scratch (`quadrangle_example.py` searches rational `M, m`
with exact LP checks of the stated conditions and writes the two JSON files):

```bash
python quadrangle_example.py
```
