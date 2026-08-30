# Exact local-stability certificate

Command:

```bash
python verify_stability.py \
  examples/quad_ring.json examples/quad_axis.json \
  examples/quad_random0.json examples/quad_random1.json examples/quad_random2.json \
  --quiet
```

All five magnitude vectors passed (about 15 s each).

For each vector the verifier (`verify_stability.py`, general input dimension and
widths) checked:

* 14 sign-flipped parameters (the two later hidden layers and the output weight;
  the output bias is fixed by the output-negation symmetry, and the first layer
  is normalised to the identity, valid because input dim >= first width);
* all 16,384 sign patterns of those parameters -- exactly the patterns of the
  sign-flip experiment;
* 46 affine preactivation forms arising from every prefix activation pattern,
  plus the line at infinity;
* 16,215 three-by-three determinants, of which 5,869 vanish symbolically for
  structural reasons and 10,346 must remain nonzero;
* 30 symbolic affine output-map classes.

Thus, per magnitude vector, 10,346 x 16,384 = 169,508,864 non-structural
determinant evaluations were certified nonzero (modular screen at the prime
1,000,003 with exact-rational recheck of every modular zero; zero rechecks were
needed).  No accidental equality between symbolically distinct output maps
occurred.  The symbolic-family SHA-256 digest begins `a7485be5f895e3c6`.

The determinant signs determine the oriented matroid of all activation-boundary
hyperplanes (including the one at infinity), which in turn determines every
strict-feasibility and relative-facet predicate used in the activation- and
linear-region counts.  Structural zero determinants vanish identically in the
parameters and persist; all other determinants and all output-map inequalities
remain nonzero in some neighborhood by continuity.  Finiteness gives one common
open neighborhood of each magnitude vector on which every count, and therefore
its sign-flip average, is constant.

In particular, `quad_random0` and `quad_random1` have different exact sign-flip
averages on two open magnitude neighborhoods.  This closes the measure-zero gap
when constructing independent, parameter-specific continuous even densities
supported in those neighborhoods.  It does not claim the same result when all
parameters are required to share one common i.i.d. density.
