# Exact local-stability certificate

Command:

```bash
python verify_stability2d.py \
  examples/quad_ring.json examples/quad_axis.json \
  examples/quad_random0.json examples/quad_random1.json examples/quad_random2.json \
  --quiet
```

All five magnitude vectors passed.

For each vector the verifier checked:

* 13 hidden-layer parameters that affect activation geometry;
* all 8,192 signs of those parameters (the output-weight sign gives the full
  16,384 sign patterns used by the experiment without changing the geometry or
  equality of output maps);
* 46 affine preactivation forms arising from every prefix activation pattern;
* 16,215 three-by-three line determinants, of which 5,869 vanish symbolically
  for structural reasons and 10,346 must remain nonzero;
* 30 symbolic affine-map classes of the final hidden activation.

Thus, per magnitude vector, 10,346 × 8,192 = 84,754,432 non-structural
determinant evaluations were certified nonzero.  No accidental equality between
symbolically distinct output maps occurred.  The symbolic-family SHA-256 digest
reported by the verifier begins `587aeafbc75d9676`.

The determinant signs determine the oriented matroid of all activation-boundary
lines (including the line at infinity).  Consequently they determine every
strict-half-plane feasibility and relative-facet feasibility predicate used in
the activation- and linear-region counts.  Structural zero determinants remain
zero identically; all other determinants and all output-map inequalities remain
nonzero in some neighborhood by continuity.  Finiteness of the determinant and
sign-pattern families gives one common open neighborhood of each magnitude
vector on which every count, and therefore its sign-flip average, is constant.

In particular, `quad_random0` and `quad_random1` have different exact sign-flip
averages on two open magnitude neighborhoods.  This closes the measure-zero gap
when constructing independent, parameter-specific continuous even densities
supported in those neighborhoods.  It does not claim the same result when all
parameters are required to share one common i.i.d. density.
