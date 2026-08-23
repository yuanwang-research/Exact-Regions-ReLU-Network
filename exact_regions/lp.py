"""Exact linear-programming feasibility via the simplex method over the rationals.

Only *feasibility* questions are needed for region counting, so we implement a
Phase-I simplex (minimise the sum of artificial variables) with Bland's
anti-cycling rule.  Every tableau entry is a ``Fraction``; there is no
floating point anywhere, so the answers are exact.

Conventions
-----------
``lp_feasible(num_vars, ge_rows, eq_rows)`` decides whether there is ``x >= 0``
(component-wise) with ``a.x >= b`` for all ``(a, b)`` in ``ge_rows`` and
``a.x == b`` for all ``(a, b)`` in ``eq_rows``.

The helpers below translate the geometric questions that arise for ReLU
networks (open polyhedra, facets, closures) into that form.  Free variables are
split as ``x = y_plus - y_minus``.

Homogenisation trick for *strict* inequalities
----------------------------------------------
The open polyhedron ``{x : w_j.x + c_j > 0 for all j}`` is non-empty iff the
closed system ``w_j.y + c_j s >= 1 for all j,  s >= 0`` is feasible:

* if ``x`` is in the open polyhedron let ``t = min_j (w_j.x + c_j) > 0`` and
  take ``y = x / t``, ``s = 1 / t``;
* conversely, if ``(y, s)`` is feasible with ``s > 0`` take ``x = y / s``; and if
  ``s = 0`` then ``w_j.y >= 1`` for all ``j``, so ``x0 + lam*y`` lies in the open
  polyhedron for any ``x0`` and large ``lam``.

The same argument works when additional homogeneous *equalities*
``w.y + c s = 0`` are imposed (they correspond to ``w.x + c = 0``).
"""
from fractions import Fraction
from .numbers import check

ZERO = Fraction(0)
ONE = Fraction(1)


class LPError(RuntimeError):
    pass


def lp_feasible(num_vars, ge_rows, eq_rows=()):
    """Exact feasibility of {x >= 0 : A_ge x >= b_ge, A_eq x = b_eq}."""
    n_s = len(ge_rows)
    ncols = num_vars + n_s            # original + surplus variables
    rows = []
    for i, (a, b) in enumerate(ge_rows):
        if len(a) != num_vars:
            raise LPError("row length mismatch")
        row = [Fraction(v) for v in a] + [ZERO] * n_s
        row[num_vars + i] = -ONE       # a.x - s = b
        rows.append((row, Fraction(b)))
    for a, b in eq_rows:
        if len(a) != num_vars:
            raise LPError("row length mismatch")
        rows.append(([Fraction(v) for v in a] + [ZERO] * n_s, Fraction(b)))
    m = len(rows)
    if m == 0:
        return True
    total = ncols + m                  # + artificial variables

    # Build tableau: each row  [coeffs..., artificials..., rhs] with rhs >= 0.
    T = []
    for i, (row, b) in enumerate(rows):
        if b < 0:
            row = [-v for v in row]
            b = -b
        full = row + [ZERO] * m
        full[ncols + i] = ONE
        full.append(b)
        T.append(full)
    basis = [ncols + i for i in range(m)]

    # Objective: minimise sum of artificials.  Reduced cost row (in terms of
    # the non-basic original columns) and negated objective value at the end.
    obj = [ZERO] * (total + 1)
    for j in range(ncols):
        obj[j] = -sum(T[i][j] for i in range(m))
    obj[total] = -sum(T[i][total] for i in range(m))   # = -z

    while True:
        # Bland: entering variable = smallest index with negative reduced cost.
        enter = -1
        for j in range(total):
            if obj[j] < 0:
                enter = j
                break
        if enter < 0:
            break
        # Ratio test, ties broken by smallest basic variable index (Bland).
        leave = -1
        best = None
        for i in range(m):
            a = T[i][enter]
            if a > 0:
                ratio = T[i][total] / a
                if best is None or ratio < best or (ratio == best and basis[i] < basis[leave]):
                    best = ratio
                    leave = i
        if leave < 0:
            raise LPError("phase-I objective unbounded below: this cannot happen")
        _pivot(T, obj, basis, leave, enter, total)

    z = -obj[total]
    return z == 0


def _pivot(T, obj, basis, r, c, total):
    prow = T[r]
    p = prow[c]
    if p != 1:
        inv = ONE / p
        for j in range(total + 1):
            if prow[j] != 0:
                prow[j] = check(prow[j] * inv)
    for i, row in enumerate(T):
        if i != r and row[c] != 0:
            f = row[c]
            for j in range(total + 1):
                if prow[j] != 0:
                    row[j] -= f * prow[j]
            row[c] = ZERO
    if obj[c] != 0:
        f = obj[c]
        for j in range(total + 1):
            if prow[j] != 0:
                obj[j] = check(obj[j] - f * prow[j])
        obj[c] = ZERO
    basis[r] = c


# ---------------------------------------------------------------------------
# Geometric helpers.  A "constraint" is a pair (w, c) with w a tuple of n
# Fractions and c a Fraction, meaning the affine function  w.x + c.
# ---------------------------------------------------------------------------

def _split_row(w, extra):
    """Row for free x = y+ - y- followed by ``extra`` coefficients."""
    return [Fraction(v) for v in w] + [-Fraction(v) for v in w] + list(extra)


def open_polyhedron_nonempty(n, constraints, equalities=()):
    """Is {x in R^n : w.x + c > 0 for (w,c) in constraints, w.x + c = 0 for (w,c) in equalities} non-empty?

    Equivalently: does the set have non-empty relative interior inside the
    affine subspace given by ``equalities``?  (With no equalities this is the
    question "is the open polyhedron non-empty / full-dimensional".)
    """
    num_vars = 2 * n + 1              # y+, y-, s
    ge = [(_split_row(w, [c]), ONE) for (w, c) in constraints]
    eq = [(_split_row(w, [c]), ZERO) for (w, c) in equalities]
    return lp_feasible(num_vars, ge, eq)


def closed_polyhedron_nonempty(n, constraints):
    """Is {x in R^n : w.x + c >= 0 for all (w, c)} non-empty?"""
    num_vars = 2 * n
    ge = [(_split_row(w, []), -Fraction(c)) for (w, c) in constraints]
    return lp_feasible(num_vars, ge, [])
