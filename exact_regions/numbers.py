"""Exact rational numbers with a configurable size guard.

All arithmetic in this package uses ``fractions.Fraction`` (arbitrary-precision
integers), so nothing is ever rounded.  To honour the requirement that the
program *fails loudly* instead of silently working with numbers it cannot
handle, every number produced by a simplex pivot and every parsed input is
checked against ``MAX_BITS``: if a numerator or denominator exceeds that many
bits a ``PrecisionOverflowError`` is raised.
"""
from fractions import Fraction
from decimal import Decimal
import re

_state = {"max_bits": 100_000}


class PrecisionOverflowError(ArithmeticError):
    """Raised when an exact rational number becomes larger than the configured limit."""


def set_max_bits(bits):
    """Set the maximal bit length allowed for numerators / denominators (None = unlimited)."""
    _state["max_bits"] = bits


def get_max_bits():
    return _state["max_bits"]


def check(fr):
    """Return ``fr`` unchanged, or raise PrecisionOverflowError if it is too large."""
    mb = _state["max_bits"]
    if mb is not None and (fr.numerator.bit_length() > mb or fr.denominator.bit_length() > mb):
        raise PrecisionOverflowError(
            "exact rational number exceeds %d bits (numerator %d bits, denominator %d bits); "
            "raise the limit with set_max_bits(...) if you really want to continue"
            % (mb, fr.numerator.bit_length(), fr.denominator.bit_length()))
    return fr


_FRACTION_RE = re.compile(r"^\s*([+-]?\d+)\s*/\s*([+-]?\d+)\s*$")


def to_fraction(value):
    """Convert an input number to an exact Fraction.

    Accepted forms:
      * ``int`` / ``Fraction`` / ``Decimal``            -> exact
      * ``str`` such as ``"3"``, ``"-1/3"``, ``"0.25"``, ``"2.5e-3"`` -> exact decimal / fraction parsing
      * ``float`` -> interpreted through its shortest decimal representation
        (``0.1`` becomes 1/10, *not* the binary value of the float).  Pass strings
        if you want to be certain.
    """
    if isinstance(value, bool):
        raise TypeError("booleans are not valid numbers")
    if isinstance(value, Fraction):
        return check(value)
    if isinstance(value, int):
        return check(Fraction(value))
    if isinstance(value, Decimal):
        return check(Fraction(value))
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite float %r" % value)
        return check(Fraction(repr(value)))
    if isinstance(value, str):
        s = value.strip()
        m = _FRACTION_RE.match(s)
        if m:
            den = int(m.group(2))
            if den == 0:
                raise ZeroDivisionError("zero denominator in %r" % value)
            return check(Fraction(int(m.group(1)), den))
        try:
            return check(Fraction(Decimal(s)))
        except Exception as exc:  # pragma: no cover - message only
            raise ValueError("cannot parse %r as an exact number" % value) from exc
    raise TypeError("unsupported number type %r" % type(value).__name__)


def fmt(fr):
    """Short human readable form of a Fraction."""
    return str(fr.numerator) if fr.denominator == 1 else "%d/%d" % (fr.numerator, fr.denominator)
