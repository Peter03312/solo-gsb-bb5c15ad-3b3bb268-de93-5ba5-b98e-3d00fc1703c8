"""Isotope-envelope computation via mass/intensity polynomial convolution.

Each element's isotope pattern is a polynomial  sum(a_k * x^mass_k)  over its
natural isotopes; the molecular envelope is the product of the per-element
polynomials (each raised to its atom count). Multiplication of polynomials
corresponds to convolving the (mass, intensity) peak lists. Pure-isotope
labels contribute a fixed mass shift with unit probability.
"""

from __future__ import annotations

from .isotopes import ISOTOPES, isotope_by_mass_number, most_abundant_isotope_mass
from .parser import ParsedFormula

Peak = tuple[float, float]  # (mass Da, probability/intensity)

# Electron rest mass in daltons, used when converting ionic mass to m/z.
ELECTRON_MASS_DA = 0.000548579909065


def merge_peaks(peaks: list[Peak], tolerance: float) -> list[Peak]:
    """Merge peaks whose mass difference is below ``tolerance`` Da.

    Peaks are sorted by mass; adjacent peaks closer than the tolerance are
    fused into one whose mass is the intensity-weighted mean and whose
    intensity is the sum of the group.
    """
    if not peaks:
        return []
    ordered = sorted(peaks)
    merged: list[Peak] = []
    cur_mass, cur_int = ordered[0]
    for mass, inten in ordered[1:]:
        if mass - cur_mass < tolerance:
            total = cur_int + inten
            cur_mass = (cur_mass * cur_int + mass * inten) / total
            cur_int = total
        else:
            merged.append((cur_mass, cur_int))
            cur_mass, cur_int = mass, inten
    merged.append((cur_mass, cur_int))
    return merged


def convolve(p1: list[Peak], p2: list[Peak], tolerance: float, prune: float) -> list[Peak]:
    """Convolve two peak lists (multiply the underlying polynomials)."""
    out: list[Peak] = [
        (m1 + m2, i1 * i2)
        for m1, i1 in p1
        for m2, i2 in p2
    ]
    out = merge_peaks(out, tolerance)
    if prune > 0 and out:
        top = max(i for _, i in out)
        out = [(m, i) for m, i in out if i >= top * prune]
    return out


def power_polynomial(base: list[Peak], n: int, tolerance: float, prune: float) -> list[Peak]:
    """Raise an element's isotope polynomial to the atom count (binary
    exponentiation keeps large counts cheap)."""
    result: list[Peak] = [(0.0, 1.0)]
    b = base
    while n:
        if n & 1:
            result = convolve(result, b, tolerance, prune)
        n >>= 1
        if n:
            b = convolve(b, b, tolerance, prune)
    return result


def label_mass_shift(parsed: ParsedFormula) -> float:
    """Total mass shift contributed by pure-isotope labels."""
    shift = 0.0
    for (symbol, mass_number), count in parsed.labels.items():
        iso = isotope_by_mass_number(symbol, mass_number)
        shift += iso[0] * count
    return shift


def monoisotopic_mass(parsed: ParsedFormula) -> float:
    """Sum of the most-abundant-isotope masses, plus label mass shifts."""
    mass = label_mass_shift(parsed)
    for symbol, count in parsed.composition.items():
        mass += most_abundant_isotope_mass(symbol) * count
    return mass


def isotope_envelope(
    parsed: ParsedFormula, tolerance: float, prune: float
) -> list[Peak]:
    """Full (unnormalised) isotope envelope as sorted (mass, probability)."""
    envelope: list[Peak] = [(0.0, 1.0)]
    for symbol, count in parsed.composition.items():
        base = list(ISOTOPES[symbol])
        element_poly = power_polynomial(base, count, tolerance, prune)
        envelope = convolve(envelope, element_poly, tolerance, prune)
    shift = label_mass_shift(parsed)
    if shift:
        envelope = [(m + shift, i) for m, i in envelope]
    return merge_peaks(envelope, tolerance)


def top_peaks(envelope: list[Peak], top_n: int) -> list[Peak]:
    """Normalise to base peak = 100 and keep the ``top_n`` most intense
    peaks, returned sorted by mass."""
    if not envelope:
        return []
    base = max(i for _, i in envelope)
    normalised = [(m, 100.0 * i / base) for m, i in envelope]
    strongest = sorted(normalised, key=lambda p: p[1], reverse=True)[:top_n]
    return sorted(strongest)


def mass_to_mz(mass: float, charge: int) -> float:
    """Convert an ionic mass (Da) to m/z for an ion with the given charge.

    A positive charge z corresponds to removing z electrons, so the ion mass
    is ``neutral_mass - z * m_e``; a negative charge adds ``|z|`` electrons.
    The m/z is then that ionic mass divided by ``|z|``.
    """
    return (mass - charge * ELECTRON_MASS_DA) / abs(charge)


def ion_envelope(envelope: list[Peak], charge: int) -> list[Peak]:
    """Map a neutral/ionic mass peak list to (m/z, intensity) via
    :func:`mass_to_mz`. Division by the positive ``|charge|`` is strictly
    monotonic, so peak order and distinctness are preserved."""
    return [(mass_to_mz(m, charge), i) for m, i in envelope]
