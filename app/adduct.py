"""Parser for adduct specifications such as '+2H-H2O' or '+Na-H'.

Grammar (case-sensitive)::

    adduct := piece+
    piece   := sign coefficient? fragment
    sign    := '+' | '-'
    coefficient := digit+             positive integer, default 1
    fragment := molecular formula     same grammar as ``parser.parse_formula``
                                      (elements plus optional [13C]-style labels)

The coefficient multiplies the *whole* fragment: '+2H' adds two H atoms and
'-H2O' removes one H2O unit. The string must be consumed completely; each
piece starts with an explicit '+' or '-'.

Examples: ``+H`` (protonated molecule), ``+2H`` (doubly protonated),
``+Na-H`` (sodium adduct minus a hydrogen), ``+2H-H2O``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .parser import FormulaError, ParsedFormula, parse_formula


class AdductError(ValueError):
    """Validation error carrying the 0-based character position in the
    adduct string."""

    def __init__(self, message: str, position: int) -> None:
        super().__init__(message)
        self.message = message
        self.position = position


@dataclass
class AdductPiece:
    # +1 for an additive fragment, -1 for a subtractive one
    sign: int
    # Positive integer multiplying every atom count in the fragment
    coefficient: int
    # Parsed molecular-formula fragment
    fragment: ParsedFormula
    # 0-based position of the fragment's first character (after sign/coefficient)
    fragment_start: int


def parse_adduct(text: str) -> list[AdductPiece]:
    """Parse and validate an adduct specification.

    Raises :class:`AdductError` with a 0-based ``position`` pointing at the
    offending character. Formula-level errors inside a fragment keep their
    relative position translated to the absolute position in ``text``.
    """
    if not isinstance(text, str) or not text.strip():
        raise AdductError(
            "Adduct must be a non-empty string, e.g. '+H', '+Na-H' or '+2H-H2O'",
            0,
        )
    for pos, ch in enumerate(text):
        if ch.isspace():
            raise AdductError("Adduct must not contain whitespace", pos)

    pieces: list[AdductPiece] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch not in "+-":
            raise AdductError(
                "Each adduct fragment must start with '+' or '-', e.g. '+2H-H2O'",
                i,
            )
        sign = 1 if ch == "+" else -1
        i += 1

        # Optional coefficient: the run of digits directly after the sign.
        coeff_start = i
        j = i
        while j < n and text[j].isdigit():
            j += 1
        coefficient = 1
        if j > i:
            coefficient = int(text[i:j])
            if coefficient < 1:
                raise AdductError(
                    f"Fragment coefficient must be a positive integer, got "
                    f"{coefficient}",
                    coeff_start,
                )
            i = j

        # The fragment runs until the next sign (or end of string).
        fragment_start = i
        end = i
        while end < n and text[end] not in "+-":
            end += 1
        try:
            fragment = parse_formula(text[fragment_start:end])
        except FormulaError as exc:
            raise AdductError(
                exc.message, fragment_start + exc.position
            ) from exc
        pieces.append(
            AdductPiece(sign, coefficient, fragment, fragment_start)
        )
        i = end
    return pieces


def apply_adduct(
    parsed: ParsedFormula, pieces: list[AdductPiece]
) -> ParsedFormula:
    """Apply adduct fragments to a parsed neutral formula.

    Fragments are applied in order; if any addition/subtraction drives an
    element (or isotope-label) count below zero, :class:`AdductError` is
    raised whose ``position`` points at the first character of the fragment
    causing the first negative count.
    """
    composition = dict(parsed.composition)
    labels = dict(parsed.labels)

    for piece in pieces:
        delta = piece.sign * piece.coefficient
        for symbol, count in piece.fragment.composition.items():
            new_count = composition.get(symbol, 0) + delta * count
            if new_count < 0:
                raise AdductError(
                    f"Adduct fragment would drive element '{symbol}' count "
                    f"from {composition.get(symbol, 0)} to {new_count}; "
                    f"counts must not become negative",
                    piece.fragment_start,
                )
            composition[symbol] = new_count
        for key, count in piece.fragment.labels.items():
            new_count = labels.get(key, 0) + delta * count
            if new_count < 0:
                symbol, mass_number = key
                raise AdductError(
                    f"Adduct fragment would drive [{mass_number}{symbol}] "
                    f"label count from {labels.get(key, 0)} to {new_count}; "
                    f"counts must not become negative",
                    piece.fragment_start,
                )
            labels[key] = new_count

    return ParsedFormula(composition=composition, labels=labels)
