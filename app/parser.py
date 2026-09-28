"""Molecular-formula parser with position-aware validation errors.

Grammar (case-sensitive)::

    formula := term+
    term    := element count? | label count?
    element := [A-Z] [a-z]?
    label   := "[" digit+ element "]"      e.g. [13C], [15N], [2H]
    count   := digit+                      positive integer, default 1

Examples: ``C6H12O6``, ``C6H6[13C]2``, ``C2H5[15N]O[18O]``.

A label ``[13C]2`` contributes two atoms of *pure* 13C in addition to any
natural-abundance ``C`` atoms listed elsewhere in the formula.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .isotopes import ISOTOPES, isotope_by_mass_number

_ELEMENT_RE = re.compile(r"[A-Z][a-z]?")
_LABEL_BODY_RE = re.compile(r"(\d+)([A-Z][a-z]?)")


class FormulaError(ValueError):
    """Validation error carrying the 0-based character position in the formula."""

    def __init__(self, message: str, position: int) -> None:
        super().__init__(message)
        self.message = message
        self.position = position


class LabelError(ValueError):
    """Validation error for a bare isotope label, carrying the 0-based
    character position within the label string."""

    def __init__(self, message: str, position: int) -> None:
        super().__init__(message)
        self.message = message
        self.position = position


@dataclass
class ParsedFormula:
    # natural-abundance element symbol -> atom count
    composition: dict[str, int] = field(default_factory=dict)
    # (element symbol, mass number) -> atom count, for pure-isotope labels
    labels: dict[tuple[str, int], int] = field(default_factory=dict)


def _parse_count(text: str, i: int) -> tuple[int, int]:
    """Parse an optional count starting at ``i``; returns (count, next_index)."""
    j = i
    while j < len(text) and text[j].isdigit():
        j += 1
    if j == i:
        return 1, i
    count = int(text[i:j])
    if count < 1:
        raise FormulaError(
            f"Element count must be a positive integer, got {count}", i
        )
    return count, j


def parse_formula(text: str) -> ParsedFormula:
    """Parse and validate a molecular formula.

    Raises :class:`FormulaError` with a 0-based ``position`` pointing at the
    offending character/token for any malformed input.
    """
    if not isinstance(text, str) or not text.strip():
        raise FormulaError("Formula must be a non-empty string", 0)
    if text != text.strip():
        pos = len(text) - len(text.lstrip()) if text != text.lstrip() else len(text.rstrip())
        raise FormulaError("Formula must not contain whitespace", pos)

    parsed = ParsedFormula()
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "[":
            i = _parse_label(text, i, parsed)
        elif ch.isupper():
            match = _ELEMENT_RE.match(text, i)
            symbol = match.group(0)
            if symbol not in ISOTOPES:
                raise FormulaError(f"Unknown element symbol '{symbol}'", i)
            count, i = _parse_count(text, match.end())
            parsed.composition[symbol] = parsed.composition.get(symbol, 0) + count
        elif ch.islower():
            raise FormulaError(
                f"Element symbols must start with an uppercase letter; "
                f"unexpected '{ch}'",
                i,
            )
        elif ch.isdigit():
            raise FormulaError(f"Unexpected count without a preceding element", i)
        else:
            raise FormulaError(f"Unexpected character '{ch}'", i)

    if not parsed.composition and not parsed.labels:
        raise FormulaError("Formula contains no elements", 0)
    return parsed


def _parse_label(text: str, i: int, parsed: ParsedFormula) -> int:
    """Parse ``[massNumberElement]count`` starting at ``i`` (the '[')."""
    close = text.find("]", i + 1)
    if close == -1:
        raise FormulaError("Unterminated isotope label: missing ']'", i)
    body = text[i + 1 : close]
    match = _LABEL_BODY_RE.fullmatch(body)
    if not match:
        raise FormulaError(
            f"Malformed isotope label '[{body}]'; expected e.g. '[13C]'",
            i,
        )
    mass_number = int(match.group(1))
    symbol = match.group(2)
    symbol_pos = i + 1 + len(match.group(1))
    if symbol not in ISOTOPES:
        raise FormulaError(f"Unknown element symbol '{symbol}' in label", symbol_pos)
    if isotope_by_mass_number(symbol, mass_number) is None:
        raise FormulaError(
            f"Element '{symbol}' has no known isotope with mass number "
            f"{mass_number}; check the label mass",
            i,
        )
    count, next_i = _parse_count(text, close + 1)
    key = (symbol, mass_number)
    parsed.labels[key] = parsed.labels.get(key, 0) + count
    return next_i


def parse_label(text: str) -> tuple[str, int]:
    """Parse and validate a bare isotope label such as ``'13C'``.

    Returns ``(element_symbol, mass_number)``. Raises :class:`LabelError`
    with a 0-based ``position`` pointing at the offending token.
    """
    if not isinstance(text, str) or not text.strip():
        raise LabelError("Label must be a non-empty string, e.g. '13C'", 0)
    match = _LABEL_BODY_RE.fullmatch(text)
    if not match:
        raise LabelError(
            f"Malformed label '{text}'; expected a mass number followed by "
            f"an element symbol, e.g. '13C'",
            0,
        )
    mass_number = int(match.group(1))
    symbol = match.group(2)
    symbol_pos = len(match.group(1))
    if symbol not in ISOTOPES:
        raise LabelError(f"Unknown element symbol '{symbol}' in label", symbol_pos)
    if isotope_by_mass_number(symbol, mass_number) is None:
        raise LabelError(
            f"Element '{symbol}' has no known isotope with mass number "
            f"{mass_number}; check the label mass",
            0,
        )
    return symbol, mass_number
