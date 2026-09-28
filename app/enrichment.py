"""Isotope-enrichment fitting by grid search over candidate abundances.

For each candidate abundance of a single isotope label (e.g. 13C), the
labelled element's natural isotope distribution is replaced by one in which
the labelled isotope carries the candidate abundance while the remaining
isotopes keep their natural ratios. The molecular envelope is rebuilt with
the existing convolution machinery, normalised to base peak = 100 (as are
the observed peaks), and paired with the observed peaks by nearest mass
within a tolerance. The abundance minimising the intensity mean squared
error wins; ties go to the lower abundance.

A dual-label variant (:func:`fit_dual_enrichment`) jointly fits two labels
of *different* elements over a 2-D grid. The natural-abundance envelope of
all other elements (which neither candidate abundance can affect) is
convolved once and reused for every grid point; the two enriched element
polynomials are convolved in per candidate pair. Ties on MSE resolve to the
lexicographically smaller abundance pair (first label's abundance first).
"""

from __future__ import annotations

from dataclasses import dataclass

from .envelope import Peak, convolve, label_mass_shift, merge_peaks, power_polynomial
from .isotopes import ISOTOPES, isotope_by_mass_number
from .parser import ParsedFormula


class PeakListError(ValueError):
    """Validation error for the observed peak list, carrying the 0-based
    index of the offending peak (``None`` when not index-specific)."""

    def __init__(self, message: str, index: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.index = index


class LabelCombinationError(ValueError):
    """Validation error for a multi-label fit request, carrying the 0-based
    index (within the label list) of the offending label."""

    def __init__(self, message: str, index: int) -> None:
        super().__init__(message)
        self.message = message
        self.index = index


def validate_observed_peaks(peaks: list[Peak]) -> None:
    """Require a non-empty list of strictly mass-increasing observed peaks."""
    if not peaks:
        raise PeakListError("Observed peak list must contain at least one peak")
    for i in range(1, len(peaks)):
        prev, cur = peaks[i - 1][0], peaks[i][0]
        if cur == prev:
            raise PeakListError(
                f"Duplicate peak mass {cur}; masses must be strictly increasing", i
            )
        if cur < prev:
            raise PeakListError(
                f"Peak masses must be sorted in increasing order; "
                f"mass {cur} at index {i} follows {prev}",
                i,
            )
    if max(i for _, i in peaks) <= 0:
        raise PeakListError(
            "At least one observed peak must have a positive intensity"
        )


def enriched_distribution(
    symbol: str, mass_number: int, fraction: float
) -> tuple[tuple[float, float], ...]:
    """Isotope distribution of ``symbol`` with the ``mass_number`` isotope
    enriched to ``fraction``; the other natural isotopes keep their natural
    abundance ratios and share the remaining ``1 - fraction``. Zero-abundance
    entries are dropped so the result keeps the positive-abundance invariant
    of the convolution machinery."""
    natural = ISOTOPES[symbol]
    labelled_mass = isotope_by_mass_number(symbol, mass_number)[0]
    others = [(m, a) for m, a in natural if round(m) != mass_number]
    if not others:
        # Single-isotope element: enrichment cannot change the distribution.
        return ((labelled_mass, 1.0),)
    scale = (1.0 - fraction) / sum(a for _, a in others)
    dist = [(m, a * scale) for m, a in others if a * scale > 0.0]
    if fraction > 0.0:
        dist.append((labelled_mass, fraction))
    dist.sort()
    return tuple(dist)


def abundance_grid(step_percent: float) -> list[float]:
    """Candidate abundances in percent: 0, step, 2*step, ..., 100.

    The 100% endpoint is always included (appended when the step does not
    land on it exactly).
    """
    n = int(100.0 / step_percent + 1e-9)
    grid = [i * step_percent for i in range(n + 1)]
    if grid[-1] < 100.0 - 1e-9:
        grid.append(100.0)
    return grid


def normalise_base_peak(peaks: list[Peak]) -> list[Peak]:
    """Rescale intensities so the base (most intense) peak is 100."""
    base = max(i for _, i in peaks)
    return [(m, 100.0 * i / base) for m, i in peaks]


def pair_and_score(
    observed: list[Peak], predicted: list[Peak], tolerance: float
) -> tuple[list[tuple[int | None, int | None]], float]:
    """Pair observed and predicted peaks by nearest mass within ``tolerance``.

    Greedy global matching: all within-tolerance (observed, predicted)
    candidate pairs are considered, smallest mass difference first; each
    peak takes part in at most one pair. Unmatched peaks are paired with a
    zero-intensity placeholder. Returns the pairs as
    ``(observed_index | None, predicted_index | None)`` plus the intensity
    mean squared error over all pairs.
    """
    candidates: list[tuple[float, int, int]] = []
    j0 = 0
    n_pred = len(predicted)
    for i, (m_obs, _) in enumerate(observed):
        lo, hi = m_obs - tolerance, m_obs + tolerance
        while j0 < n_pred and predicted[j0][0] < lo:
            j0 += 1
        j = j0
        while j < n_pred and predicted[j][0] <= hi:
            candidates.append((abs(predicted[j][0] - m_obs), i, j))
            j += 1
    candidates.sort()

    matched_obs: dict[int, int] = {}
    used_pred: set[int] = set()
    for _, i, j in candidates:
        if i not in matched_obs and j not in used_pred:
            matched_obs[i] = j
            used_pred.add(j)

    pairs: list[tuple[int | None, int | None]] = []
    squared_error = 0.0
    for i, (_, i_obs) in enumerate(observed):
        j = matched_obs.get(i)
        if j is None:
            pairs.append((i, None))
            squared_error += i_obs * i_obs
        else:
            pairs.append((i, j))
            diff = i_obs - predicted[j][1]
            squared_error += diff * diff
    for j, (_, i_pred) in enumerate(predicted):
        if j not in used_pred:
            pairs.append((None, j))
            squared_error += i_pred * i_pred
    return pairs, squared_error / len(pairs)


@dataclass
class FitResult:
    best_abundance_percent: float
    mse: float
    search_points: int
    # (observed_index | None, predicted_index | None) pairs at the best fit
    pairs: list[tuple[int | None, int | None]]
    predicted: list[Peak]  # normalised envelope at the best abundance


def _envelope_excluding(
    parsed: ParsedFormula,
    symbols: str | tuple[str, ...],
    tolerance: float,
    prune: float,
) -> list[Peak]:
    """Natural-abundance envelope of every element except ``symbols``
    (computed once and reused across all candidate abundances)."""
    excluded = (symbols,) if isinstance(symbols, str) else symbols
    envelope: list[Peak] = [(0.0, 1.0)]
    for sym, count in parsed.composition.items():
        if sym in excluded:
            continue
        element_poly = power_polynomial(list(ISOTOPES[sym]), count, tolerance, prune)
        envelope = convolve(envelope, element_poly, tolerance, prune)
    return envelope


def _combine(base: list[Peak], extra: list[Peak], tolerance: float, prune: float) -> list[Peak]:
    """Convolve ``extra`` into ``base``, iterating the smaller operand
    first so the merge sort has fewer, longer runs to fuse."""
    if len(extra) <= len(base):
        return convolve(extra, base, tolerance, prune)
    return convolve(base, extra, tolerance, prune)


def _with_label_shift(parsed: ParsedFormula, envelope: list[Peak], tolerance: float) -> list[Peak]:
    """Apply the fixed mass shift of any pure-isotope labels and merge."""
    shift = label_mass_shift(parsed)
    if shift:
        envelope = [(m + shift, i) for m, i in envelope]
    return merge_peaks(envelope, tolerance)


def _element_polynomial(
    parsed: ParsedFormula,
    symbol: str,
    distribution: tuple[tuple[float, float], ...],
    tolerance: float,
    prune: float,
) -> list[Peak] | None:
    """``symbol`` raised to its atom count under ``distribution``; ``None``
    when the formula contains no atoms of that element."""
    count = parsed.composition.get(symbol, 0)
    if not count:
        return None
    return power_polynomial(list(distribution), count, tolerance, prune)


def _candidate_envelope(
    parsed: ParsedFormula,
    symbol: str,
    distribution: tuple[tuple[float, float], ...],
    others_envelope: list[Peak],
    tolerance: float,
    prune: float,
) -> list[Peak]:
    """Full envelope with ``symbol``'s distribution replaced by ``distribution``."""
    envelope = others_envelope
    element_poly = _element_polynomial(parsed, symbol, distribution, tolerance, prune)
    if element_poly is not None:
        envelope = _combine(envelope, element_poly, tolerance, prune)
    return _with_label_shift(parsed, envelope, tolerance)


@dataclass
class DualFitResult:
    best_abundances_percent: tuple[float, float]
    mse: float
    search_points: int
    # (observed_index | None, predicted_index | None) pairs at the best fit
    pairs: list[tuple[int | None, int | None]]
    predicted: list[Peak]  # normalised envelope at the best abundance pair


def fit_enrichment(
    parsed: ParsedFormula,
    symbol: str,
    mass_number: int,
    observed: list[Peak],
    step_percent: float,
    mass_tolerance: float,
    merge_tolerance: float,
    prune: float,
) -> FitResult:
    """Grid-search the label abundance (0–100%) best matching the observed
    peaks. ``observed`` must be sorted by strictly increasing mass with
    non-negative intensities, at least one of them positive."""
    obs = normalise_base_peak(observed)
    others = _envelope_excluding(parsed, symbol, merge_tolerance, prune)
    grid = abundance_grid(step_percent)

    best: FitResult | None = None
    for pct in grid:
        dist = enriched_distribution(symbol, mass_number, pct / 100.0)
        env = _candidate_envelope(parsed, symbol, dist, others, merge_tolerance, prune)
        pred = normalise_base_peak(env)
        pairs, mse = pair_and_score(obs, pred, mass_tolerance)
        # Strict improvement only: on a tie the lower abundance already
        # visited is kept.
        if best is None or mse < best.mse:
            best = FitResult(pct, mse, len(grid), pairs, pred)
    # The grid always spans 0–100%, so a best candidate always exists.
    assert best is not None
    return best


def fit_dual_enrichment(
    parsed: ParsedFormula,
    label1: tuple[str, int],
    label2: tuple[str, int],
    observed: list[Peak],
    step_percent: tuple[float, float],
    mass_tolerance: float,
    merge_tolerance: float,
    prune: float,
) -> DualFitResult:
    """Joint 2-D grid search over the abundances of two labels belonging to
    *different* elements. Raises :class:`LabelCombinationError` when both
    labels name the same element.

    The natural-abundance envelope of every other element is built once;
    each labelled element's enriched polynomial is rebuilt only when its own
    abundance changes. MSE ties resolve lexicographically: the lower
    abundance of the first label wins, then that of the second label.
    """
    (symbol1, mass_number1), (symbol2, mass_number2) = label1, label2
    if symbol1 == symbol2:
        raise LabelCombinationError(
            f"The two labels must target different elements, but both "
            f"target '{symbol1}'",
            1,
        )

    obs = normalise_base_peak(observed)
    others = _envelope_excluding(parsed, (symbol1, symbol2), merge_tolerance, prune)
    grid1 = abundance_grid(step_percent[0])
    grid2 = abundance_grid(step_percent[1])
    search_points = len(grid1) * len(grid2)

    best: DualFitResult | None = None
    # Grids ascend, so this visit order realises the lexicographic
    # (abundance1, abundance2) tie-break by accepting strict gains only.
    for pct1 in grid1:
        dist1 = enriched_distribution(symbol1, mass_number1, pct1 / 100.0)
        poly1 = _element_polynomial(parsed, symbol1, dist1, merge_tolerance, prune)
        base1 = _combine(others, poly1, merge_tolerance, prune) if poly1 is not None else others
        for pct2 in grid2:
            dist2 = enriched_distribution(symbol2, mass_number2, pct2 / 100.0)
            poly2 = _element_polynomial(parsed, symbol2, dist2, merge_tolerance, prune)
            envelope = _combine(base1, poly2, merge_tolerance, prune) if poly2 is not None else base1
            env = _with_label_shift(parsed, envelope, merge_tolerance)
            pred = normalise_base_peak(env)
            pairs, mse = pair_and_score(obs, pred, mass_tolerance)
            if best is None or mse < best.mse:
                best = DualFitResult((pct1, pct2), mse, search_points, pairs, pred)
    # Both grids always span 0–100%, so a best candidate always exists.
    assert best is not None
    return best
