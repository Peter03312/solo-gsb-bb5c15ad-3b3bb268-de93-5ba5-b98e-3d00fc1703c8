"""Pairwise ion-mixture fitting over candidate adduct/charge envelopes.

Each valid candidate reuses the base-peak-normalised ionic m/z envelope of
:mod:`app.identify`. For every unordered pair of valid candidates (ordered
by request index) and every mixture ratio on a 0-100% grid, the two
normalised envelopes are scaled by their ratio and overlaid — peaks of the
two envelopes closer than the mass tolerance are fused — and the mixed
spectrum is re-normalised to base peak 100 before being paired one-to-one
with the (also normalised) observed peaks. The MSE over all pairs,
unmatched peaks on both sides included, ranks the combinations; ties
resolve to the lexicographically smaller candidate index pair, then to the
lower first-candidate ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enrichment import abundance_grid, normalise_base_peak, pair_and_score
from .envelope import merge_peaks
from .identify import IonCandidatePrediction, MzPeak


def mix_envelopes(
    first: list[MzPeak],
    second: list[MzPeak],
    first_fraction: float,
    tolerance: float,
) -> list[MzPeak]:
    """Overlay two base-peak-normalised envelopes at ``first_fraction``.

    Each envelope contributes its normalised intensities scaled by its
    mixture fraction; peaks closer than ``tolerance`` are fused
    (intensity-weighted mass, summed intensity) and the mixed spectrum is
    re-normalised to base peak 100. A zero fraction skips that envelope
    entirely, so pure 0%/100% mixtures reproduce the other envelope.
    """
    second_fraction = 1.0 - first_fraction
    scaled: list[MzPeak] = []
    if first_fraction > 0.0:
        scaled.extend((m, i * first_fraction) for m, i in first)
    if second_fraction > 0.0:
        scaled.extend((m, i * second_fraction) for m, i in second)
    return normalise_base_peak(merge_peaks(scaled, tolerance))


def score_mixture(
    first: list[MzPeak],
    second: list[MzPeak],
    first_fraction: float,
    observed: list[MzPeak],
    tolerance: float,
) -> tuple[list[tuple[int | None, int | None]], float, list[MzPeak]]:
    """Mix two envelopes and pair the result with the observed peaks.

    Returns the one-to-one pairs, the MSE (unmatched peaks counted at zero
    intensity) and the re-normalised mixed spectrum.
    """
    mixed = mix_envelopes(first, second, first_fraction, tolerance)
    pairs, mse = pair_and_score(observed, mixed, tolerance)
    return pairs, mse, mixed


@dataclass
class MixtureCombination:
    """One scored (candidate pair, mixture ratio) combination."""

    first_index: int
    second_index: int
    first_fraction_percent: float
    mse: float


def fit_ion_mixture(
    predictions: list[tuple[int, IonCandidatePrediction]],
    observed: list[MzPeak],
    step_percent: float,
    mass_tolerance: float,
) -> list[MixtureCombination]:
    """Score every (valid candidate pair, mixture ratio) combination.

    ``predictions`` holds ``(request_index, prediction)`` of the valid
    candidates in request order; ``observed`` must already be base-peak
    normalised. Combinations are returned sorted by ascending MSE, then by
    candidate index pair, then by first-candidate ratio, so the first
    entry realises the tie-break rules.
    """
    grid = abundance_grid(step_percent)
    combinations: list[MixtureCombination] = []
    for a in range(len(predictions) - 1):
        first_index, first = predictions[a]
        for b in range(a + 1, len(predictions)):
            second_index, second = predictions[b]
            for pct in grid:
                _, mse, _ = score_mixture(
                    first.peaks, second.peaks, pct / 100.0, observed, mass_tolerance
                )
                combinations.append(
                    MixtureCombination(first_index, second_index, pct, mse)
                )
    combinations.sort(
        key=lambda c: (
            c.mse,
            c.first_index,
            c.second_index,
            c.first_fraction_percent,
        )
    )
    return combinations


__all__ = [
    "MixtureCombination",
    "fit_ion_mixture",
    "mix_envelopes",
    "score_mixture",
]
