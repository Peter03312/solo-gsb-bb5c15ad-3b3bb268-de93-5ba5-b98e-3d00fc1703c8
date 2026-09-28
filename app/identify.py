"""Ion-identification scoring for candidate adduct/charge combinations.

Given a neutral formula and a list of observed m/z peaks, each candidate
(adduct string, signed charge) is evaluated with the same machinery as
``/api/v1/ion-envelope``: the adduct is parsed and applied to the parsed
atom counts, the isotope envelope is convolved with the neutral merge
tolerance scaled by |charge|, normalised to base peak 100, and converted to
m/z. Predicted and observed (also base-peak normalised) peaks are then paired
one-to-one by smallest mass difference within tolerance; unmatched peaks
count at zero intensity in the normalised MSE.
"""

from __future__ import annotations

from dataclasses import dataclass

from .adduct import AdductError, apply_adduct, parse_adduct
from .enrichment import normalise_base_peak, pair_and_score
from .envelope import (
    ELECTRON_MASS_DA,
    ion_envelope,
    isotope_envelope,
    monoisotopic_mass,
)
from .parser import ParsedFormula

# (m/z, intensity) m/z peak; mirrors envelope.Peak but on the m/z axis.
MzPeak = tuple[float, float]


@dataclass
class IonCandidatePrediction:
    """Base-peak normalised m/z envelope predicted by one candidate."""

    adduct: str
    charge: int
    composition: dict[str, int]
    labels: dict[str, int]
    monoisotopic_mz: float
    peaks: list[MzPeak]


def predict_ion_candidate(
    parsed: ParsedFormula,
    adduct: str,
    charge: int,
    merge_tolerance: float,
    prune: float,
) -> IonCandidatePrediction:
    """Apply the adduct and build the normalised ionic m/z envelope.

    Raises :class:`AdductError` (with its character ``position``) exactly as
    ``parse_adduct``/``apply_adduct`` do.
    """
    pieces = parse_adduct(adduct)
    adjusted = apply_adduct(parsed, pieces)

    # Same scaling as /api/v1/ion-envelope: isotope peaks of higher charge
    # states sit closer together in m/z.
    env = isotope_envelope(adjusted, abs(charge) * merge_tolerance, prune)
    pred = normalise_base_peak(env)
    mz_peaks = ion_envelope(pred, charge)
    mono_mz = (
        monoisotopic_mass(adjusted) - charge * ELECTRON_MASS_DA
    ) / abs(charge)

    return IonCandidatePrediction(
        adduct=adduct,
        charge=charge,
        composition=dict(adjusted.composition),
        labels={
            f"[{mass_no}{symbol}]": count
            for (symbol, mass_no), count in adjusted.labels.items()
        },
        monoisotopic_mz=mono_mz,
        peaks=mz_peaks,
    )


@dataclass
class IonCandidateScore:
    """Result of scoring one candidate against the observed peaks."""

    index: int
    adduct: str
    charge: int
    mse: float
    monoisotopic_mz: float
    composition: dict[str, int]
    labels: dict[str, int]
    pairs: list[tuple[int | None, int | None]]
    predicted: list[MzPeak]


def score_ion_candidate(
    index: int,
    prediction: IonCandidatePrediction,
    observed: list[MzPeak],
    mass_tolerance: float,
) -> IonCandidateScore:
    """Pair one candidate's predicted m/z peaks with the observed peaks and
    compute the normalised MSE (unmatched peaks counted at zero intensity)."""
    pairs, mse = pair_and_score(observed, prediction.peaks, mass_tolerance)
    return IonCandidateScore(
        index=index,
        adduct=prediction.adduct,
        charge=prediction.charge,
        mse=mse,
        monoisotopic_mz=prediction.monoisotopic_mz,
        composition=prediction.composition,
        labels=prediction.labels,
        pairs=pairs,
        predicted=prediction.peaks,
    )


__all__ = [
    "AdductError",
    "IonCandidatePrediction",
    "IonCandidateScore",
    "predict_ion_candidate",
    "score_ion_candidate",
]
