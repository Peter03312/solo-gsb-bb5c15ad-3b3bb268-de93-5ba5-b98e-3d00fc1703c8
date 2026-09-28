"""FastAPI application: mass-spectrometry isotope-envelope precompute API."""

from __future__ import annotations

import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, Field, field_validator

from . import __version__, db
from .adduct import AdductError, apply_adduct, parse_adduct
from .calibration import (
    MAX_POINTS as CALIBRATION_MAX_POINTS,
    MIN_POINTS as CALIBRATION_MIN_POINTS,
    CalibrationError,
    CalibrationPoint,
    calibrate_mass_axis,
    inverse_predict,
)
from .centroid import CentroidError, centroid_profile
from .config import settings
from .consensus import ConsensusError, build_consensus_spectrum
from .enrichment import (
    LabelCombinationError,
    PeakListError,
    fit_dual_enrichment,
    fit_enrichment,
    normalise_base_peak,
    validate_observed_peaks,
)
from .envelope import (
    ELECTRON_MASS_DA,
    ion_envelope,
    isotope_envelope,
    monoisotopic_mass,
    top_peaks,
)
from .formula_search import (
    MAX_VISITED_NODES,
    ElementSpec,
    SearchLimitExceeded,
    search_formulas,
)
from .identify import IonCandidatePrediction, predict_ion_candidate, score_ion_candidate
from .isotope_clusters import IsotopeClusterError, detect_isotope_clusters
from .isotopes import ISOTOPES
from .mixture import fit_ion_mixture, score_mixture
from .parser import FormulaError, LabelError, parse_formula, parse_label
from .profile import ProfileParameterError, simulate_profile


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dependencies are installed at image build time; the SQLite schema is
    # created here so a fresh container is ready to serve immediately.
    db.init_db(settings.db_path)
    yield
    db.close_db()


app = FastAPI(
    title="MS Precompute API",
    version=__version__,
    description=(
        "Computes natural-abundance isotope envelopes (mass/intensity "
        "polynomial convolution) and monoisotopic masses for molecular "
        "formulas, with optional pure-isotope labels such as [13C]; also "
        "estimates single-isotope enrichment by fitting candidate "
        "abundances against observed peaks, jointly fits two isotopes of "
        "different elements on a 2-D abundance grid, fits pairwise "
        "0-100% mixtures of candidate adduct/charge ion envelopes, "
        "simulates Gaussian instrument profiles of neutral stick spectra, "
        "centroids equally-spaced continuous mass/intensity profiles, "
        "reverse-searches molecular formulas from a target monoisotopic "
        "mass and observed isotope peaks via branch-and-bound enumeration, "
        "calibrates the mass axis with a robust median-slope initial "
        "model followed by an inlier least-squares line, and detects "
        "unknown isotope clusters de novo in centroid m/z peak lists via "
        "neutron-spacing candidate chains and weighted-interval selection, "
        "and builds a consensus spectrum from replicate centroid peak "
        "lists via deterministic complete-linkage clustering."
    ),
    lifespan=lifespan,
)


class EnvelopeRequest(BaseModel):
    formula: str = Field(
        ...,
        description="Molecular formula, e.g. 'C6H12O6' or 'C6H6[13C]2'",
        examples=["C6H12O6"],
    )
    top_n: int = Field(
        settings.top_n_default,
        ge=1,
        le=settings.top_n_max,
        description="Number of most intense peaks to return",
    )


class PeakOut(BaseModel):
    mass: float
    intensity: float


class EnvelopeResponse(BaseModel):
    formula: str
    composition: dict[str, int]
    labels: dict[str, int]
    monoisotopic_mass: float
    base_peak_intensity: float
    peak_count: int
    peaks: list[PeakOut]
    cached: bool


class IonEnvelopeRequest(BaseModel):
    formula: str = Field(
        ...,
        description="Neutral molecular formula, e.g. 'C6H12O6' or 'C6H6[13C]2'",
        examples=["C6H12O6"],
    )
    adduct: str = Field(
        ...,
        description=(
            "Adduct made of consecutive '+/-[coefficient]formula' fragments; "
            "the coefficient multiplies the whole fragment, e.g. '+H', "
            "'+2H-H2O' or '+Na-H'"
        ),
        examples=["+H"],
    )
    charge: int = Field(
        ...,
        ge=-10,
        le=10,
        description="Signed ion charge; must be -10..-1 or 1..10 (zero is not allowed)",
        examples=[1],
    )
    top_n: int = Field(
        settings.top_n_default,
        ge=1,
        le=settings.top_n_max,
        description="Number of most intense peaks to return",
    )

    @field_validator("charge")
    @classmethod
    def charge_must_be_nonzero(cls, value: int) -> int:
        if value == 0:
            raise ValueError("charge must be non-zero: use -10..-1 or 1..10")
        return value


class MzPeakOut(BaseModel):
    mz: float
    intensity: float


class IonEnvelopeResponse(BaseModel):
    formula: str
    adduct: str
    charge: int
    composition: dict[str, int]
    labels: dict[str, int]
    monoisotopic_mz: float
    base_peak_intensity: float
    peak_count: int
    peaks: list[MzPeakOut]


class NeutralProfileRequest(BaseModel):
    formula: str = Field(
        ...,
        description="Neutral molecular formula, e.g. 'C6H12O6' or 'C6H6[13C]2'",
        examples=["C6H12O6"],
    )
    resolving_power: float = Field(
        ...,
        gt=0,
        description=(
            "Instrument resolving power R = m/FWHM; each stick at mass m is "
            "broadened with FWHM = m/R"
        ),
        examples=[100000],
    )
    mass_step: float = Field(
        ...,
        gt=0,
        description="Uniform output-grid spacing in Da (must be > 0)",
        examples=[0.001],
    )
    mass_min: float | None = Field(
        None,
        gt=0,
        description=(
            "Inclusive lower grid bound in Da; defaults to the smallest "
            "mass covered by any stick's +/-4 sigma kernel"
        ),
    )
    mass_max: float | None = Field(
        None,
        gt=0,
        description=(
            "Inclusive upper grid bound in Da; defaults to the largest mass "
            "covered by any stick's +/-4 sigma kernel"
        ),
    )

    @field_validator("resolving_power", "mass_step", "mass_min", "mass_max")
    @classmethod
    def values_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("value must be a finite number")
        return value


class NeutralProfilePointOut(BaseModel):
    mass: float
    intensity: float


class NeutralProfileResponse(BaseModel):
    formula: str
    resolving_power: float
    mass_min: float
    mass_max: float
    mass_step: float
    point_count: int
    profile: list[NeutralProfilePointOut]


class ProfilePointIn(BaseModel):
    mass: float = Field(
        ...,
        description=(
            "Sample mass (Da); the series must be strictly increasing on a "
            "uniform grid (constant step within a 1e-6 relative error)"
        ),
    )
    intensity: float = Field(
        ...,
        ge=0,
        description="Sample intensity (non-negative; at least one must be positive)",
    )


class ProfileCentroidRequest(BaseModel):
    profile: list[ProfilePointIn] = Field(
        ...,
        min_length=3,
        max_length=20000,
        description=(
            "Equally-spaced continuous profile, 3-20000 points with strictly "
            "increasing masses"
        ),
    )
    relative_threshold: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Points whose base-peak-normalised intensity (base peak = 100) "
            "is below this percentage are discarded before peak picking"
        ),
        examples=[5.0],
    )
    min_separation: float = Field(
        ...,
        gt=0,
        description=(
            "Minimum mass separation (Da) between reported peaks; non-maximum "
            "suppression keeps the strongest candidate within each window"
        ),
        examples=[0.01],
    )

    @field_validator("relative_threshold", "min_separation")
    @classmethod
    def values_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be a finite number")
        return value

    @field_validator("profile")
    @classmethod
    def points_must_be_finite(cls, value: list[ProfilePointIn]) -> list[ProfilePointIn]:
        for point in value:
            if not math.isfinite(point.mass) or not math.isfinite(point.intensity):
                raise ValueError("mass and intensity must be finite numbers")
        return value


class CentroidPeakOut(BaseModel):
    mass: float = Field(
        ..., description="Intensity-weighted centroid mass (Da) of the assigned points"
    )
    intensity: float = Field(
        ...,
        description=(
            "Peak-top intensity normalised to the profile base peak (100); "
            "for a flat-topped plateau this is the plateau level"
        ),
    )
    total_intensity_percent: float = Field(
        ...,
        description=(
            "Sum of the original intensities assigned to this peak as a "
            "percentage of all assigned above-threshold intensity"
        ),
    )


class ProfileCentroidResponse(BaseModel):
    point_count: int
    relative_threshold: float
    min_separation: float
    candidate_count: int = Field(
        ..., description="Number of flat-topped local-maximum candidates before NMS"
    )
    peak_count: int
    peaks: list[CentroidPeakOut]


class ObservedPeak(BaseModel):
    mass: float = Field(..., description="Observed peak mass (Da)")
    intensity: float = Field(..., ge=0, description="Observed peak intensity")


class EnrichmentFitRequest(BaseModel):
    formula: str = Field(
        ...,
        description="Molecular formula, e.g. 'C6H12O6' or 'C6H6[13C]2'",
        examples=["C6H12O6"],
    )
    label: str = Field(
        ...,
        description="Single isotope label to estimate, e.g. '13C'",
        examples=["13C"],
    )
    peaks: list[ObservedPeak] = Field(
        ...,
        max_length=200,
        description="Observed peaks, sorted by strictly increasing mass",
    )
    step_percent: float = Field(
        1.0,
        ge=0.5,
        le=10.0,
        description="Abundance grid step in percentage points (0.5-10)",
    )
    mass_tolerance: float = Field(
        ...,
        gt=0,
        description="Mass tolerance (Da) for pairing observed/predicted peaks",
    )


class PeakPairOut(BaseModel):
    observed_mass: float | None
    observed_intensity: float | None
    predicted_mass: float | None
    predicted_intensity: float | None


class EnrichmentFitResponse(BaseModel):
    formula: str
    label: str
    best_abundance_percent: float
    mse: float
    search_points: int
    pair_count: int
    pairs: list[PeakPairOut]


class DualLabelSpec(BaseModel):
    label: str = Field(
        ...,
        description="Isotope label to estimate, e.g. '13C'",
        examples=["13C"],
    )
    step_percent: float = Field(
        1.0,
        ge=0.5,
        le=10.0,
        description="Abundance grid step in percentage points (0.5-10)",
    )


class DualEnrichmentFitRequest(BaseModel):
    formula: str = Field(
        ...,
        description="Molecular formula, e.g. 'C6H12O6' or 'C6H6[13C]2'",
        examples=["C6H12O6"],
    )
    labels: list[DualLabelSpec] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Exactly two isotope labels, which must target different elements",
        examples=[[{"label": "13C", "step_percent": 1.0}, {"label": "15N", "step_percent": 1.0}]],
    )
    peaks: list[ObservedPeak] = Field(
        ...,
        max_length=200,
        description="Observed peaks, sorted by strictly increasing mass",
    )
    mass_tolerance: float = Field(
        ...,
        gt=0,
        description="Mass tolerance (Da) for pairing observed/predicted peaks",
    )


class BestAbundanceOut(BaseModel):
    label: str
    best_abundance_percent: float


class DualEnrichmentFitResponse(BaseModel):
    formula: str
    best_abundances: list[BestAbundanceOut]
    mse: float
    search_points: int
    pair_count: int
    pairs: list[PeakPairOut]


class ObservedMzPeak(BaseModel):
    mz: float = Field(..., description="Observed mass-to-charge ratio")
    intensity: float = Field(..., ge=0, description="Observed peak intensity")


class IonCandidateSpec(BaseModel):
    adduct: str = Field(
        ...,
        description=(
            "Adduct made of consecutive '+/-[coefficient]formula' fragments, "
            "e.g. '+H', '+2H-H2O' or '+Na-H'"
        ),
        examples=["+H"],
    )
    charge: int = Field(
        ...,
        ge=-10,
        le=10,
        description="Signed ion charge; must be -10..-1 or 1..10 (zero is not allowed)",
        examples=[1],
    )

    @field_validator("charge")
    @classmethod
    def charge_must_be_nonzero(cls, value: int) -> int:
        if value == 0:
            raise ValueError("charge must be non-zero: use -10..-1 or 1..10")
        return value


class IonCandidatePairOut(BaseModel):
    observed_mz: float | None
    observed_intensity: float | None
    predicted_mz: float | None
    predicted_intensity: float | None


class IonCandidateOut(BaseModel):
    index: int = Field(..., description="0-based position of the candidate in the request")
    valid: bool
    adduct: str
    charge: int
    mse: float | None = Field(None, description="Normalised intensity MSE (null for invalid candidates)")
    monoisotopic_mz: float | None = None
    composition: dict[str, int] | None = None
    labels: dict[str, int] | None = None
    pair_count: int | None = None
    pairs: list[IonCandidatePairOut] | None = None
    error: str | None = Field(
        None, description="Machine-readable error code for invalid candidates"
    )
    message: str | None = Field(
        None, description="Human-readable reason why the candidate is invalid"
    )
    position: int | None = Field(
        None, description="0-based character position in the adduct string"
    )


class IonIdentifyBestOut(BaseModel):
    index: int = Field(..., description="0-based index of the best candidate in the request")
    adduct: str
    charge: int
    mse: float


class IonIdentifyResponse(BaseModel):
    formula: str
    mass_tolerance: float
    observed_peak_count: int
    candidates: list[IonCandidateOut]
    best_candidate: IonIdentifyBestOut | None = None


class IonIdentifyRequest(BaseModel):
    formula: str = Field(
        ...,
        description="Neutral molecular formula, e.g. 'C6H12O6' or 'C6H6[13C]2'",
        examples=["C6H12O6"],
    )
    peaks: list[ObservedMzPeak] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Observed m/z peaks, 1-200 entries, sorted by strictly increasing m/z",
    )
    mass_tolerance: float = Field(
        ...,
        gt=0,
        description="m/z tolerance for pairing observed/predicted peaks",
    )
    candidates: list[IonCandidateSpec] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="1-20 candidate adduct/charge combinations to score",
    )


class IonMixtureFitRequest(BaseModel):
    formula: str = Field(
        ...,
        description="Neutral molecular formula, e.g. 'C6H12O6' or 'C6H6[13C]2'",
        examples=["C6H12O6"],
    )
    peaks: list[ObservedMzPeak] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Observed m/z peaks, 1-200 entries, sorted by strictly increasing m/z",
    )
    mass_tolerance: float = Field(
        ...,
        gt=0,
        description=(
            "m/z tolerance for fusing overlaid mixture peaks and for "
            "pairing observed/predicted peaks"
        ),
    )
    candidates: list[IonCandidateSpec] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="2-10 candidate adduct/charge combinations to mix pairwise",
    )
    step_percent: float = Field(
        1.0,
        ge=0.5,
        le=10.0,
        description="Mixture-ratio grid step in percentage points (0.5-10)",
    )


class IonMixtureCandidateOut(BaseModel):
    index: int = Field(..., description="0-based position of the candidate in the request")
    valid: bool
    adduct: str
    charge: int
    monoisotopic_mz: float | None = None
    error: str | None = Field(
        None, description="Machine-readable error code for invalid candidates"
    )
    message: str | None = Field(
        None, description="Human-readable reason why the candidate is invalid"
    )
    position: int | None = Field(
        None, description="0-based character position in the adduct string"
    )


class IonMixtureCombinationOut(BaseModel):
    rank: int = Field(..., description="1-based rank by ascending MSE")
    first_index: int = Field(..., description="0-based index of the lower-index candidate")
    second_index: int = Field(..., description="0-based index of the higher-index candidate")
    first_adduct: str
    first_charge: int
    second_adduct: str
    second_charge: int
    first_fraction_percent: float = Field(
        ..., description="Mixture ratio of the first candidate, 0-100%"
    )
    second_fraction_percent: float = Field(
        ..., description="Mixture ratio of the second candidate, 0-100%"
    )
    mse: float


class IonMixtureBestOut(BaseModel):
    first_index: int
    second_index: int
    first_adduct: str
    first_charge: int
    second_adduct: str
    second_charge: int
    first_fraction_percent: float
    second_fraction_percent: float
    mse: float
    pair_count: int
    pairs: list[IonCandidatePairOut]


class IonMixtureFitResponse(BaseModel):
    formula: str
    mass_tolerance: float
    step_percent: float
    observed_peak_count: int
    valid_candidate_count: int
    combination_count: int
    candidates: list[IonMixtureCandidateOut]
    combinations: list[IonMixtureCombinationOut]
    best_combination: IonMixtureBestOut | None = None


class FormulaSearchElement(BaseModel):
    symbol: str = Field(
        ...,
        description="Element symbol supported by /api/v1/elements, e.g. 'C'",
        examples=["C"],
    )
    min_count: int = Field(
        ...,
        ge=0,
        le=200,
        description="Minimum (inclusive) atom count; 0 <= min_count <= max_count",
        examples=[0],
    )
    max_count: int = Field(
        ...,
        ge=0,
        le=200,
        description="Maximum (inclusive) atom count; min_count <= max_count <= 200",
        examples=[20],
    )


class FormulaSearchRequest(BaseModel):
    target_mass: float = Field(
        ...,
        gt=0,
        description="Target monoisotopic mass (Da) of the neutral formula",
        examples=[180.063388],
    )
    observed_peaks: list[ObservedPeak] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Observed peaks, sorted by strictly increasing mass",
    )
    elements: list[FormulaSearchElement] = Field(
        ...,
        min_length=2,
        max_length=6,
        description="2-6 distinct elements with their inclusive atom-count ranges",
        examples=[
            [
                {"symbol": "C", "min_count": 0, "max_count": 20},
                {"symbol": "H", "min_count": 0, "max_count": 40},
                {"symbol": "O", "min_count": 0, "max_count": 10},
            ]
        ],
    )
    mass_tolerance: float = Field(
        ...,
        gt=0,
        description="Mass tolerance (Da) around target_mass for a formula to hit",
        examples=[0.005],
    )
    max_results: int = Field(
        ...,
        ge=1,
        le=50,
        description="Maximum number of ranked candidates to return (1-50)",
        examples=[10],
    )

    @field_validator("target_mass", "mass_tolerance")
    @classmethod
    def values_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be a finite number")
        return value

    @field_validator("observed_peaks")
    @classmethod
    def peaks_must_be_finite(cls, value: list[ObservedPeak]) -> list[ObservedPeak]:
        for peak in value:
            if not math.isfinite(peak.mass) or not math.isfinite(peak.intensity):
                raise ValueError("mass and intensity must be finite numbers")
        return value

    @field_validator("elements")
    @classmethod
    def elements_must_be_unique_and_ordered(
        cls, value: list[FormulaSearchElement]
    ) -> list[FormulaSearchElement]:
        seen: set[str] = set()
        for element in value:
            if element.symbol not in ISOTOPES:
                raise ValueError(f"Unknown element symbol '{element.symbol}'")
            if element.symbol in seen:
                raise ValueError(
                    f"Element '{element.symbol}' is repeated; elements must be unique"
                )
            seen.add(element.symbol)
            if element.min_count > element.max_count:
                raise ValueError(
                    f"min_count ({element.min_count}) must not exceed max_count "
                    f"({element.max_count}) for element '{element.symbol}'"
                )
        return value


class FormulaSearchCandidateOut(BaseModel):
    rank: int = Field(..., description="1-based rank by ascending MSE")
    formula: str = Field(
        ..., description="Formula with element symbols in lexicographic order"
    )
    composition: dict[str, int]
    monoisotopic_mass: float
    mass_error: float = Field(
        ..., description="Predicted monoisotopic mass minus target_mass (Da)"
    )
    mse: float
    pair_count: int
    pairs: list[PeakPairOut]


class FormulaSearchResponse(BaseModel):
    target_mass: float
    mass_tolerance: float
    max_results: int
    observed_peak_count: int
    visited_nodes: int = Field(
        ..., description="Number of branch-and-bound tree nodes visited"
    )
    pruned_branches: int = Field(
        ..., description="Number of count-value branches cut by the mass bounds"
    )
    total_hits: int = Field(
        ..., description="Number of within-tolerance compositions found"
    )
    candidates: list[FormulaSearchCandidateOut]


class MassCalibrationPoint(BaseModel):
    observed_mz: float = Field(
        ...,
        gt=0,
        description="Observed monoisotopic m/z of this reference ion",
        examples=[181.070665],
    )
    formula: str = Field(
        ...,
        description="Neutral molecular formula of the reference ion, e.g. 'C6H12O6'",
        examples=["C6H12O6"],
    )
    adduct: str = Field(
        ...,
        description=(
            "Adduct made of consecutive '+/-[coefficient]formula' fragments; "
            "the same grammar as /api/v1/ion-envelope, e.g. '+H' or '+Na-H'"
        ),
        examples=["+H"],
    )
    charge: int = Field(
        ...,
        ge=-10,
        le=10,
        description="Signed ion charge; must be -10..-1 or 1..10 (zero is not allowed)",
        examples=[1],
    )

    @field_validator("charge")
    @classmethod
    def charge_must_be_nonzero(cls, value: int) -> int:
        if value == 0:
            raise ValueError("charge must be non-zero: use -10..-1 or 1..10")
        return value

    @field_validator("observed_mz")
    @classmethod
    def value_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("observed_mz must be a finite number")
        return value


class MassCalibrateRequest(BaseModel):
    calibration_points: list[MassCalibrationPoint] = Field(
        ...,
        min_length=CALIBRATION_MIN_POINTS,
        max_length=CALIBRATION_MAX_POINTS,
        description=(
            f"{CALIBRATION_MIN_POINTS}-{CALIBRATION_MAX_POINTS} reference "
            "calibration points, each pairing an observed monoisotopic m/z "
            "with the reference ion's neutral formula, adduct and charge"
        ),
    )
    tolerance_ppm: float = Field(
        ...,
        gt=0,
        description=(
            "Absolute residual tolerance in ppm for the robust initial model "
            "when selecting inliers"
        ),
        examples=[5.0],
    )
    mz: list[float] = Field(
        ...,
        min_length=1,
        max_length=200,
        validation_alias=AliasChoices("mz", "mzs", "mz_values"),
        serialization_alias="mz",
        description=(
            "Observed m/z values to calibrate with the fitted inverse "
            "transform, 1-200 finite positive values (also accepted as "
            "'mzs' or 'mz_values')"
        ),
        examples=[[181.071, 91.039]],
    )

    @field_validator("tolerance_ppm")
    @classmethod
    def tolerance_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("tolerance_ppm must be a finite number")
        return value

    @field_validator("mz")
    @classmethod
    def mz_values_must_be_finite_and_positive(cls, value: list[float]) -> list[float]:
        for item in value:
            if not math.isfinite(item):
                raise ValueError("each mz value must be a finite number")
            if item <= 0:
                raise ValueError("each mz value must be positive")
        return value


class MassCalibrationPointOut(BaseModel):
    index: int = Field(
        ..., description="0-based position of the calibration point in the request"
    )
    valid: bool
    observed_mz: float
    formula: str
    adduct: str
    charge: int
    theoretical_mz: float | None = Field(
        None,
        description="Theoretical monoisotopic m/z computed from formula/adduct/charge "
        "(null for points that fail to parse)",
    )
    residual_mz: float | None = Field(
        None, description="Final-model residual: observed - (slope * theoretical + intercept) Da"
    )
    residual_ppm: float | None = Field(
        None, description="Final-model residual relative to the theoretical m/z, in ppm"
    )
    inlier: bool | None = Field(
        None,
        description=(
            "Whether the point is an inlier of the robust initial model "
            "(null for points that fail to parse)"
        ),
    )
    error: str | None = Field(
        None, description="Machine-readable error code for invalid points"
    )
    message: str | None = Field(
        None, description="Human-readable reason why the point is invalid"
    )


class CalibratedMzOut(BaseModel):
    mz: float = Field(..., description="Observed m/z submitted for calibration")
    calibrated_mz: float = Field(
        ..., description="m/z after applying the fitted inverse transform"
    )


class MassCalibrateResponse(BaseModel):
    tolerance_ppm: float
    slope: float = Field(
        ..., description="Final least-squares slope of observed = slope * theoretical + intercept"
    )
    intercept: float = Field(..., description="Final least-squares intercept (Da)")
    calibration_point_count: int
    valid_point_count: int
    inlier_count: int
    calibration_points: list[MassCalibrationPointOut] = Field(
        ..., description="Per-point residuals and inlier flags, kept in request order"
    )
    calibrated_mz: list[CalibratedMzOut] = Field(
        ..., description="Calibrated m/z values, kept in request order"
    )


class IsotopeClusterPeakIn(BaseModel):
    mz: float = Field(
        ..., description="Observed centroid m/z; the list must be strictly increasing"
    )
    intensity: float = Field(
        ..., ge=0, description="Observed peak intensity (non-negative; at least one positive)"
    )


class IsotopeClustersRequest(BaseModel):
    peaks: list[IsotopeClusterPeakIn] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Centroid m/z peaks, 1-200 entries, sorted by strictly increasing m/z",
    )
    min_charge: int = Field(
        1,
        ge=1,
        le=10,
        description="Lowest charge state to test (1-10); must not exceed max_charge",
    )
    max_charge: int = Field(
        10,
        ge=1,
        le=10,
        description="Highest charge state to test (1-10); must not be below min_charge",
    )
    spacing_tolerance: float = Field(
        ...,
        gt=0,
        description=(
            "Maximum allowed deviation (m/z units) of the measured gap "
            "between adjacent cluster members from the integer nominal "
            "spacing 1.003354835/charge"
        ),
        examples=[0.005],
    )
    min_peaks: int = Field(
        ...,
        ge=2,
        le=200,
        description="Minimum number of observed members a candidate cluster must have",
        examples=[3],
    )

    @field_validator("spacing_tolerance")
    @classmethod
    def spacing_tolerance_must_be_finite(cls, value: float) -> float:
        # gt=0 accepts +Infinity, which would later fail JSON response
        # rendering, so reject non-finite values explicitly with 422.
        if not math.isfinite(value):
            raise ValueError("spacing_tolerance must be a finite number")
        return value

    @field_validator("peaks")
    @classmethod
    def peaks_must_be_finite(
        cls, value: list[IsotopeClusterPeakIn]
    ) -> list[IsotopeClusterPeakIn]:
        for peak in value:
            if not math.isfinite(peak.mz) or not math.isfinite(peak.intensity):
                raise ValueError("mz and intensity must be finite numbers")
        return value

    @field_validator("max_charge")
    @classmethod
    def charge_range_must_be_ordered(cls, value: int, info) -> int:
        min_charge = info.data.get("min_charge")
        if min_charge is not None and value < min_charge:
            raise ValueError(
                f"max_charge ({value}) must not be below min_charge ({min_charge})"
            )
        return value


class IsotopeClusterOut(BaseModel):
    charge: int = Field(..., description="Charge state of the cluster")
    monoisotopic_mz: float = Field(
        ..., description="m/z of the first (lowest-m/z) cluster member"
    )
    score: float = Field(
        ...,
        description="Sum of base-peak-normalised member intensities minus 10 per missing isotope position",
    )
    member_indices: list[int] = Field(
        ...,
        description="0-based indices of member peaks in the request peak list, m/z ascending",
    )
    missing_positions: list[int] = Field(
        ...,
        description=(
            "0-based nominal isotope positions (offset from the "
            "monoisotopic member) absent from the cluster because of a "
            "two-step gap; empty or a single position"
        ),
    )


class IsotopeClustersResponse(BaseModel):
    min_charge: int
    max_charge: int
    spacing_tolerance: float
    min_peaks: int
    peak_count: int
    cluster_count: int
    clusters: list[IsotopeClusterOut] = Field(
        ..., description="Selected clusters, ordered by ascending first member index"
    )
    unassigned_peaks: list[int] = Field(
        ..., description="0-based indices of peaks belonging to no selected cluster"
    )


class ConsensusPeakIn(BaseModel):
    mz: float = Field(
        ..., description="Observed centroid m/z; strictly increasing within its spectrum"
    )
    intensity: float = Field(
        ...,
        ge=0,
        description="Peak intensity (non-negative; at least one positive per spectrum)",
    )


class ConsensusSpectrumIn(BaseModel):
    peaks: list[ConsensusPeakIn] = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "Centroid peaks of one replicate spectrum, 1-200 entries "
            "sorted by strictly increasing m/z"
        ),
    )

    @field_validator("peaks")
    @classmethod
    def peaks_must_be_finite(
        cls, value: list[ConsensusPeakIn]
    ) -> list[ConsensusPeakIn]:
        for peak in value:
            if not math.isfinite(peak.mz) or not math.isfinite(peak.intensity):
                raise ValueError("mz and intensity must be finite numbers")
        return value


class ConsensusSpectrumRequest(BaseModel):
    spectra: list[ConsensusSpectrumIn] = Field(
        ...,
        min_length=2,
        max_length=20,
        description="2-20 replicate centroid spectra of the same sample",
    )
    tolerance_da: float = Field(
        ...,
        gt=0,
        description=(
            "Absolute m/z tolerance (Da); a merged cluster's endpoint span "
            "must not exceed max(tolerance_da, endpoint mean x "
            "tolerance_ppm / 1e6)"
        ),
        examples=[0.005],
    )
    tolerance_ppm: float = Field(
        ...,
        ge=0,
        description=(
            "Relative m/z tolerance (ppm); combined with tolerance_da via "
            "max() on the merged cluster's endpoint mean"
        ),
        examples=[5.0],
    )
    min_replicates: int = Field(
        ...,
        ge=2,
        le=20,
        description="Minimum number of spectra a cluster must cover to be reported",
        examples=[2],
    )

    @field_validator("tolerance_da", "tolerance_ppm")
    @classmethod
    def tolerances_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("tolerance must be a finite number")
        return value


class ConsensusMemberOut(BaseModel):
    spectrum_index: int = Field(..., description="0-based index of the input spectrum")
    peak_index: int = Field(
        ..., description="0-based index of the peak within that spectrum"
    )


class ConsensusClusterOut(BaseModel):
    mz: float = Field(
        ...,
        description=(
            "Consensus m/z: member m/z average weighted by the original "
            "(request) intensities"
        ),
    )
    intensity: float = Field(
        ...,
        description=(
            "Median of the base-peak-normalised member intensities; "
            "spectra absent from the cluster count as 0"
        ),
    )
    spectrum_count: int = Field(
        ..., description="Number of distinct spectra covered by the cluster"
    )
    members: list[ConsensusMemberOut]


class ConsensusSpectrumResponse(BaseModel):
    spectrum_count: int
    tolerance_da: float
    tolerance_ppm: float
    min_replicates: int
    peak_count: int = Field(
        ..., description="Total number of input peaks across all spectra"
    )
    cluster_count: int
    clusters: list[ConsensusClusterOut] = Field(
        ..., description="Consensus peaks, ordered by ascending m/z"
    )


@app.exception_handler(CalibrationError)
async def calibration_error_handler(
    request: Request, exc: CalibrationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": exc.error,
                "message": exc.message,
            }
        },
    )


@app.exception_handler(FormulaError)
async def formula_error_handler(request: Request, exc: FormulaError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": "invalid_formula",
                "message": exc.message,
                "position": exc.position,
            }
        },
    )


@app.exception_handler(AdductError)
async def adduct_error_handler(request: Request, exc: AdductError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": "invalid_adduct",
                "message": exc.message,
                "position": exc.position,
            }
        },
    )


@app.exception_handler(LabelError)
async def label_error_handler(request: Request, exc: LabelError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": "invalid_label",
                "message": exc.message,
                "position": exc.position,
            }
        },
    )


@app.exception_handler(PeakListError)
async def peak_list_error_handler(request: Request, exc: PeakListError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": "invalid_peaks",
                "message": exc.message,
                "index": exc.index,
            }
        },
    )


@app.exception_handler(LabelCombinationError)
async def label_combination_error_handler(
    request: Request, exc: LabelCombinationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": "invalid_labels",
                "message": exc.message,
                "index": exc.index,
            }
        },
    )


@app.exception_handler(ProfileParameterError)
async def profile_parameter_error_handler(
    request: Request, exc: ProfileParameterError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": exc.error,
                "message": exc.message,
                "field": exc.field,
            }
        },
    )


@app.exception_handler(CentroidError)
async def centroid_error_handler(request: Request, exc: CentroidError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": exc.error,
                "message": exc.message,
                "field": exc.field,
                "index": exc.index,
            }
        },
    )


@app.exception_handler(IsotopeClusterError)
async def isotope_cluster_error_handler(
    request: Request, exc: IsotopeClusterError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": exc.error,
                "message": exc.message,
                "field": exc.field,
            }
        },
    )


@app.exception_handler(ConsensusError)
async def consensus_error_handler(
    request: Request, exc: ConsensusError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": exc.error,
                "message": exc.message,
                "spectrum_index": exc.spectrum_index,
                "peak_index": exc.peak_index,
            }
        },
    )


@app.exception_handler(SearchLimitExceeded)
async def formula_search_limit_handler(
    request: Request, exc: SearchLimitExceeded
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": "search_limit_exceeded",
                "message": (
                    f"Formula search visited {exc.visited_nodes} nodes, more "
                    f"than the limit of {MAX_VISITED_NODES}; narrow the element "
                    f"count ranges or the mass tolerance"
                ),
                "visited_nodes": exc.visited_nodes,
                "pruned_branches": exc.pruned_branches,
                "limit": MAX_VISITED_NODES,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Mirror FastAPI's default 422 body, but sanitise non-finite floats in
    # the echoed input so a literal NaN/Infinity request value stays JSON
    # encodable instead of failing response rendering.
    detail = jsonable_encoder(exc.errors())

    def _sanitise(value):
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, dict):
            return {key: _sanitise(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_sanitise(item) for item in value]
        return value

    return JSONResponse(status_code=422, content={"detail": _sanitise(detail)})


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/v1/elements")
def elements() -> dict:
    """List supported element symbols and their natural isotopes."""
    return {
        symbol: [{"mass": m, "abundance": a} for m, a in isotopes]
        for symbol, isotopes in sorted(ISOTOPES.items())
    }


@app.post("/api/v1/envelope", response_model=EnvelopeResponse)
def envelope(req: EnvelopeRequest) -> dict:
    """Compute the isotope envelope and monoisotopic mass of a formula."""
    try:
        parsed = parse_formula(req.formula)
    except FormulaError as exc:
        db.log_request(req.formula, req.top_n, ok=False, message=exc.message)
        raise

    cached = db.get_cached(req.formula, req.top_n)
    if cached is not None:
        db.log_request(req.formula, req.top_n, ok=True, message="cache hit")
        return {**cached, "cached": True}

    env = isotope_envelope(parsed, settings.merge_tolerance, settings.prune_threshold)
    peaks = top_peaks(env, req.top_n)
    result = {
        "formula": req.formula,
        "composition": parsed.composition,
        "labels": {f"[{mass_no}{symbol}]": count for (symbol, mass_no), count in parsed.labels.items()},
        "monoisotopic_mass": round(monoisotopic_mass(parsed), 6),
        "base_peak_intensity": 100.0,
        "peak_count": len(peaks),
        "peaks": [
            {"mass": round(m, 6), "intensity": round(i, 6)} for m, i in peaks
        ],
        "cached": False,
    }
    db.put_cache(req.formula, req.top_n, {k: v for k, v in result.items() if k != "cached"})
    db.log_request(req.formula, req.top_n, ok=True, message="computed")
    return result


@app.post("/api/v1/ion-envelope", response_model=IonEnvelopeResponse)
def ion_envelope_endpoint(req: IonEnvelopeRequest) -> dict:
    """Compute the adducted ion isotope envelope as m/z peaks.

    The neutral formula is parsed, the adduct fragments are applied to the
    atom counts, and the envelope is convolved with the neutral peak merge
    tolerance scaled by |charge|; ionic masses are then converted to m/z via
    ``(mass - charge * m_e) / |charge|``.
    """
    try:
        parsed = parse_formula(req.formula)
        pieces = parse_adduct(req.adduct)
        adjusted = apply_adduct(parsed, pieces)
    except (FormulaError, AdductError) as exc:
        db.log_request(req.formula, req.top_n, ok=False, message=exc.message)
        raise

    # Larger charge states spread isotope peaks more tightly in m/z, so the
    # neutral-mass merge tolerance scales with |charge| before conversion.
    merge_tolerance = abs(req.charge) * settings.merge_tolerance
    env = isotope_envelope(adjusted, merge_tolerance, settings.prune_threshold)
    peaks = top_peaks(env, req.top_n)
    mz_peaks = ion_envelope(peaks, req.charge)
    mono_mz = (
        monoisotopic_mass(adjusted) - req.charge * ELECTRON_MASS_DA
    ) / abs(req.charge)

    result = {
        "formula": req.formula,
        "adduct": req.adduct,
        "charge": req.charge,
        "composition": adjusted.composition,
        "labels": {
            f"[{mass_no}{symbol}]": count
            for (symbol, mass_no), count in adjusted.labels.items()
        },
        "monoisotopic_mz": round(mono_mz, 6),
        "base_peak_intensity": 100.0,
        "peak_count": len(mz_peaks),
        "peaks": [
            {"mz": round(mz, 6), "intensity": round(i, 6)} for mz, i in mz_peaks
        ],
    }
    db.log_request(req.formula, req.top_n, ok=True, message="ion envelope computed")
    return result


@app.post("/api/v1/neutral-profile", response_model=NeutralProfileResponse)
def neutral_profile(req: NeutralProfileRequest) -> dict:
    """Simulate the instrument profile of a neutral-molecule stick envelope.

    The formula is parsed and convolved with the existing natural-abundance
    envelope machinery (no ``top_n`` truncation), then every stick at mass m
    is broadened into a Gaussian with FWHM = m/resolving_power and sampled on
    one uniform grid (spacing ``mass_step``) within +-4 sigma; kernels are
    clipped to the requested window but keep their in-window contribution.
    The summed profile is normalised to a peak intensity of 100.
    """
    try:
        parsed = parse_formula(req.formula)
    except FormulaError as exc:
        db.log_request(req.formula, None, ok=False, message=exc.message)
        raise

    env = isotope_envelope(parsed, settings.merge_tolerance, settings.prune_threshold)
    result = simulate_profile(
        env,
        req.resolving_power,
        req.mass_step,
        req.mass_min,
        req.mass_max,
    )
    db.log_request(req.formula, None, ok=True, message="neutral profile computed")
    return {
        "formula": req.formula,
        "resolving_power": req.resolving_power,
        "mass_min": result.mass_min,
        "mass_max": result.mass_max,
        "mass_step": result.mass_step,
        "point_count": result.point_count,
        "profile": [
            {"mass": mass, "intensity": round(intensity, 6)}
            for mass, intensity in result.profile
        ],
    }


@app.post("/api/v1/profile-centroid", response_model=ProfileCentroidResponse)
def profile_centroid(req: ProfileCentroidRequest) -> dict:
    """Convert an equally-spaced continuous profile into centroid peaks.

    The submitted grid is base-peak normalised and points below
    ``relative_threshold`` are discarded; flat-topped local maxima (one per
    plateau, positioned at the plateau midpoint) are picked as candidates,
    suppressed with a descending-intensity (lower mass on ties)
    ``min_separation`` non-maximum suppression, and each above-threshold
    point is assigned to its nearest surviving candidate (equidistant
    points go to the lower-mass one). Peak masses are intensity-weighted
    centroids over the *original* intensities; peaks are returned by
    ascending mass.
    """
    points = [(p.mass, p.intensity) for p in req.profile]
    result = centroid_profile(
        points, req.relative_threshold, req.min_separation
    )
    return {
        "point_count": result.point_count,
        "relative_threshold": result.relative_threshold,
        "min_separation": result.min_separation,
        "candidate_count": result.candidate_count,
        "peak_count": len(result.peaks),
        "peaks": [
            {
                "mass": round(peak.mass, 6),
                "intensity": round(peak.intensity, 6),
                "total_intensity_percent": round(peak.total_intensity_percent, 6),
            }
            for peak in result.peaks
        ],
    }


def _build_pairs(
    observed: list[tuple[float, float]],
    fit_pairs: list[tuple[int | None, int | None]],
    predicted: list[tuple[float, float]],
) -> list[dict]:
    """Serialise the best-fit pairing, sorted by mass (observed first)."""
    obs = normalise_base_peak(observed)
    pairs = [
        {
            "observed_mass": round(obs[i][0], 6) if i is not None else None,
            "observed_intensity": round(obs[i][1], 6) if i is not None else None,
            "predicted_mass": round(predicted[j][0], 6) if j is not None else None,
            "predicted_intensity": round(predicted[j][1], 6) if j is not None else None,
        }
        for i, j in fit_pairs
    ]
    pairs.sort(
        key=lambda p: p["observed_mass"]
        if p["observed_mass"] is not None
        else p["predicted_mass"]
    )
    return pairs


def _build_mz_pairs(
    observed: list[tuple[float, float]],
    fit_pairs: list[tuple[int | None, int | None]],
    predicted: list[tuple[float, float]],
) -> list[dict]:
    """Serialise an ion-identification pairing on the m/z axis, sorted by
    m/z (observed m/z first for unmatched-predicted placeholders)."""
    pairs = [
        {
            "observed_mz": round(observed[i][0], 6) if i is not None else None,
            "observed_intensity": round(observed[i][1], 6) if i is not None else None,
            "predicted_mz": round(predicted[j][0], 6) if j is not None else None,
            "predicted_intensity": round(predicted[j][1], 6) if j is not None else None,
        }
        for i, j in fit_pairs
    ]
    pairs.sort(
        key=lambda p: p["observed_mz"]
        if p["observed_mz"] is not None
        else p["predicted_mz"]
    )
    return pairs


@app.post("/api/v1/enrichment-fit", response_model=EnrichmentFitResponse)
def enrichment_fit(req: EnrichmentFitRequest) -> dict:
    """Estimate the abundance of a single isotope label by fitting the
    computed envelope against observed peaks over a 0-100% grid."""
    parsed = parse_formula(req.formula)
    symbol, mass_number = parse_label(req.label)

    observed = [(p.mass, p.intensity) for p in req.peaks]
    validate_observed_peaks(observed)

    fit = fit_enrichment(
        parsed,
        symbol,
        mass_number,
        observed,
        step_percent=req.step_percent,
        mass_tolerance=req.mass_tolerance,
        merge_tolerance=settings.merge_tolerance,
        prune=settings.prune_threshold,
    )

    pairs = _build_pairs(observed, fit.pairs, fit.predicted)
    return {
        "formula": req.formula,
        "label": req.label,
        "best_abundance_percent": round(fit.best_abundance_percent, 6),
        "mse": fit.mse,
        "search_points": fit.search_points,
        "pair_count": len(pairs),
        "pairs": pairs,
    }


@app.post(
    "/api/v1/dual-enrichment-fit", response_model=DualEnrichmentFitResponse
)
def dual_enrichment_fit(req: DualEnrichmentFitRequest) -> dict:
    """Jointly estimate the abundances of two isotope labels (on different
    elements) by fitting the computed envelope against observed peaks over
    a 2-D 0-100% grid."""
    parsed = parse_formula(req.formula)
    parsed_labels = [parse_label(spec.label) for spec in req.labels]

    observed = [(p.mass, p.intensity) for p in req.peaks]
    validate_observed_peaks(observed)

    # Labels targeting the same element are rejected here with a locatable
    # index; parse_label above already covered malformed/unknown labels.
    if parsed_labels[0][0] == parsed_labels[1][0]:
        raise LabelCombinationError(
            f"The two labels must target different elements, but "
            f"'{req.labels[1].label}' targets element "
            f"'{parsed_labels[1][0]}' already named by "
            f"'{req.labels[0].label}'",
            1,
        )

    fit = fit_dual_enrichment(
        parsed,
        parsed_labels[0],
        parsed_labels[1],
        observed,
        step_percent=(req.labels[0].step_percent, req.labels[1].step_percent),
        mass_tolerance=req.mass_tolerance,
        merge_tolerance=settings.merge_tolerance,
        prune=settings.prune_threshold,
    )

    pairs = _build_pairs(observed, fit.pairs, fit.predicted)
    return {
        "formula": req.formula,
        "best_abundances": [
            {"label": req.labels[k].label, "best_abundance_percent": round(pct, 6)}
            for k, pct in enumerate(fit.best_abundances_percent)
        ],
        "mse": fit.mse,
        "search_points": fit.search_points,
        "pair_count": len(pairs),
        "pairs": pairs,
    }


@app.post("/api/v1/ion-identify", response_model=IonIdentifyResponse)
def ion_identify(req: IonIdentifyRequest) -> dict:
    """Score candidate adduct/charge combinations against observed m/z peaks.

    Each candidate reuses the adduct parsing and ionic envelope computation
    of ``/api/v1/ion-envelope``; predicted and observed (base-peak
    normalised) peaks are paired one-to-one, smallest within-tolerance mass
    difference first, and unmatched peaks count at zero intensity in the
    normalised MSE. Invalid candidates are reported in place with their
    request index and the same position-aware adduct error; valid candidates
    are returned sorted by MSE with input-order tie-break.
    """
    parsed = parse_formula(req.formula)

    observed = [(p.mz, p.intensity) for p in req.peaks]
    validate_observed_peaks(observed)
    obs = normalise_base_peak(observed)

    valid_results: list[dict] = []
    invalid_results: list[dict] = []
    for idx, spec in enumerate(req.candidates):
        try:
            prediction = predict_ion_candidate(
                parsed,
                spec.adduct,
                spec.charge,
                settings.merge_tolerance,
                settings.prune_threshold,
            )
        except AdductError as exc:
            # Bad candidates are part of the 200 response (with their index
            # and the existing position-aware error), not a request failure.
            invalid_results.append(
                {
                    "index": idx,
                    "valid": False,
                    "adduct": spec.adduct,
                    "charge": spec.charge,
                    "mse": None,
                    "error": "invalid_adduct",
                    "message": exc.message,
                    "position": exc.position,
                }
            )
            continue

        score = score_ion_candidate(idx, prediction, obs, req.mass_tolerance)
        pairs = _build_mz_pairs(obs, score.pairs, score.predicted)
        valid_results.append(
            {
                "index": idx,
                "valid": True,
                "adduct": spec.adduct,
                "charge": spec.charge,
                "mse": score.mse,
                "monoisotopic_mz": round(score.monoisotopic_mz, 6),
                "composition": score.composition,
                "labels": score.labels,
                "pair_count": len(pairs),
                "pairs": pairs,
                "error": None,
                "message": None,
                "position": None,
            }
        )

    # Stable ordering: ascending MSE; an MSE tie keeps the request order.
    # Invalid candidates follow the scored ones, also in request order.
    valid_results.sort(key=lambda c: (c["mse"], c["index"]))
    candidates = valid_results + invalid_results

    best = None
    if valid_results:
        winner = valid_results[0]
        best = {
            "index": winner["index"],
            "adduct": winner["adduct"],
            "charge": winner["charge"],
            "mse": winner["mse"],
        }

    db.log_request(
        req.formula,
        None,
        ok=bool(valid_results),
        message=(
            f"ion identify: {len(valid_results)} valid, "
            f"{len(invalid_results)} invalid candidates"
        ),
    )
    return {
        "formula": req.formula,
        "mass_tolerance": req.mass_tolerance,
        "observed_peak_count": len(observed),
        "candidates": candidates,
        "best_candidate": best,
    }


@app.post("/api/v1/ion-mixture-fit", response_model=IonMixtureFitResponse)
def ion_mixture_fit(req: IonMixtureFitRequest) -> dict:
    """Fit pairwise mixtures of candidate ion envelopes to observed m/z peaks.

    Each candidate reuses the adduct parsing and ionic envelope computation
    of ``/api/v1/ion-envelope`` (base-peak normalised). For every unordered
    pair of valid candidates — the lower request index acting as the first
    candidate — and every first-candidate ratio on a 0-100% grid
    (``step_percent``), the two envelopes are scaled by their ratio,
    overlaid (peaks within ``mass_tolerance`` fused) and re-normalised to
    base peak 100, then paired one-to-one with the observed peaks; the MSE
    over all pairs, unmatched peaks on both sides included, ranks the
    combinations. Ties resolve to the smaller candidate index pair, then
    the lower first-candidate ratio. Invalid candidates are reported in
    place with the same position-aware error as ``/api/v1/ion-identify``
    and their combinations are skipped; with fewer than two valid
    candidates the combination list is empty and the best combination null.
    """
    parsed = parse_formula(req.formula)

    observed = [(p.mz, p.intensity) for p in req.peaks]
    validate_observed_peaks(observed)
    obs = normalise_base_peak(observed)

    candidates: list[dict] = []
    valid: list[tuple[int, IonCandidatePrediction]] = []
    for idx, spec in enumerate(req.candidates):
        try:
            prediction = predict_ion_candidate(
                parsed,
                spec.adduct,
                spec.charge,
                settings.merge_tolerance,
                settings.prune_threshold,
            )
        except AdductError as exc:
            # Bad candidates are part of the 200 response (with their index
            # and the existing position-aware error); every combination
            # involving them is skipped.
            candidates.append(
                {
                    "index": idx,
                    "valid": False,
                    "adduct": spec.adduct,
                    "charge": spec.charge,
                    "monoisotopic_mz": None,
                    "error": "invalid_adduct",
                    "message": exc.message,
                    "position": exc.position,
                }
            )
            continue
        valid.append((idx, prediction))
        candidates.append(
            {
                "index": idx,
                "valid": True,
                "adduct": spec.adduct,
                "charge": spec.charge,
                "monoisotopic_mz": round(prediction.monoisotopic_mz, 6),
                "error": None,
                "message": None,
                "position": None,
            }
        )

    combinations: list[dict] = []
    best: dict | None = None
    if len(valid) >= 2:
        scored = fit_ion_mixture(valid, obs, req.step_percent, req.mass_tolerance)
        by_index = dict(valid)
        for rank, combo in enumerate(scored, start=1):
            first = by_index[combo.first_index]
            second = by_index[combo.second_index]
            combinations.append(
                {
                    "rank": rank,
                    "first_index": combo.first_index,
                    "second_index": combo.second_index,
                    "first_adduct": first.adduct,
                    "first_charge": first.charge,
                    "second_adduct": second.adduct,
                    "second_charge": second.charge,
                    "first_fraction_percent": round(combo.first_fraction_percent, 6),
                    "second_fraction_percent": round(
                        100.0 - combo.first_fraction_percent, 6
                    ),
                    "mse": combo.mse,
                }
            )

        winner = scored[0]
        first = by_index[winner.first_index]
        second = by_index[winner.second_index]
        # Recompute the winning mixture once to expose its pairing detail.
        fit_pairs, _, mixed = score_mixture(
            first.peaks,
            second.peaks,
            winner.first_fraction_percent / 100.0,
            obs,
            req.mass_tolerance,
        )
        pairs = _build_mz_pairs(obs, fit_pairs, mixed)
        best = {
            "first_index": winner.first_index,
            "second_index": winner.second_index,
            "first_adduct": first.adduct,
            "first_charge": first.charge,
            "second_adduct": second.adduct,
            "second_charge": second.charge,
            "first_fraction_percent": round(winner.first_fraction_percent, 6),
            "second_fraction_percent": round(
                100.0 - winner.first_fraction_percent, 6
            ),
            "mse": winner.mse,
            "pair_count": len(pairs),
            "pairs": pairs,
        }

    db.log_request(
        req.formula,
        None,
        ok=best is not None,
        message=(
            f"ion mixture fit: {len(valid)} valid candidates, "
            f"{len(combinations)} combinations"
        ),
    )
    return {
        "formula": req.formula,
        "mass_tolerance": req.mass_tolerance,
        "step_percent": req.step_percent,
        "observed_peak_count": len(observed),
        "valid_candidate_count": len(valid),
        "combination_count": len(combinations),
        "candidates": candidates,
        "combinations": combinations,
        "best_combination": best,
    }


@app.post("/api/v1/formula-search", response_model=FormulaSearchResponse)
def formula_search(req: FormulaSearchRequest) -> dict:
    """Reverse-search molecular formulas from a target monoisotopic mass.

    All compositions of the 2-6 requested elements within their
    ``min_count``..``max_count`` ranges are enumerated by branch and bound:
    elements are visited heaviest monoisotopic mass first, and at every node
    the minimum/maximum mass the remaining elements can still contribute
    restricts (or entirely closes) the current element's count window, so
    branches that cannot land within ``mass_tolerance`` of ``target_mass``
    are never recursed into. Each hit reuses the natural-abundance isotope
    convolution, base-peak normalisation and one-to-one peak matching
    (unmatched peaks counted at zero intensity) of the enrichment-fit
    endpoints; candidates rank by MSE, then absolute mass error, then
    lexicographic formula. A search with no hit is a normal 200 with an
    empty candidate list; a search visiting more than one million nodes
    aborts with 422.
    """
    observed = [(p.mass, p.intensity) for p in req.observed_peaks]
    validate_observed_peaks(observed)

    elements = [
        ElementSpec(symbol=p.symbol, min_count=p.min_count, max_count=p.max_count)
        for p in req.elements
    ]
    result = search_formulas(
        target_mass=req.target_mass,
        observed=observed,
        elements=elements,
        mass_tolerance=req.mass_tolerance,
        max_results=req.max_results,
        merge_tolerance=settings.merge_tolerance,
        prune=settings.prune_threshold,
    )

    # Normalise once more for serialisation; scoring already normalised the
    # same observed list internally.
    obs = normalise_base_peak(observed)
    candidates = []
    for rank, hit in enumerate(result.hits, start=1):
        pairs = _build_pairs(obs, hit.pairs, hit.predicted)
        candidates.append(
            {
                "rank": rank,
                "formula": hit.formula,
                "composition": hit.composition,
                "monoisotopic_mass": round(hit.monoisotopic_mass, 6),
                "mass_error": round(hit.mass_error, 6),
                "mse": hit.mse,
                "pair_count": len(pairs),
                "pairs": pairs,
            }
        )

    db.log_request(
        f"formula-search:{','.join(e.symbol for e in elements)}",
        None,
        ok=result.total_hits > 0,
        message=(
            f"formula search: {result.total_hits} hits, "
            f"{result.visited_nodes} nodes visited, "
            f"{result.pruned_branches} branches pruned"
        ),
    )
    return {
        "target_mass": req.target_mass,
        "mass_tolerance": req.mass_tolerance,
        "max_results": req.max_results,
        "observed_peak_count": len(observed),
        "visited_nodes": result.visited_nodes,
        "pruned_branches": result.pruned_branches,
        "total_hits": result.total_hits,
        "candidates": candidates,
    }


@app.post("/api/v1/mass-calibrate", response_model=MassCalibrateResponse)
def mass_calibrate(req: MassCalibrateRequest) -> dict:
    """Calibrate the mass axis from reference calibration points.

    Each of the 3-200 calibration points pairs an observed monoisotopic m/z
    with the reference ion's neutral formula, adduct and signed charge; the
    theoretical monoisotopic m/z reuses the existing formula parser, adduct
    application and the ``(mass - charge * m_e) / |charge|`` electron-mass
    rule of ``/api/v1/ion-envelope``. Points that fail formula/adduct
    parsing are reported in place (with the same position-aware errors) and
    excluded from the fit. An initial affine model (observed vs.
    theoretical) is built from the median slope and median intercept of all
    valid point pairs; points whose absolute residual against it is at most
    ``tolerance_ppm`` are inliers, and one ordinary least-squares line is
    fitted to them. The final slope, intercept, every point's residual and
    inlier flag are returned in request order, together with the requested
    m/z values mapped through the inverse transform. No valid points, no
    distinct-theoretical-m/z pair, fewer than three inliers or a
    non-positive fitted slope answer 422.
    """
    points = [
        CalibrationPoint(
            index=idx,
            observed_mz=p.observed_mz,
            formula=p.formula,
            adduct=p.adduct,
            charge=p.charge,
        )
        for idx, p in enumerate(req.calibration_points)
    ]
    try:
        fit = calibrate_mass_axis(points, req.tolerance_ppm)
    except CalibrationError as exc:
        db.log_request(
            f"mass-calibrate:{len(points)}-points", None, ok=False, message=exc.message
        )
        raise

    calibrated = [
        {"mz": mz, "calibrated_mz": round(inverse_predict(fit, mz), 6)}
        for mz in req.mz
    ]
    point_out = [
        {
            "index": p.index,
            "valid": p.valid,
            "observed_mz": round(p.observed_mz, 6),
            "formula": p.formula,
            "adduct": p.adduct,
            "charge": p.charge,
            "theoretical_mz": round(p.theoretical_mz, 6)
            if p.theoretical_mz is not None
            else None,
            "residual_mz": round(p.residual_mz, 9)
            if p.residual_mz is not None
            else None,
            "residual_ppm": round(p.residual_ppm, 6)
            if p.residual_ppm is not None
            else None,
            "inlier": p.inlier,
            "error": p.error,
            "message": p.message,
        }
        for p in fit.points
    ]

    db.log_request(
        f"mass-calibrate:{len(points)}-points",
        None,
        ok=True,
        message=(
            f"mass calibration: {fit.valid_point_count} valid, "
            f"{fit.inlier_count} inliers, slope={fit.slope:.9g}, "
            f"intercept={fit.intercept:.9g}"
        ),
    )
    return {
        "tolerance_ppm": req.tolerance_ppm,
        "slope": round(fit.slope, 9),
        "intercept": round(fit.intercept, 9),
        "calibration_point_count": len(points),
        "valid_point_count": fit.valid_point_count,
        "inlier_count": fit.inlier_count,
        "calibration_points": point_out,
        "calibrated_mz": calibrated,
    }


@app.post("/api/v1/isotope-clusters", response_model=IsotopeClustersResponse)
def isotope_clusters(req: IsotopeClustersRequest) -> dict:
    """Detect unknown isotope clusters de novo in centroid m/z peaks.

    No molecular formula is needed. For every charge z in
    ``min_charge..max_charge`` (1..10), candidate chains are grown on the
    fixed nominal spacing ``1.003354835/z``: adjacent members must be one
    or two integer steps apart within ``spacing_tolerance`` and a chain may
    contain at most one two-step (single missing isotope position) gap.
    Only chains with at least ``min_peaks`` observed members qualify.

    Intensities are normalised to the spectrum base peak (100); a
    candidate's score is the sum of its member intensities minus 10 per
    missing position. A weighted-interval dynamic program selects the
    mutually index-non-overlapping candidates with the highest total score;
    ties resolve to more members, then the lexicographically smaller
    monoisotopic-m/z sequence, then the smaller charge sequence. Peaks used
    by no selected cluster come back as ``unassigned_peaks``. With no
    qualifying candidate the cluster list is empty and every peak is
    unassigned.
    """
    peaks = [(p.mz, p.intensity) for p in req.peaks]
    # Strictly increasing m/z is a request-level error with the same
    # locatable invalid_peaks shape as the other peak-list endpoints.
    # Non-negative finite intensities are enforced by the request model;
    # unlike the fitting endpoints an all-zero spectrum is accepted.
    for idx in range(1, len(peaks)):
        prev_mz, cur_mz = peaks[idx - 1][0], peaks[idx][0]
        if cur_mz == prev_mz:
            raise PeakListError(
                f"Duplicate peak m/z {cur_mz}; m/z values must be strictly "
                f"increasing",
                idx,
            )
        if cur_mz < prev_mz:
            raise PeakListError(
                f"Peak m/z values must be sorted in increasing order; "
                f"m/z {cur_mz} at index {idx} follows {prev_mz}",
                idx,
            )

    result = detect_isotope_clusters(
        peaks,
        (req.min_charge, req.max_charge),
        req.spacing_tolerance,
        req.min_peaks,
    )

    db.log_request(
        f"isotope-clusters:{len(peaks)}-peaks",
        None,
        ok=bool(result.clusters),
        message=(
            f"isotope clusters: {len(result.clusters)} found over charges "
            f"{req.min_charge}-{req.max_charge}"
        ),
    )
    return {
        "min_charge": result.charge_min,
        "max_charge": result.charge_max,
        "spacing_tolerance": result.spacing_tolerance,
        "min_peaks": result.min_peaks,
        "peak_count": len(peaks),
        "cluster_count": len(result.clusters),
        "clusters": [
            {
                "charge": cluster.charge,
                "monoisotopic_mz": round(cluster.monoisotopic_mz, 6),
                "score": round(cluster.score, 6),
                "member_indices": list(cluster.member_indices),
                "missing_positions": list(cluster.missing_positions),
            }
            for cluster in result.clusters
        ],
        "unassigned_peaks": list(result.unassigned_indices),
    }


@app.post("/api/v1/consensus-spectrum", response_model=ConsensusSpectrumResponse)
def consensus_spectrum(req: ConsensusSpectrumRequest) -> dict:
    """Build a consensus spectrum from replicate centroid measurements.

    Each of the 2-20 spectra is first normalised to its own base peak
    (100); all peaks are then pooled onto one m/z axis and clustered by a
    deterministic complete-linkage agglomerative clustering: two clusters
    merge only when the merged cluster would not hold two peaks of the
    same input spectrum and its endpoint span does not exceed
    ``max(tolerance_da, endpoint_mean * tolerance_ppm / 1e6)``. Every
    round merges the best legal pair by (merged span, merged lowest m/z,
    member spectrum/peak indices) until no legal merge remains. Clusters
    covering at least ``min_replicates`` spectra are returned with the
    original-intensity-weighted m/z, the member (spectrum, peak) indices
    and the median of the base-peak-normalised intensities (spectra absent
    from the cluster count as 0), sorted by ascending m/z.
    """
    spectra = [
        [(peak.mz, peak.intensity) for peak in spectrum.peaks]
        for spectrum in req.spectra
    ]
    result = build_consensus_spectrum(
        spectra, req.tolerance_da, req.tolerance_ppm, req.min_replicates
    )

    db.log_request(
        f"consensus-spectrum:{result.spectrum_count}-spectra",
        None,
        ok=bool(result.clusters),
        message=(
            f"consensus spectrum: {len(result.clusters)} clusters from "
            f"{result.peak_count} peaks over {result.spectrum_count} spectra"
        ),
    )
    return {
        "spectrum_count": result.spectrum_count,
        "tolerance_da": result.tolerance_da,
        "tolerance_ppm": result.tolerance_ppm,
        "min_replicates": result.min_replicates,
        "peak_count": result.peak_count,
        "cluster_count": len(result.clusters),
        "clusters": [
            {
                "mz": round(cluster.mz, 6),
                "intensity": round(cluster.intensity, 6),
                "spectrum_count": len(cluster.members),
                "members": [
                    {
                        "spectrum_index": member.spectrum_index,
                        "peak_index": member.peak_index,
                    }
                    for member in cluster.members
                ],
            }
            for cluster in result.clusters
        ],
    }
