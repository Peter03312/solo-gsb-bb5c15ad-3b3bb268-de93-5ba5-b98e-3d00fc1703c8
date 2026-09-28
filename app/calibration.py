"""Mass-axis calibration from reference calibration points.

Each calibration point pairs an observed monoisotopic m/z with the neutral
formula, adduct string and signed charge of the reference ion; its
theoretical monoisotopic m/z is computed with the same formula parsing,
adduct application and electron-mass rule as ``/api/v1/ion-envelope``::

    m/z = (monoisotopic_mass(adjusted) - charge * m_e) / |charge|

The calibration model is an affine map from theoretical m/z (x) to observed
m/z (y): ``y = slope * x + intercept``. A robust initial model is built with
the Theil–Sen/pairwise construction: the initial slope is the median of the
slopes defined by every pair of valid points with distinct theoretical m/z,
and the initial intercept is the median of that pair's intercept (the same
line passes through both points, so the intercept computed from either one
is identical). Points whose initial-model residual is within
``tolerance_ppm`` (``|residual / theoretical| * 1e6 <= tolerance_ppm``) are
the inliers; a single ordinary least-squares line is then fitted to the
inliers and reported as the final model. Calibrating an observed m/z applies
the inverse transform ``(mz - intercept) / slope``.

The fit aborts with :class:`CalibrationError` when no point is valid, when
no pair with distinct theoretical m/z exists, when fewer than three points
are inliers, or when the least-squares slope is not positive (the inverse
transform needs a strictly positive slope).
"""

from __future__ import annotations

from dataclasses import dataclass

from .adduct import AdductError, apply_adduct, parse_adduct
from .envelope import ELECTRON_MASS_DA, monoisotopic_mass
from .parser import FormulaError, parse_formula

# A residual exactly on the tolerance boundary is an inlier; this guard only
# absorbs ordinary floating-point round-off at that boundary.
_PPM_EPS = 1e-9

MIN_POINTS = 3
MAX_POINTS = 200
MIN_INLIERS = 3


class CalibrationError(ValueError):
    """Calibration could not be built from the submitted points."""

    def __init__(self, error: str, message: str) -> None:
        super().__init__(message)
        self.error = error
        self.message = message


@dataclass(frozen=True)
class CalibrationPoint:
    """One reference calibration point in request order."""

    index: int
    observed_mz: float
    formula: str
    adduct: str
    charge: int


@dataclass
class PointResidual:
    """Per-point outcome: theoretical m/z, final-model residual and the
    inlier/outlier flag assigned by the robust initial model."""

    index: int
    valid: bool
    observed_mz: float
    formula: str
    adduct: str
    charge: int
    theoretical_mz: float | None = None
    residual_mz: float | None = None
    residual_ppm: float | None = None
    inlier: bool | None = None
    error: str | None = None
    message: str | None = None


@dataclass
class MassCalibrationFit:
    """Final affine calibration model plus per-point diagnostics."""

    slope: float
    intercept: float
    valid_point_count: int
    inlier_count: int
    points: list[PointResidual]


def theoretical_mz(formula: str, adduct: str, charge: int) -> float:
    """Theoretical monoisotopic m/z of a reference ion.

    Reuses the existing formula parser, adduct application and the
    ``(mass - charge * m_e) / |charge|`` electron-mass rule shared with the
    ion-envelope endpoints. Raises :class:`FormulaError` or
    :class:`AdductError` exactly as those parsers do.
    """
    parsed = parse_formula(formula)
    pieces = parse_adduct(adduct)
    adjusted = apply_adduct(parsed, pieces)
    return (monoisotopic_mass(adjusted) - charge * ELECTRON_MASS_DA) / abs(charge)


def _median(values: list[float]) -> float:
    """Median of a non-empty list (average of the two middle values on ties)."""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _point_result_invalid(point: CalibrationPoint, exc: Exception) -> PointResidual:
    if isinstance(exc, FormulaError):
        error = "invalid_formula"
    else:
        error = "invalid_adduct"
    return PointResidual(
        index=point.index,
        valid=False,
        observed_mz=point.observed_mz,
        formula=point.formula,
        adduct=point.adduct,
        charge=point.charge,
        error=error,
        message=str(exc.args[0]) if exc.args else str(exc),
    )


def calibrate_mass_axis(
    calibration_points: list[CalibrationPoint],
    tolerance_ppm: float,
) -> MassCalibrationFit:
    """Build the robust-then-least-squares mass-axis calibration model.

    Invalid points (formula/adduct that fail to parse or apply) are reported
    in place and excluded from the fit. The fit raises
    :class:`CalibrationError` for the failure cases described in the module
    docstring.
    """
    points: list[PointResidual] = []
    valid: list[tuple[float, float]] = []  # (theoretical x, observed y)

    for point in calibration_points:
        try:
            x = theoretical_mz(point.formula, point.adduct, point.charge)
        except (FormulaError, AdductError) as exc:
            points.append(_point_result_invalid(point, exc))
            continue
        points.append(
            PointResidual(
                index=point.index,
                valid=True,
                observed_mz=point.observed_mz,
                formula=point.formula,
                adduct=point.adduct,
                charge=point.charge,
                theoretical_mz=x,
            )
        )
        valid.append((x, point.observed_mz))

    if not valid:
        raise CalibrationError(
            "no_valid_points",
            "No valid calibration points: every point failed formula or "
            "adduct parsing; see the per-point error/message fields",
        )

    # Robust initial model: median of pairwise slopes (Theil–Sen) and the
    # median intercept of the line through each pair. Only pairs with
    # distinct theoretical m/z define a finite slope.
    slopes: list[float] = []
    intercepts: list[float] = []
    for i in range(len(valid) - 1):
        x_i, y_i = valid[i]
        for j in range(i + 1, len(valid)):
            x_j, y_j = valid[j]
            if x_i == x_j:
                continue
            pair_slope = (y_j - y_i) / (x_j - x_i)
            slopes.append(pair_slope)
            intercepts.append(y_i - pair_slope * x_i)

    if not slopes:
        raise CalibrationError(
            "no_valid_point_pairs",
            "Cannot build an initial calibration model: all valid points "
            "share the same theoretical m/z",
        )

    initial_slope = _median(slopes)
    initial_intercept = _median(intercepts)

    inlier_flags: list[bool] = []
    for x, y in valid:
        residual = y - (initial_slope * x + initial_intercept)
        residual_ppm = residual / x * 1e6
        inlier_flags.append(abs(residual_ppm) <= tolerance_ppm + _PPM_EPS)

    inlier_count = sum(inlier_flags)
    if inlier_count < MIN_INLIERS:
        raise CalibrationError(
            "too_few_inliers",
            f"Only {inlier_count} of {len(valid)} valid calibration points "
            f"are inliers within {tolerance_ppm:g} ppm; at least "
            f"{MIN_INLIERS} are required (widen tolerance_ppm or replace "
            f"outlying points)",
        )

    inlier_x = [x for (x, _), flag in zip(valid, inlier_flags) if flag]
    inlier_y = [y for (_, y), flag in zip(valid, inlier_flags) if flag]
    mean_x = sum(inlier_x) / inlier_count
    mean_y = sum(inlier_y) / inlier_count
    sxx = sum((x - mean_x) ** 2 for x in inlier_x)
    sxy = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(inlier_x, inlier_y)
    )
    if sxx <= 0.0:
        raise CalibrationError(
            "degenerate_inliers",
            "The inlier points share one theoretical m/z, so a "
            "least-squares slope is undefined",
        )
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if not slope > 0.0:
        raise CalibrationError(
            "non_positive_slope",
            f"The least-squares calibration slope is {slope:.6g}, which is "
            "not positive; the inverse transform is undefined (check the "
            "calibration points for swapped or inconsistent m/z values)",
        )

    # Final-model residuals for every valid point; the inlier/outlier flag
    # stays the one assigned by the robust initial model.
    valid_iter = iter(zip(valid, inlier_flags))
    for result in points:
        if not result.valid:
            continue
        (x, y), flag = next(valid_iter)
        residual = y - (slope * x + intercept)
        result.residual_mz = residual
        result.residual_ppm = residual / x * 1e6
        result.inlier = flag

    return MassCalibrationFit(
        slope=slope,
        intercept=intercept,
        valid_point_count=len(valid),
        inlier_count=inlier_count,
        points=points,
    )


def inverse_predict(fit: MassCalibrationFit, mz: float) -> float:
    """Apply the fitted inverse transform to one observed m/z."""
    return (mz - fit.intercept) / fit.slope


__all__ = [
    "MAX_POINTS",
    "MIN_INLIERS",
    "MIN_POINTS",
    "CalibrationError",
    "CalibrationPoint",
    "MassCalibrationFit",
    "PointResidual",
    "calibrate_mass_axis",
    "inverse_predict",
    "theoretical_mz",
]
