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

All arithmetic is overflow-stable for legal finite inputs across the full
double-precision range: observed m/z and theoretical m/z may sit within a
few factors of the float maximum. The pairwise intercepts, the robust
residuals, the least-squares mean/variance accumulators and the final
per-point residuals are evaluated in power-of-two-scaled form whenever the
physical-unit expression would overflow, so a representable model never
collapses into an infinite slope or NaN residuals; likewise the inverse
transform divides by the slope before subtracting the intercept, so the
cancellation cannot be pre-empted by an overflowing subtraction. A model or
calibrated value that genuinely exceeds the float range stays non-finite and
is rejected as a non-positive/non-representable slope (422).
"""

from __future__ import annotations

import math
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
    # Halve each middle value before adding: ordinary inputs differ from the
    # plain mean by at most one ULP, while two values near the float maximum
    # cannot spuriously sum to infinity.
    return ordered[mid - 1] / 2.0 + ordered[mid] / 2.0


def _power_of_two_bound(value: float) -> float:
    """Smallest representable power of two >= ``abs(value)``.

    Dividing finite numbers by such a power of two is exact, so scaling by
    this bound loses no precision; it only keeps products and sums away
    from the float overflow threshold. Values just below the float
    maximum are bounded by the largest representable power of two
    (2**1023) rather than the unrepresentable next power of two.
    """
    mantissa, exponent = math.frexp(value)
    if mantissa == 0.5:  # value itself is a power of two: keep it unchanged
        exponent -= 1
    return math.ldexp(1.0, min(exponent, 1023))


def _stable_mean(values: list[float]) -> float:
    """Overflow-resistant mean of a non-empty list of finite values.

    The direct ``sum(values)/n`` overflows when the inputs sit near the
    float maximum (three legal observed m/z of ~1e308 already sum to
    infinity), which turns every downstream least-squares accumulator
    non-finite. :func:`math.fsum` is used for ordinary magnitudes; should
    the true sum itself exceed the float maximum, the values are divided
    by the largest-magnitude power-of-two bound (an exact rescale) before
    summing and the mean rescaled back.
    """
    count = len(values)
    try:
        total = math.fsum(values)
        if math.isfinite(total):
            return total / count
    except OverflowError:
        pass
    bound = _power_of_two_bound(max(abs(v) for v in values))
    scaled_total = math.fsum(v / bound for v in values)
    # Multiply by bound only after dividing by the count: multiplying the
    # full scaled sum first can overflow even when the mean itself fits.
    return bound * (scaled_total / count)


def _stable_affine(slope: float, x: float, intercept: float) -> float:
    """Evaluate ``slope * x + intercept`` without spurious overflow.

    The direct product ``slope * x`` can overflow to infinity even when
    the true affine value is finite (and vice versa: a large product that
    cancels a large intercept). The rescue path rescales by one
    power-of-two bound S = next power of two >= max(|slope|, |x|,
    |intercept|), chosen from the finite operands so it never depends on
    an already-infinite product; the product is then evaluated as
    S*(slope/S)*(x/S), whose inner factors stay within [-1, 1]. The final
    multiplication overflows only when the mathematical result itself is
    not representable as a finite float (in which case +-inf is the
    correct answer rather than a numerical accident).
    """
    product = slope * x
    direct = product + intercept
    if math.isfinite(direct):
        return direct
    # Choose S >= max(|slope|, |x|, |intercept|); then |slope*x|/S^2 <= 1
    # and |intercept|/S <= 1, so the inner expression S*((slope/S)*(x/S))
    # + intercept/S stays within a few units. Using S^2 as the product's
    # divisor avoids ever forming the overflowing product.
    try:
        bound = _power_of_two_bound(max(abs(slope), abs(x), abs(intercept)))
        result = bound * (
            bound * (slope / bound) * (x / bound) + intercept / bound
        )
    except (OverflowError, ValueError):
        return direct
    return result if math.isfinite(result) else direct


def _pair_intercept(x_i: float, y_i: float, pair_slope: float) -> float:
    """Intercept ``y_i - pair_slope * x_i`` of the line through a pair.

    Uses :func:`_stable_affine` so a slope*x product that merely shares a
    power of two with the (representable) intercept cannot overflow to
    +-inf and poison the median intercept with a non-finite value.
    """
    return _stable_affine(-pair_slope, x_i, y_i)


def _least_squares(
    inlier_x: list[float], inlier_y: list[float]
) -> tuple[float, float]:
    """Ordinary least-squares slope/intercept for the inlier points.

    For ordinary magnitudes the textbook accumulators on the raw points
    are used unchanged. When a mean or product overflows -- legal
    observed m/z can sit within a few factors of the float maximum -- the
    fit is repeated on power-of-two-scaled centred coordinates: every
    scaled deviation lies in [-1, 1], so the variance/covariance sums
    stay finite, and the scale factors cancel out of the slope (they only
    reappear as the y/x-scale product on the slope and as the x scale on
    the intercept). A genuinely non-representable model still yields a
    non-finite parameter and is rejected by the caller.
    """
    mean_x = _stable_mean(inlier_x)
    mean_y = _stable_mean(inlier_y)
    try:
        sxx = math.fsum((x - mean_x) ** 2 for x in inlier_x)
        sxy = math.fsum(
            (x - mean_x) * (y - mean_y)
            for x, y in zip(inlier_x, inlier_y)
        )
    except (OverflowError, ValueError):
        # OverflowError: a squared deviation exceeded the float range;
        # ValueError ("-inf + inf"): cross products carry both infinite
        # signs when the x and y scales differ. Both take the rescaled path.
        sxx = sxy = float("inf")
    if not (math.isfinite(sxx) and math.isfinite(sxy)):
        # Rescaled path: bound each centred deviation at 1 with exact
        # power-of-two division so no product, square or sum can overflow.
        bound_x = _power_of_two_bound(max(abs(x - mean_x) for x in inlier_x))
        bound_y = _power_of_two_bound(max(abs(y - mean_y) for y in inlier_y))
        scaled_sxx = math.fsum(
            ((x - mean_x) / bound_x) ** 2 for x in inlier_x
        )
        scaled_sxy = math.fsum(
            ((x - mean_x) / bound_x) * ((y - mean_y) / bound_y)
            for x, y in zip(inlier_x, inlier_y)
        )
        sxx = scaled_sxx
        sxy = (bound_y / bound_x) * scaled_sxy
    if sxx <= 0.0:
        return None
    slope = sxy / sxx
    intercept = _stable_affine(-slope, mean_x, mean_y)
    return slope, intercept


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
            intercepts.append(_pair_intercept(x_i, y_i, pair_slope))

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
        # y - prediction via the stable affine: a prediction at the other
        # end of the float range makes the plain subtraction overflow even
        # though the ppm comparison only needs the sign/magnitude.
        residual = _stable_affine(-1.0, _stable_affine(initial_slope, x, initial_intercept), y)
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
    fitted = _least_squares(inlier_x, inlier_y)
    if fitted is None:
        raise CalibrationError(
            "degenerate_inliers",
            "The inlier points share one theoretical m/z, so a "
            "least-squares slope is undefined",
        )
    slope, intercept = fitted
    if not (math.isfinite(slope) and math.isfinite(intercept)):
        raise CalibrationError(
            "non_positive_slope",
            "The least-squares calibration model is not representable as a "
            "finite affine line at the submitted m/z scale; the inverse "
            "transform is undefined (rescale the observed/theoretical m/z "
            "values and refit)",
        )
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
        predicted = _stable_affine(slope, x, intercept)
        residual = _stable_affine(-1.0, predicted, y)
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
    """Apply the fitted inverse transform to one observed m/z.

    Computed as ``mz/slope - intercept/slope``: dividing each term by the
    slope *before* subtracting avoids a spurious overflow of
    ``mz - intercept`` that occurs with a negative intercept -- the two
    large same-magnitude numbers would cancel only after dividing, but the
    subtraction is evaluated first in the textbook form and can reach
    infinity even when the calibrated result is well within the float
    range. A result that genuinely exceeds the float range stays +-inf,
    which is the mathematically correct (non-representable) answer.
    """
    mz_part = mz / fit.slope
    intercept_part = fit.intercept / fit.slope
    return mz_part - intercept_part


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
