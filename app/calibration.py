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

All arithmetic on the m/z values is carried out in shared binary scales
(``frexp``/``ldexp``) rather than in raw floats, so legal observations near
the top of the finite float range neither overflow the least-squares
accumulations (which previously gave an infinite slope or NaN residuals) nor
overflow the inverse subtraction against a negative intercept; every input
whose mathematical result is representable yields a finite slope, intercept,
per-point residual and calibrated m/z.

The fit aborts with :class:`CalibrationError` when no point is valid, when
no finite pair with distinct theoretical m/z exists, when fewer than three
points are inliers, or when the least-squares slope is not a finite positive
number (the inverse transform needs a strictly positive finite slope).
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


def _ldexp(mantissa: float, exponent: int) -> float:
    """``math.ldexp`` that saturates to +-inf instead of raising
    :class:`OverflowError` for a genuinely non-representable result."""
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return math.copysign(math.inf, mantissa)


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
    # Scale the tie average so two large same-sign middle values cannot
    # overflow while being averaged.
    a, b = ordered[mid - 1], ordered[mid]
    scale = max(1.0, abs(a), abs(b))
    result = scale * ((a / scale) + (b / scale)) / 2.0
    if math.isfinite(result):
        return result
    # The true average lies between a and b; when it is not representable
    # return the endpoint with the smaller magnitude instead of +-inf.
    return a if abs(a) < abs(b) else b


def _sum_products(terms: list[tuple[float, float, float]]) -> float:
    """Finite, overflow-safe sum of ``sign * a * b`` terms.

    Each product is decomposed into ``mantissa * 2**exponent`` from the two
    factors (so a product like ``slope * x`` near 1e309 never has to be
    formed) and every term is accumulated in a common binary scale chosen
    from the largest term. Intermediate values then stay in ``[-n, n]``
    (``n <= 200``) instead of overflowing around 1.8e308. Terms with a
    non-finite factor are accumulated naively (such a term genuinely
    dominates). A non-representable result rounds to +-inf; catastrophic
    cancellation can round to zero, but never raises.
    """
    if not terms:
        return 0.0
    if any(
        not (math.isfinite(a) and math.isfinite(b))
        for _, a, b in terms
    ):
        return sum(sign * a * b for sign, a, b in terms)

    decomposed: list[tuple[float, float, int]] = []
    max_exponent = 0
    for sign, a, b in terms:
        if a == 0.0 or b == 0.0:
            continue
        ma, ea = math.frexp(a)
        mb, eb = math.frexp(b)
        mantissa = ma * mb  # |mantissa| in [0.25, 1), always finite
        exponent = ea + eb
        decomposed.append((sign, mantissa, exponent))
        max_exponent = max(max_exponent, exponent)

    total = sum(
        sign * _ldexp(mantissa, exponent - max_exponent)
        for sign, mantissa, exponent in decomposed
    )
    return _ldexp(total, max_exponent)


def _sum_products_divide(
    terms: list[tuple[float, float, float]], denominator: float
) -> float:
    """Overflow-safe ``sum(sign * a * b) / denominator`` for positive finite
    ``denominator``.

    The quotient can be finite even when the numerator alone is not
    representable (e.g. subtracting a large negative intercept from a large
    observed m/z before dividing by a large positive slope), so the
    divisor's binary scale is folded into every term instead of dividing a
    pre-saturated numerator. Callers always pass a positive finite divisor
    (a non-zero m/z difference or a strictly positive fitted slope).
    """
    if not math.isfinite(denominator):
        return sum(sign * a * b for sign, a, b in terms) / denominator
    if denominator == 1.0:
        return _sum_products(terms)
    md, ed = math.frexp(denominator)

    def _term(sign: float, a: float, b: float) -> tuple[float, int] | None:
        if a == 0.0 or b == 0.0:
            return None
        ma, ea = math.frexp(a)
        mb, eb = math.frexp(b)
        return sign * (ma * mb) / md, ea + eb - ed

    if not terms:
        return 0.0
    if any(
        not (math.isfinite(a) and math.isfinite(b))
        for _, a, b in terms
    ):
        return sum(sign * a * b for sign, a, b in terms) / denominator
    decomposed = [
        part
        for sign, a, b in terms
        if (part := _term(sign, a, b)) is not None
    ]
    max_exponent = max((exponent for _, exponent in decomposed), default=0)
    total = sum(
        _ldexp(mantissa, exponent - max_exponent)
        for mantissa, exponent in decomposed
    )
    return _ldexp(total, max_exponent)


def _affine(slope: float, x: float, intercept: float) -> float:
    """Overflow-safe ``slope * x + intercept`` (fused multiply-add style).

    The product is decomposed at the factor level and the addition carried
    out in a common binary scale, so a finite model value does not overflow
    to ``inf`` merely because the two addends are both huge.
    """
    if slope == 0.0:
        return intercept
    if not (math.isfinite(slope) and math.isfinite(x) and math.isfinite(intercept)):
        return slope * x + intercept
    ms, es = math.frexp(slope)
    mx, ex = math.frexp(x)
    mi, ei = math.frexp(intercept)
    scale = max(es + ex, ei)
    return _ldexp(
        _ldexp(ms * mx, es + ex - scale) + _ldexp(mi, ei - scale),
        scale,
    )


def _scaled_divide(numerator: float, denominator: float) -> float:
    """Overflow-safe ``numerator / denominator`` for positive finite
    ``denominator``: the operands are reduced by their binary exponents so
    a quotient that rounds to a finite value never overflows en route."""
    if numerator == 0.0:
        return 0.0
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return numerator / denominator
    mn, en = math.frexp(numerator)
    md, ed = math.frexp(denominator)
    return _ldexp(mn / md, en - ed)


def _residual_ppm(y: float, predicted: float, x: float) -> float:
    """Overflow-safe ``(y - predicted) / x * 1e6`` with ``x`` positive."""
    residual = _sum_products([(1.0, y, 1.0), (-1.0, predicted, 1.0)])
    if x > 0.0 and math.isfinite(x):
        return _scaled_divide(residual, x) * 1e6
    return residual * 1e6


def _product_divide(a: float, b: float, c: float) -> float:
    """Overflow-safe ``a * b / c`` for non-zero finite ``c``.

    ``a * b`` may overflow by itself even though the ratio it contributes
    to is representable; the product is assembled from the factors' binary
    exponents so the intermediate stays within ``[-1, 1]``.
    """
    if a == 0.0 or b == 0.0:
        return 0.0
    if not (math.isfinite(a) and math.isfinite(b)):
        return a * b / c
    ma, ea = math.frexp(a)
    mb, eb = math.frexp(b)
    mc, ec = math.frexp(c)
    # |ma * mb| in [0.25, 1); |mc| in [0.5, 1), and the ratio mantissa
    # stays within ~[-4, 4].
    return _ldexp((ma * mb) / mc, ea + eb - ec)


def _scaled_difference_divide(a: float, b: float, c: float) -> float:
    """Overflow-safe ``(a - b) / c`` for positive finite ``c``.

    The quotient can be finite while the subtraction itself overflows
    (large opposite-sign magnitudes), so the division is fused into the
    scaled accumulation rather than applied to a saturated difference.
    """
    return _sum_products_divide([(1.0, a, 1.0), (-1.0, b, 1.0)], c)


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
    # distinct theoretical m/z define a finite slope. With legal but very
    # large m/z values a pair's slope/intercept can overflow even though a
    # finite model of the well-behaved points exists; such pairs are skipped
    # the same way as the x_i == x_j pairs rather than poisoning the medians.
    slopes: list[float] = []
    intercepts: list[float] = []
    for i in range(len(valid) - 1):
        x_i, y_i = valid[i]
        for j in range(i + 1, len(valid)):
            x_j, y_j = valid[j]
            if x_i == x_j:
                continue
            pair_slope = _scaled_difference_divide(y_j, y_i, x_j - x_i)
            # b = y_i - slope * x_i, evaluated without intermediate overflow.
            pair_intercept = _sum_products(
                [(1.0, y_i, 1.0), (-1.0, pair_slope, x_i)]
            )
            if not (math.isfinite(pair_slope) and math.isfinite(pair_intercept)):
                continue
            slopes.append(pair_slope)
            intercepts.append(pair_intercept)

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
        if not math.isfinite(x):
            # A theoretical m/z beyond the finite float range can never be
            # an inlier of a finite model and would only inject NaNs.
            inlier_flags.append(False)
            continue
        predicted = _affine(initial_slope, x, initial_intercept)
        residual_ppm = _residual_ppm(y, predicted, x)
        inlier_flags.append(
            math.isfinite(residual_ppm)
            and abs(residual_ppm) <= tolerance_ppm + _PPM_EPS
        )

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

    # Ordinary least squares in scaled coordinates u = (x - x_min) / x_range,
    # v = (y - y_mean) / y_scale. Every accumulation then stays within
    # [-1, 1] (n <= 200), so legal observations near 1e308 cannot overflow
    # the sums of squares/products the way mean-centred raw values do, and
    # the scaling adds no round-off beyond the ordinary relative error.
    x_min = min(inlier_x)
    x_max = max(inlier_x)
    x_range = x_max - x_min
    if not (x_range > 0.0) or not math.isfinite(x_range):
        raise CalibrationError(
            "degenerate_inliers",
            "The inlier points share one theoretical m/z, so a "
            "least-squares slope is undefined",
        )
    y_scale = max(abs(y) for y in inlier_y)
    if not (y_scale > 0.0) or not math.isfinite(y_scale):
        # All inlier observations are (non-finite) extremes; no finite
        # positive-slope model can be fitted.
        raise CalibrationError(
            "non_positive_slope",
            "The least-squares calibration slope is not a finite positive "
            "number; the inverse transform is undefined (check the "
            "calibration points for swapped or inconsistent m/z values)",
        )
    mean_u = sum((x - x_min) / x_range for x in inlier_x) / inlier_count
    mean_v = sum(y / y_scale for y in inlier_y) / inlier_count
    s_uu = sum((((x - x_min) / x_range) - mean_u) ** 2 for x in inlier_x)
    s_uv = sum(
        (((x - x_min) / x_range) - mean_u) * ((y / y_scale) - mean_v)
        for x, y in zip(inlier_x, inlier_y)
    )
    if not (s_uu > 0.0):
        raise CalibrationError(
            "degenerate_inliers",
            "The inlier points share one theoretical m/z, so a "
            "least-squares slope is undefined",
        )
    # slope = (s_uv / s_uu) * (y_scale / x_range), assembled so the
    # intermediate product y_scale / x_range cannot spuriously overflow.
    slope = _product_divide(s_uv / s_uu, y_scale, x_range)
    # Means reconstructed from the scaled accumulators: summing the raw
    # inlier values would overflow (n * 1e308) even though each mean is a
    # finite value within the observed range.
    mean_x = _sum_products(
        [(1.0, x_min, 1.0), (1.0, mean_u, x_range)]
    )
    mean_y = mean_v * y_scale
    # intercept = mean(y) - slope * mean(x); evaluated in a shared binary
    # scale, so a finite intercept never cancels to NaN or overflows at the
    # subtraction for a representable (possibly very large-magnitude) value.
    intercept = _sum_products(
        [(1.0, mean_y, 1.0), (-1.0, slope, mean_x)]
    )
    if not (math.isfinite(slope) and slope > 0.0):
        raise CalibrationError(
            "non_positive_slope",
            f"The least-squares calibration slope is {slope:.6g}, which is "
            "not a finite positive number; the inverse transform is "
            "undefined (check the calibration points for swapped or "
            "inconsistent m/z values)",
        )
    if not math.isfinite(intercept):
        # With a finite positive slope this only happens when the true
        # intercept is itself outside the finite range; keep the same
        # 422 rule rather than emitting a non-encodable model.
        raise CalibrationError(
            "non_positive_slope",
            "The least-squares calibration intercept is not finite; the "
            "inverse transform is undefined (check the calibration points "
            "for swapped or inconsistent m/z values)",
        )

    # Final-model residuals for every valid point; the inlier/outlier flag
    # stays the one assigned by the robust initial model.
    valid_iter = iter(zip(valid, inlier_flags))
    for result in points:
        if not result.valid:
            continue
        (x, y), flag = next(valid_iter)
        # Sum the raw terms (never the pre-rounded slope*x + intercept) so
        # cancellation retains full precision for small residuals.
        residual = _sum_products(
            [(1.0, y, 1.0), (-1.0, slope, x), (-1.0, intercept, 1.0)]
        )
        result.residual_mz = residual
        if x > 0.0 and math.isfinite(x):
            result.residual_ppm = _scaled_divide(residual, x) * 1e6
        else:
            result.residual_ppm = residual * 1e6
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

    Computed as ``(mz - intercept) / slope`` with the division fused into
    the scaled accumulation: subtracting a large negative intercept from a
    large positive m/z (or vice versa) must not saturate to ``inf`` before
    the division when the calibrated value itself is representable.
    """
    return _sum_products_divide(
        [(1.0, mz, 1.0), (-1.0, fit.intercept, 1.0)], fit.slope
    )


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
