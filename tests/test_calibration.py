"""Unit tests for the robust mass-axis calibration module."""

import math

import pytest

from app.calibration import (
    CalibrationError,
    CalibrationPoint,
    MassCalibrationFit,
    calibrate_mass_axis,
    inverse_predict,
    theoretical_mz,
)


REFERENCE_IONS = [
    ("CH4O", "+H", 1),       # methanol [M+H]+
    ("C2H6O", "+H", 1),      # ethanol [M+H]+
    ("C6H12O6", "+H", 1),    # glucose [M+H]+
    ("C2H4O2", "-H", -1),    # acetate [M-H]-
    ("C5H5N5", "+H", 1),     # adenine [M+H]+
    ("H2O", "+H", 1),
]


def make_points(observed):
    """Build CalibrationPoint list from (observed_mz, spec) rows."""
    return [
        CalibrationPoint(
            index=i,
            observed_mz=mz,
            formula=formula,
            adduct=adduct,
            charge=charge,
        )
        for i, (mz, (formula, adduct, charge)) in enumerate(observed)
    ]


def affine_points(slope, intercept, specs):
    return make_points(
        [(slope * theoretical_mz(*spec) + intercept, spec) for spec in specs]
    )


def test_theoretical_mz_matches_ion_envelope_rule():
    # glucose [M+H]+: 180.063388 neutral + H - electron
    mz = theoretical_mz("C6H12O6", "+H", 1)
    assert mz == pytest.approx(181.070665, abs=1e-6)
    # doubly charged [M+2H]2+
    mz2 = theoretical_mz("C6H12O6", "+2H", 2)
    assert mz2 == pytest.approx(91.038971, abs=1e-6)
    # negative ion [M-H]-
    mz_neg = theoretical_mz("C6H12O6", "-H", -1)
    assert mz_neg == pytest.approx(179.056112, abs=1e-6)


def test_theoretical_mz_supports_labels():
    mz_natural = theoretical_mz("C6H12O6", "+H", 1)
    mz_label = theoretical_mz("C5H12O6[13C]", "+H", 1)
    assert mz_label - mz_natural == pytest.approx(
        13.00335483507 - 12.0, abs=1e-9
    )


def test_perfect_affine_fit_recovers_parameters():
    slope, intercept = 1.000003, -0.0007
    points = affine_points(slope, intercept, REFERENCE_IONS[:5])
    fit = calibrate_mass_axis(points, tolerance_ppm=5.0)
    assert fit.slope == pytest.approx(slope, abs=1e-12)
    assert fit.intercept == pytest.approx(intercept, abs=1e-12)
    assert fit.inlier_count == 5
    assert fit.valid_point_count == 5
    # Inverse transform reproduces the theoretical m/z.
    for p in fit.points:
        assert p.inlier is True
        assert abs(p.residual_ppm) < 1e-6
        assert inverse_predict(fit, p.observed_mz) == pytest.approx(
            p.theoretical_mz, abs=1e-9
        )


def test_gross_outlier_excluded_then_least_squares_recovers_line():
    slope, intercept = 1.000002, -0.0005
    rows = []
    for spec in REFERENCE_IONS[:5]:
        x = theoretical_mz(*spec)
        rows.append((slope * x + intercept, spec))
    # A wildly off point at the end: -10% observed m/z.
    bad_spec = REFERENCE_IONS[5]
    rows.append((0.9 * theoretical_mz(*bad_spec), bad_spec))
    points = make_points(rows)

    fit = calibrate_mass_axis(points, tolerance_ppm=10.0)
    assert fit.inlier_count == 5
    assert fit.points[-1].inlier is False
    assert fit.slope == pytest.approx(slope, abs=1e-10)
    assert fit.intercept == pytest.approx(intercept, abs=1e-9)
    # Final-model residual of the outlier is reported even though it is excluded.
    outlier = fit.points[-1]
    assert outlier.valid is True
    assert abs(outlier.residual_ppm) > 1000.0


def test_points_kept_in_request_order():
    slope = 1.000001
    specs = list(reversed(REFERENCE_IONS[:5]))
    points = affine_points(slope, 0.0001, specs)
    fit = calibrate_mass_axis(points, 5.0)
    assert [p.index for p in fit.points] == list(range(5))
    assert [p.formula for p in fit.points] == [s[0] for s in specs]


def test_invalid_point_reported_in_place_and_ignored_in_fit():
    slope = 1.000001
    good_specs = REFERENCE_IONS[:3]
    rows = [(slope * theoretical_mz(*spec) + 0.0002, spec) for spec in good_specs]
    rows.append((500.0, ("C6Xx", "+H", 1)))  # unknown element
    points = make_points(rows)

    fit = calibrate_mass_axis(points, 5.0)
    assert fit.valid_point_count == 3
    invalid = fit.points[3]
    assert invalid.valid is False
    assert invalid.error == "invalid_formula"
    assert invalid.theoretical_mz is None
    assert invalid.residual_ppm is None
    assert invalid.inlier is None
    assert fit.slope == pytest.approx(slope, abs=1e-10)


def test_invalid_adduct_point_uses_existing_adduct_error():
    good = REFERENCE_IONS[:3]
    rows = [(1.0001 * theoretical_mz(*spec), spec) for spec in good]
    rows.append((100.0, ("C6H12O6", "-99H", -1)))
    fit = calibrate_mass_axis(make_points(rows), 200.0)
    invalid = fit.points[3]
    assert invalid.valid is False
    assert invalid.error == "invalid_adduct"
    assert "H" in invalid.message


def test_all_points_invalid_raises_no_valid_points():
    points = [
        CalibrationPoint(0, 1.0, "C6Xx", "+H", 1),
        CalibrationPoint(1, 1.0, "Zz", "+H", 1),
        CalibrationPoint(2, 1.0, "H2O", "-99H", 1),
    ]
    with pytest.raises(CalibrationError) as exc:
        calibrate_mass_axis(points, 5.0)
    assert exc.value.error == "no_valid_points"


def test_identical_theoretical_mz_raises_no_valid_point_pairs():
    spec = ("CH4O", "+H", 1)
    x = theoretical_mz(*spec)
    points = make_points(
        [(x, spec), (x * 1.000001, spec), (x * 0.999999, spec)]
    )
    with pytest.raises(CalibrationError) as exc:
        calibrate_mass_axis(points, 500.0)
    assert exc.value.error == "no_valid_point_pairs"


def test_fewer_than_three_inliers_raises():
    specs = REFERENCE_IONS[:3]
    rows = []
    for i, spec in enumerate(specs):
        x = theoretical_mz(*spec)
        # First point sits on y=x; the other two are +1% off (hundreds of ppm).
        rows.append((x if i == 0 else x * 1.01, spec))
    with pytest.raises(CalibrationError) as exc:
        calibrate_mass_axis(make_points(rows), 1.0)
    assert exc.value.error == "too_few_inliers"
    assert "at least 3" in exc.value.message


def test_boundary_residual_is_inlier():
    # Closed bound: a residual exactly at tolerance_ppm is retained; one
    # just beyond it is not. The initial model is forced onto y=x by
    # triplicating two on-line points, so the perturbed point's residual
    # against the median model is its own ppm offset (almost exactly).
    on_line = [REFERENCE_IONS[1], REFERENCE_IONS[2]]

    def fit_with(extra_ppm, tol):
        rows = [
            (theoretical_mz(*spec), spec)
            for spec in on_line
            for _ in range(3)
        ]
        spec = REFERENCE_IONS[0]
        rows.append((theoretical_mz(*spec) * (1.0 + extra_ppm * 1e-6), spec))
        return calibrate_mass_axis(make_points(rows), tol)

    tol = 5.0
    fit = fit_with(tol, tol)
    assert fit.inlier_count == 7
    assert all(p.inlier for p in fit.points)

    beyond = fit_with(6.0, tol)
    assert beyond.inlier_count == 6
    assert beyond.points[-1].inlier is False


def test_negative_slope_raises_non_positive_slope():
    specs = REFERENCE_IONS[:3]
    rows = [(-2.0 * theoretical_mz(*spec) + 600.0, spec) for spec in specs]
    with pytest.raises(CalibrationError) as exc:
        calibrate_mass_axis(make_points(rows), 1e7)
    assert exc.value.error == "non_positive_slope"


def test_residual_sign_convention():
    # Residual = observed - predicted under the final model; a point above
    # the fitted identity line must report a positive residual even after
    # least-squares pulls the line slightly toward it.
    spec = REFERENCE_IONS[0]
    x = theoretical_mz(*spec)
    points = make_points(
        [
            (theoretical_mz(*REFERENCE_IONS[1]), REFERENCE_IONS[1]),
            (theoretical_mz(*REFERENCE_IONS[2]), REFERENCE_IONS[2]),
            (x * (1.0 + 2e-6), spec),  # +2 ppm above the identity line
        ]
    )
    fit = calibrate_mass_axis(points, 10.0)
    target = next(p for p in fit.points if p.formula == spec[0])
    assert target.residual_mz > 0
    assert target.residual_ppm > 0


def test_huge_observed_values_keep_least_squares_finite():
    # Observed m/z near the float maximum are legal finite inputs. The
    # physical-unit OLS accumulators (sums, squared/cross deviations)
    # overflow there and used to yield an infinite slope with NaN
    # residuals; the scaled accumulators must still recover the model.
    big_specs = [
        ("C" + "9" * 305, "+H", 1),
        ("C" + "9" * 306, "+H", 1),
        ("C" + "9" * 307, "+H", 1),
    ]
    slope = 1.0000001
    rows = [(slope * theoretical_mz(*spec), spec) for spec in big_specs]
    fit = calibrate_mass_axis(make_points(rows), 5.0)
    assert math.isfinite(fit.slope)
    assert math.isfinite(fit.intercept)
    assert fit.slope == pytest.approx(slope, rel=1e-6)
    assert fit.inlier_count == 3
    for point in fit.points:
        assert point.inlier is True
        assert math.isfinite(point.residual_mz)
        assert math.isfinite(point.residual_ppm)
        assert abs(point.residual_ppm) < 1.0


def test_huge_slope_least_squares_does_not_overflow():
    # Slope near 1e300 with ordinary theoretical m/z: y values around
    # 1e302 keep the mean finite in a naive sum, but a still-larger slope
    # (or a few more points) pushes the accumulators over the float edge;
    # the intercept must stay at the y-scale to be representable there.
    slope, intercept = 1e300, 1e287
    rows = [
        (slope * theoretical_mz(*spec) + intercept, spec)
        for spec in REFERENCE_IONS[:5]
    ]
    fit = calibrate_mass_axis(make_points(rows), 5.0)
    assert math.isfinite(fit.slope)
    assert math.isfinite(fit.intercept)
    assert fit.slope == pytest.approx(slope, rel=1e-6)
    for point in fit.points:
        assert math.isfinite(point.residual_ppm)


def test_inverse_predict_negative_intercept_does_not_overflow():
    # (mz - intercept)/slope with intercept = -1e308 subtracts two
    # max-scale positives and overflows before the division; dividing by
    # the slope first keeps the representable result finite.
    fit = MassCalibrationFit(
        slope=2.0,
        intercept=-1e308,
        valid_point_count=0,
        inlier_count=0,
        points=[],
    )
    assert inverse_predict(fit, 1e308) == pytest.approx(1e308, rel=0)
    assert math.isfinite(inverse_predict(fit, 1.0))


def test_inverse_predict_positive_intercept_large_mz():
    fit = MassCalibrationFit(
        slope=2.0,
        intercept=1e308,
        valid_point_count=0,
        inlier_count=0,
        points=[],
    )
    # mz == intercept: difference is zero either way.
    assert inverse_predict(fit, 1e308) == 0.0


def test_end_to_end_huge_inputs_return_finite_calibration():
    # Full chain: huge inlier fit plus a huge m/z to calibrate, whose
    # inverse transform used to be defeated by the subtraction order.
    big_specs = [
        ("C" + "9" * 305, "+H", 1),
        ("C" + "9" * 306, "+H", 1),
        ("C" + "9" * 307, "+H", 1),
    ]
    slope = 1.0000002
    rows = [(slope * theoretical_mz(*spec), spec) for spec in big_specs]
    fit = calibrate_mass_axis(make_points(rows), 5.0)
    big_mz = theoretical_mz(*big_specs[-1]) * slope
    calibrated = inverse_predict(fit, big_mz)
    assert math.isfinite(calibrated)
    assert calibrated == pytest.approx(theoretical_mz(*big_specs[-1]), rel=1e-9)


def test_least_squares_cross_scale_deviations_stay_finite():
    # x near 1e308 and y = 2*x - 1e308 near 1e307: the centred x and y
    # deviations sit an order of magnitude apart, so naive cross products
    # carry both infinite signs and math.fsum raises "-inf + inf"; the
    # centroid sums themselves also overflow before division by n.
    counts = ("5" + "0" * 306, "6" + "0" * 306, "7" + "0" * 306)
    specs = [("C" + c, "+H", 1) for c in counts]
    slope, intercept = 2.0, -1e308
    rows = [
        (slope * theoretical_mz(*spec) + intercept, spec) for spec in specs
    ]
    fit = calibrate_mass_axis(make_points(rows), 5.0)
    assert math.isfinite(fit.slope) and math.isfinite(fit.intercept)
    assert fit.slope == pytest.approx(2.0, rel=1e-9)
    for point in fit.points:
        assert math.isfinite(point.residual_ppm)
    # The exact case the old inverse transform dropped to infinity:
    assert inverse_predict(fit, 1e308) == pytest.approx(1e308, rel=1e-9)
