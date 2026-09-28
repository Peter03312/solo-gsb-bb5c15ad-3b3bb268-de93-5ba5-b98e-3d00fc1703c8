"""Tests for continuous-profile centroid conversion (unit + API)."""

import math

import pytest
from fastapi.testclient import TestClient

from app.centroid import (
    MAX_POINTS,
    MIN_POINTS,
    STEP_REL_TOLERANCE,
    CentroidError,
    centroid_profile,
)
from app.config import settings


def grid_profile(levels: list[float], start: float = 100.0, step: float = 0.01):
    """Build an equally spaced (mass, intensity) profile from levels."""
    return [(round(start + k * step, 12), value) for k, value in enumerate(levels)]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_single_triangle_peak_uses_weighted_centroid():
    points = grid_profile([0.0, 1.0, 2.0, 4.0, 2.0, 1.0, 0.0])
    res = centroid_profile(points, 0.0, 1.0)
    assert res.candidate_count == 1
    peak = res.peaks[0]
    weights = 1.0 + 2.0 + 4.0 + 2.0 + 1.0
    expected_mass = sum(m * i for m, i in points) / weights
    assert peak.mass == pytest.approx(expected_mass)
    # Symmetric profile: centroid lands on the apex grid point.
    assert peak.mass == pytest.approx(100.03)
    assert peak.intensity == pytest.approx(100.0)
    assert peak.total_intensity_percent == pytest.approx(100.0)


def test_flat_top_plateau_position_is_midpoint():
    points = grid_profile([0.0, 1.0, 4.0, 4.0, 4.0, 1.0, 0.0])
    res = centroid_profile(points, 0.0, 1.0)
    assert res.candidate_count == 1
    peak = res.peaks[0]
    # Plateau spans indices 2..4 -> midpoint mass 100.03.
    assert peak.intensity == pytest.approx(100.0)
    weights = 1.0 + 4.0 + 4.0 + 4.0 + 1.0
    expected_mass = sum(m * i for m, i in points) / weights
    assert peak.mass == pytest.approx(expected_mass)


def test_two_flat_topped_plateaus_both_reported():
    levels = [0.0, 3.0, 3.0, 0.0, 2.0, 2.0, 0.0]
    points = grid_profile(levels, step=0.1)
    res = centroid_profile(points, 0.0, 0.15)
    assert res.candidate_count == 2
    assert [p.intensity for p in res.peaks] == [100.0, pytest.approx(200.0 / 3.0)]
    # Candidate positions are plateau midpoints: indices 1-2 -> 100.15,
    # indices 4-5 -> 100.45.
    assert res.peaks[0].mass < res.peaks[1].mass


def test_threshold_filters_low_points_before_centroid():
    levels = [0.0, 0.001, 1.0, 0.001, 0.0]
    points = grid_profile(levels)
    # Threshold 5% of base: the 0.001 flanks (0.1% of base) are dropped,
    # leaving only the apex point.
    res = centroid_profile(points, 5.0, 1.0)
    assert res.candidate_count == 1
    assert res.peaks[0].mass == pytest.approx(100.02)
    assert res.peaks[0].total_intensity_percent == pytest.approx(100.0)


def test_threshold_is_inclusive():
    # A plateau at exactly 10% relative intensity must survive a 10%
    # threshold (inclusive comparison).
    points = grid_profile([0.0, 1.0, 10.0, 10.0, 1.0, 0.0])
    res = centroid_profile(points, 10.0, 1.0)
    assert res.candidate_count == 1
    assert res.peaks[0].intensity == pytest.approx(100.0)


def test_threshold_100_keeps_base_plateau():
    points = grid_profile([1.0, 10.0, 10.0, 1.0])
    res = centroid_profile(points, 100.0, 1.0)
    assert res.candidate_count == 1
    peak = res.peaks[0]
    assert peak.intensity == pytest.approx(100.0)
    # Only the two plateau points (intensity 10) are assigned.
    assert peak.mass == pytest.approx(100.015)
    assert peak.total_intensity_percent == pytest.approx(100.0)


def test_nms_drops_closer_lower_candidate():
    # Candidates at 100.01 (intensity 50) and 100.03 (intensity 100),
    # separated by exactly 0.02 < min_separation 0.025.
    points = grid_profile([0.0, 5.0, 0.0, 10.0, 0.0])
    res = centroid_profile(points, 0.0, 0.025)
    assert res.candidate_count == 2
    assert len(res.peaks) == 1
    # The 100-level candidate (index 3, mass 100.03) wins; every kept point
    # is assigned to it, so the centroid is pulled toward 100.01.
    assert res.peaks[0].intensity == pytest.approx(100.0)
    assert res.peaks[0].mass == pytest.approx(
        (100.01 * 5.0 + 100.03 * 10.0) / 15.0
    )
    assert res.peaks[0].total_intensity_percent == pytest.approx(100.0)


def test_nms_equal_intensity_prefers_lower_mass():
    points = grid_profile([0.0, 5.0, 0.0, 5.0, 0.0])
    res = centroid_profile(points, 0.0, 0.025)
    assert res.candidate_count == 2
    assert len(res.peaks) == 1
    # Lower candidate (index 1, mass 100.01) survives and absorbs all points.
    assert res.peaks[0].intensity == pytest.approx(100.0)
    assert res.peaks[0].mass == pytest.approx(100.02)


def test_nms_keeps_candidates_at_exactly_min_separation():
    points = grid_profile([0.0, 5.0, 0.0, 10.0, 0.0])
    res = centroid_profile(points, 0.0, 0.02)
    assert len(res.peaks) == 2
    assert [p.mass for p in res.peaks] == [100.01, 100.03]


def test_assignment_equidistant_tie_goes_to_lower_mass():
    # Candidates at 100.00 (level 50) and 100.04 (level 50), midpoint point
    # at 100.02 equidistant. min_separation 0.025 would NMS them, so use
    # positions one grid step further apart: 100.00 and 100.04 are 0.04
    # apart, both survive min_separation 0.025... but the inner point at
    # 100.02 is equidistant and must go to the lower-mass candidate.
    points = grid_profile([5.0, 1.0, 0.9, 1.0, 5.0])
    res = centroid_profile(points, 0.0, 0.025)
    assert len(res.peaks) == 2
    left, right = res.peaks
    # Left owner gets points 5+1+0.9 = 6.9, right gets 1+5 = 6.
    assert left.total_intensity_percent == pytest.approx(100.0 * 6.9 / 12.9)
    assert right.total_intensity_percent == pytest.approx(100.0 * 6.0 / 12.9)
    assert left.total_intensity_percent + right.total_intensity_percent == pytest.approx(100.0)


def test_intensity_is_peak_top_normalised_even_when_assignment_shifts_mass():
    # A weak satellite kept as a separate peak (far enough from the base
    # peak) reports its own plateau level, not an assignment-area quantity.
    levels = [0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0]
    points = grid_profile(levels, step=0.1)
    res = centroid_profile(points, 0.0, 0.2)
    assert len(res.peaks) == 2
    weak, strong = res.peaks
    assert weak.intensity == pytest.approx(40.0)
    assert strong.intensity == pytest.approx(100.0)
    # Each isolated peak owns only its own positive point.
    assert weak.mass == pytest.approx(100.1)
    assert strong.mass == pytest.approx(100.7)


def test_peaks_returned_by_ascending_mass():
    levels = [0.0, 4.0, 0.0, 0.0, 10.0, 0.0]
    points = grid_profile(levels, step=0.1)
    res = centroid_profile(points, 0.0, 0.2)
    masses = [p.mass for p in res.peaks]
    assert masses == sorted(masses)


def test_share_of_total_uses_original_unscaled_intensities():
    # Doubling every raw intensity must not change the percentage split.
    levels = [0.0, 2.0, 0.0, 0.0, 5.0, 0.0]
    a = centroid_profile(grid_profile(levels, step=0.1), 0.0, 0.2)
    b = centroid_profile(
        grid_profile([2.0 * v for v in levels], step=0.1), 0.0, 0.2
    )
    for pa, pb in zip(a.peaks, b.peaks):
        assert pa.total_intensity_percent == pytest.approx(pb.total_intensity_percent)
        assert pa.intensity == pytest.approx(pb.intensity)
        assert pa.mass == pytest.approx(pb.mass)


def test_large_finite_intensities_do_not_overflow():
    # Finite intensities near the float maximum used to overflow in
    # 100*intensity and in the mass*intensity products, yielding
    # inf/NaN; the endpoint now returns finite values.
    points = [(100.0, 8e307), (101.0, 1e308), (102.0, 8e307)]
    res = centroid_profile(points, 5.0, 0.5)
    assert res.candidate_count == 1
    peak = res.peaks[0]
    assert math.isfinite(peak.mass)
    assert math.isfinite(peak.intensity)
    assert math.isfinite(peak.total_intensity_percent)
    assert peak.mass == pytest.approx(101.0)
    assert peak.intensity == pytest.approx(100.0)
    assert peak.total_intensity_percent == pytest.approx(100.0)


def test_large_intensities_match_scaled_down_profile():
    # Multiplying every intensity by a common finite factor (even one that
    # pushes the values to ~1e308) must leave the centroid, normalised peak
    # intensity and percentage split unchanged.
    levels = [0.0, 8.0, 0.0, 0.0, 10.0, 0.0]
    small = centroid_profile(grid_profile(levels, step=0.1), 5.0, 0.2)
    factor = 1e307
    big = centroid_profile(
        grid_profile([factor * v for v in levels], step=0.1), 5.0, 0.2
    )
    assert big.candidate_count == small.candidate_count
    for b, s in zip(big.peaks, small.peaks):
        assert math.isfinite(b.mass)
        assert math.isfinite(b.intensity)
        assert math.isfinite(b.total_intensity_percent)
        assert b.mass == pytest.approx(s.mass)
        assert b.intensity == pytest.approx(s.intensity)
        assert b.total_intensity_percent == pytest.approx(
            s.total_intensity_percent
        )
    assert sum(p.total_intensity_percent for p in big.peaks) == pytest.approx(100.0)


def test_large_intensity_threshold_uses_ratio_before_scale():
    # The flanks are 80% of the base peak; the threshold comparison must be
    # made on the ratio intensity/base, not on 100*intensity which overflows.
    points = [(100.0, 8e307), (101.0, 1e308), (102.0, 8e307)]
    assert len(centroid_profile(points, 50.0, 0.5).peaks) == 1
    res = centroid_profile(points, 90.0, 0.5)
    assert len(res.peaks) == 1
    # Flanks dropped: the base peak alone owns the assigned intensity.
    assert res.peaks[0].total_intensity_percent == pytest.approx(100.0)


def test_threshold_100_with_float_max_base_peak():
    points = [
        (100.0, 8e307),
        (101.0, 1.7976931348623157e308),
        (102.0, 8e307),
    ]
    res = centroid_profile(points, 100.0, 0.5)
    assert res.candidate_count == 1
    peak = res.peaks[0]
    assert peak.intensity == pytest.approx(100.0)
    assert math.isfinite(peak.mass)
    assert math.isfinite(peak.total_intensity_percent)


def test_giant_mass_and_intensity_centroid_stays_finite():
    # Rescue path: the mass*weight products overflow even with weights in
    # [0, 1] because the masses themselves are ~1e308.
    points = [(1e308, 8e307), (1.01e308, 1e308), (1.02e308, 8e307)]
    res = centroid_profile(points, 5.0, 1e305)
    assert res.candidate_count == 1
    peak = res.peaks[0]
    assert math.isfinite(peak.mass)
    assert math.isfinite(peak.intensity)
    assert math.isfinite(peak.total_intensity_percent)
    assert peak.mass == pytest.approx(1.01e308, rel=1e-12)


def test_flat_plateau_near_float_ceiling_has_finite_midpoint():
    # A two-point plateau whose endpoint masses add to more than the float
    # maximum used to get an inf plateau midpoint; the overflow-free
    # half-each-endpoint form keeps the candidate position finite.
    points = [
        (1.70e308, 8.0),
        (1.73e308, 10.0),
        (1.76e308, 10.0),
        (1.79e308, 8.0),
    ]
    res = centroid_profile(points, 0.0, 1e307)
    assert res.candidate_count == 1
    peak = res.peaks[0]
    assert math.isfinite(peak.mass)
    assert math.isfinite(peak.intensity)
    assert math.isfinite(peak.total_intensity_percent)
    assert peak.mass == pytest.approx(1.745e308, rel=1e-12)
    assert peak.intensity == pytest.approx(100.0)


def test_giant_mass_twin_plateaus_both_kept_with_finite_centroids():
    # Regression: two flat-topped plateaus near the float ceiling whose
    # separation exceeds min_separation. Their (left+right)/2 midpoints both
    # overflowed to inf, so NMS measured abs(inf-inf)=NaN and dropped the
    # second candidate as "coincident"; all points were then assigned to the
    # single surviving peak. Both plateaus must now survive with finite
    # centroids and their own intensity shares. Zero-intensity gaps keep the
    # point assignment unambiguous (the giant-mass ULP is ~2e292).
    step = 1e306
    m0 = 1.0e308
    points = [
        (m0, 10.0),              # plateau A, indices 0-1
        (m0 + step, 10.0),
        (m0 + 2 * step, 0.0),    # zero gap
        (m0 + 3 * step, 5.0),    # plateau B, indices 3-4
        (m0 + 4 * step, 5.0),
    ]
    res = centroid_profile(points, 0.0, 2.0 * step)
    assert res.candidate_count == 2
    assert len(res.peaks) == 2
    for peak in res.peaks:
        assert math.isfinite(peak.mass)
        assert math.isfinite(peak.intensity)
        assert math.isfinite(peak.total_intensity_percent)
    left, right = res.peaks
    assert left.mass < right.mass
    # Each plateau owns exactly its own points, so the centroid equals the
    # (overflow-free) plateau midpoint and the shares are 20/30 and 10/30.
    assert left.mass == pytest.approx(1.005e308, rel=1e-12)
    assert right.mass == pytest.approx(1.035e308, rel=1e-12)
    assert left.intensity == pytest.approx(100.0)
    assert right.intensity == pytest.approx(50.0)
    assert left.total_intensity_percent == pytest.approx(100.0 * 20.0 / 30.0)
    assert right.total_intensity_percent == pytest.approx(100.0 * 10.0 / 30.0)


def test_giant_mass_nms_still_suppresses_genuinely_close_pair():
    # The overflow fix must not loosen NMS: two maxima genuinely closer than
    # min_separation still collapse to one peak.
    step = 1e306
    m0 = 1.0e308
    points = [
        (m0, 10.0),
        (m0 + step, 10.0),
        (m0 + 2 * step, 1.0),
        (m0 + 3 * step, 5.0),
        (m0 + 4 * step, 5.0),
    ]
    res = centroid_profile(points, 0.0, 4.0 * step)
    assert res.candidate_count == 2
    assert len(res.peaks) == 1
    assert math.isfinite(res.peaks[0].mass)


def test_plateau_midpoint_between_grid_points():
    # Even-length plateau (indices 1-2): midpoint lies between grid points.
    points = grid_profile([0.0, 5.0, 5.0, 0.0])
    res = centroid_profile(points, 0.0, 1.0)
    # Candidate position is 100.015; all points are assigned to it but the
    # reported mass is the intensity-weighted centroid (also symmetric here).
    expected = sum(m * i for m, i in points) / sum(i for _, i in points)
    assert res.peaks[0].mass == pytest.approx(expected)
    assert expected == pytest.approx(100.015)


def test_edge_plateau_is_a_local_maximum():
    points = grid_profile([5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], step=0.1)
    res = centroid_profile(points, 0.0, 0.2)
    assert res.candidate_count == 2
    masses = [p.mass for p in res.peaks]
    assert masses[0] == pytest.approx(100.05)  # left-edge plateau midpoint
    assert masses[1] == pytest.approx(100.6)


def test_shoulder_equal_level_runs_are_not_two_candidates():
    # A single monotone run below the apex may contain repeated levels; only
    # strict local maxima become candidates.
    points = grid_profile([0.0, 2.0, 2.0, 4.0, 2.0, 2.0, 0.0])
    res = centroid_profile(points, 0.0, 1.0)
    assert res.candidate_count == 1


def test_gaps_checked_against_first_gap_relative_error():
    # A deviation at the 1e-6 boundary is accepted; beyond it is rejected.
    points = [(100.0, 1.0), (100.1, 1.0), (100.20000005, 1.0)]
    res = centroid_profile(points, 0.0, 1.0)
    assert res.point_count == 3

    bad = [(100.0, 1.0), (100.1, 1.0), (100.2001, 1.0)]
    with pytest.raises(CentroidError) as exc:
        centroid_profile(bad, 0.0, 1.0)
    assert exc.value.error == "invalid_mass_grid"
    assert exc.value.index == 2


def test_non_increasing_mass_reports_index():
    with pytest.raises(CentroidError) as exc:
        centroid_profile(
            [(1.0, 1.0), (3.0, 1.0), (2.0, 1.0), (4.0, 1.0)], 0.0, 1.0
        )
    assert exc.value.error == "invalid_mass_grid"
    assert exc.value.index == 2


def test_equal_mass_reports_index():
    with pytest.raises(CentroidError) as exc:
        centroid_profile([(1.0, 1.0), (1.0, 1.0), (2.0, 1.0)], 0.0, 1.0)
    assert exc.value.error == "invalid_mass_grid"
    assert exc.value.index == 1


def test_all_zero_profile_is_rejected():
    with pytest.raises(CentroidError) as exc:
        centroid_profile(grid_profile([0.0, 0.0, 0.0]), 0.0, 1.0)
    assert exc.value.error == "zero_profile"
    assert exc.value.field == "profile"


def test_large_uniform_grid_accepted():
    n = MAX_POINTS
    points = [(float(k), 1.0 if 0 < k < n - 1 else 0.0) for k in range(n)]
    res = centroid_profile(points, 0.0, 1.0)
    assert res.point_count == n
    assert len(res.peaks) == 1


@pytest.mark.parametrize("n", [MIN_POINTS - 1, 0, MAX_POINTS + 1])
def test_point_count_bounds(n):
    points = [(float(k), 1.0) for k in range(n)] if n else []
    with pytest.raises(CentroidError) as exc:
        centroid_profile(points, 0.0, 1.0)
    assert exc.value.error == "invalid_profile"
    assert exc.value.field == "profile"


def test_negative_intensity_rejected():
    with pytest.raises(CentroidError) as exc:
        centroid_profile([(1.0, 1.0), (2.0, -0.5), (3.0, 1.0)], 0.0, 1.0)
    assert exc.value.error == "invalid_profile"
    assert exc.value.index == 1


def test_constant_tolerance_value():
    assert STEP_REL_TOLERANCE == 1e-6


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def two_peak_payload(**overrides):
    levels = [0.0, 2.0, 0.0, 0.0, 0.0, 10.0, 2.0, 0.0]
    profile = [
        {"mass": round(100.0 + 0.1 * k, 6), "intensity": v}
        for k, v in enumerate(levels)
    ]
    payload = {
        "profile": profile,
        "relative_threshold": 0.0,
        "min_separation": 0.2,
    }
    payload.update(overrides)
    return payload


def test_api_centroid_two_peaks(client):
    resp = client.post("/api/v1/profile-centroid", json=two_peak_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["point_count"] == 8
    assert body["candidate_count"] == 2
    assert body["peak_count"] == 2
    assert len(body["peaks"]) == 2
    weak, strong = body["peaks"]
    assert weak["intensity"] == pytest.approx(20.0)
    assert strong["intensity"] == pytest.approx(100.0)
    assert weak["mass"] < strong["mass"]
    total = sum(p["total_intensity_percent"] for p in body["peaks"])
    assert total == pytest.approx(100.0)
    assert body["relative_threshold"] == 0.0
    assert body["min_separation"] == 0.2


def test_api_centroid_flat_top(client):
    levels = [0.0, 1.0, 5.0, 5.0, 1.0, 0.0]
    payload = {
        "profile": [
            {"mass": 50.0 + 0.01 * k, "intensity": v} for k, v in enumerate(levels)
        ],
        "relative_threshold": 0.0,
        "min_separation": 1.0,
    }
    resp = client.post("/api/v1/profile-centroid", json=payload)
    assert resp.status_code == 200, resp.text
    peak = resp.json()["peaks"][0]
    # All five positive points assigned: weighted centroid.
    expected = sum((50.0 + 0.01 * k) * v for k, v in enumerate(levels)) / 12.0
    assert peak["mass"] == pytest.approx(expected, abs=1e-6)
    assert peak["intensity"] == pytest.approx(100.0)
    assert peak["total_intensity_percent"] == pytest.approx(100.0)


def test_api_centroid_rounds_to_six_decimals(client):
    payload = two_peak_payload()
    body = client.post("/api/v1/profile-centroid", json=payload).json()
    for peak in body["peaks"]:
        for key in ("mass", "intensity", "total_intensity_percent"):
            assert isinstance(peak[key], float)
            assert abs(peak[key] - round(peak[key], 6)) < 1e-12


def test_api_centroid_nms_collapses_close_pair(client):
    payload = two_peak_payload(min_separation=1.0)
    body = client.post("/api/v1/profile-centroid", json=payload).json()
    assert body["candidate_count"] == 2
    assert body["peak_count"] == 1
    assert body["peaks"][0]["intensity"] == pytest.approx(100.0)
    # The weak satellite's points are assigned to the surviving base peak.
    assert body["peaks"][0]["mass"] < 100.5


def test_api_non_uniform_grid_422_with_index(client):
    payload = two_peak_payload()
    payload["profile"][3]["mass"] += 0.05
    resp = client.post("/api/v1/profile-centroid", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_mass_grid"
    assert detail["field"] == "profile"
    assert detail["index"] == 3


def test_api_non_increasing_mass_422_with_index(client):
    payload = two_peak_payload()
    payload["profile"][2]["mass"] = payload["profile"][3]["mass"]
    resp = client.post("/api/v1/profile-centroid", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_mass_grid"
    assert detail["index"] == 2


def test_api_all_zero_profile_422(client):
    payload = two_peak_payload()
    for point in payload["profile"]:
        point["intensity"] = 0.0
    resp = client.post("/api/v1/profile-centroid", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "zero_profile"
    assert detail["field"] == "profile"
    assert detail["index"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"relative_threshold": -0.1},
        {"relative_threshold": 100.1},
        {"min_separation": 0.0},
        {"min_separation": -1.0},
    ],
)
def test_api_parameter_bounds_422(client, change):
    resp = client.post(
        "/api/v1/profile-centroid", json=two_peak_payload(**change)
    )
    assert resp.status_code == 422


def test_api_negative_intensity_422(client):
    payload = two_peak_payload()
    payload["profile"][1]["intensity"] = -2.0
    resp = client.post("/api/v1/profile-centroid", json=payload)
    assert resp.status_code == 422


def test_api_profile_length_bounds_422(client):
    short = {
        "profile": [
            {"mass": 1.0, "intensity": 1.0},
            {"mass": 1.01, "intensity": 2.0},
        ],
        "relative_threshold": 0.0,
        "min_separation": 0.1,
    }
    assert client.post("/api/v1/profile-centroid", json=short).status_code == 422

    long_profile = [
        {"mass": 0.01 * k, "intensity": 1.0} for k in range(MAX_POINTS + 1)
    ]
    long_payload = {
        "profile": long_profile,
        "relative_threshold": 0.0,
        "min_separation": 0.1,
    }
    assert client.post("/api/v1/profile-centroid", json=long_payload).status_code == 422


def test_api_missing_fields_422(client):
    resp = client.post("/api/v1/profile-centroid", json={"profile": []})
    assert resp.status_code == 422


def test_api_non_finite_values_422(client):
    import json

    payload = two_peak_payload()
    payload["profile"][0]["mass"] = float("inf")
    resp = client.post(
        "/api/v1/profile-centroid",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422

    payload = two_peak_payload()
    payload["min_separation"] = float("nan")
    resp = client.post(
        "/api/v1/profile-centroid",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_api_documented_example_round_trip(client):
    # End-to-end: feed a /neutral-profile style response straight back in.
    levels = [
        0.0, 0.5, 2.0, 8.0, 25.0, 60.0, 92.0, 100.0, 92.0, 60.0,
        25.0, 8.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.5, 2.0, 7.0, 20.0, 45.0, 50.0, 45.0, 20.0, 7.0,
    ]
    payload = {
        "profile": [
            {"mass": round(100.0 + 0.001 * k, 6), "intensity": v}
            for k, v in enumerate(levels)
        ],
        "relative_threshold": 1.0,
        "min_separation": 0.005,
    }
    resp = client.post("/api/v1/profile-centroid", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["candidate_count"] == 2
    assert body["peak_count"] == 2
    masses = [p["mass"] for p in body["peaks"]]
    assert masses == sorted(masses)
    # First centroid near the apex index 7 (100.007), second near index 26.
    assert body["peaks"][0]["mass"] == pytest.approx(100.007, abs=0.001)
    assert body["peaks"][1]["mass"] == pytest.approx(100.026, abs=0.001)
    assert body["peaks"][1]["intensity"] == pytest.approx(50.0)
    assert sum(p["total_intensity_percent"] for p in body["peaks"]) == pytest.approx(100.0)


def test_api_large_finite_intensities_return_finite_200(client):
    # Regression: intensities near the float maximum overflowed during
    # normalisation/accumulation and made the response fail with NaN/Infinity.
    import math as _math

    payload = {
        "profile": [
            {"mass": 100.0, "intensity": 8e307},
            {"mass": 101.0, "intensity": 1e308},
            {"mass": 102.0, "intensity": 8e307},
        ],
        "relative_threshold": 5.0,
        "min_separation": 0.5,
    }
    resp = client.post("/api/v1/profile-centroid", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["candidate_count"] == 1
    assert body["peak_count"] == 1
    peak = body["peaks"][0]
    for key in ("mass", "intensity", "total_intensity_percent"):
        assert isinstance(peak[key], float)
        assert _math.isfinite(peak[key]), (key, peak[key])
    assert peak["mass"] == pytest.approx(101.0, abs=1e-6)
    assert peak["intensity"] == pytest.approx(100.0)
    assert peak["total_intensity_percent"] == pytest.approx(100.0)


def test_api_giant_mass_twin_plateaus_return_two_finite_peaks(client):
    # Regression: plateau endpoints near the float ceiling summed to infinity
    # in the midpoint, making two well-separated plateaus look coincident so
    # the response collapsed to a single peak carrying 100% of the intensity.
    import math as _math

    step = 1e306
    m0 = 1.0e308
    levels = [10.0, 10.0, 0.0, 5.0, 5.0]
    payload = {
        "profile": [
            {"mass": m0 + k * step, "intensity": value}
            for k, value in enumerate(levels)
        ],
        "relative_threshold": 0.0,
        "min_separation": 2.0 * step,
    }
    resp = client.post("/api/v1/profile-centroid", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["candidate_count"] == 2
    assert body["peak_count"] == 2
    for peak in body["peaks"]:
        for key in ("mass", "intensity", "total_intensity_percent"):
            assert _math.isfinite(peak[key]), (key, peak[key])
    left, right = body["peaks"]
    assert left["mass"] < right["mass"]
    assert left["intensity"] == pytest.approx(100.0)
    assert right["intensity"] == pytest.approx(50.0)
    assert left["total_intensity_percent"] == pytest.approx(100.0 * 20.0 / 30.0)
    assert right["total_intensity_percent"] == pytest.approx(100.0 * 10.0 / 30.0)
