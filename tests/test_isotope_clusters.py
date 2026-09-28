"""Tests for de novo isotope-cluster detection and its API endpoint."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.isotope_clusters import NEUTRON_STEP_DA, detect_isotope_clusters


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def _cluster_payload(peaks, min_charge=1, max_charge=10, tol=0.005, min_peaks=3):
    return {
        "peaks": [{"mz": m, "intensity": i} for m, i in peaks],
        "min_charge": min_charge,
        "max_charge": max_charge,
        "spacing_tolerance": tol,
        "min_peaks": min_peaks,
    }


# --- unit-level detection ---------------------------------------------------

def test_detect_single_charge_cluster():
    step = NEUTRON_STEP_DA
    peaks = [
        (100.0, 100.0),
        (100.0 + step, 60.0),
        (100.0 + 2 * step, 10.0),
        (100.0 + 3 * step, 2.0),
    ]
    result = detect_isotope_clusters(peaks, (1, 3), 0.005, 3)
    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert cluster.charge == 1
    assert cluster.member_indices == (0, 1, 2, 3)
    assert cluster.missing_positions == ()
    assert cluster.monoisotopic_mz == 100.0
    assert cluster.score == pytest.approx(172.0)
    assert result.unassigned_indices == ()


def test_detect_charge_two_spacing():
    step = NEUTRON_STEP_DA / 2
    peaks = [
        (200.0, 50.0),
        (200.0 + step, 25.0),
        (200.0 + 2 * step, 10.0),
    ]
    result = detect_isotope_clusters(peaks, (1, 3), 0.005, 3)
    assert len(result.clusters) == 1
    assert result.clusters[0].charge == 2
    # Base-peak normalised: 100 + 50 + 20 = 170.
    assert result.clusters[0].score == pytest.approx(170.0)


def test_detect_missing_one_isotope_penalty():
    step = NEUTRON_STEP_DA
    # Nominal positions 0, 2, 3 -> position 1 missing, one two-step gap.
    peaks = [
        (300.0, 100.0),
        (300.0 + 2 * step, 30.0),
        (300.0 + 3 * step, 5.0),
    ]
    result = detect_isotope_clusters(peaks, (1, 1), 0.005, 3)
    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert cluster.member_indices == (0, 1, 2)
    assert cluster.missing_positions == (1,)
    assert cluster.score == pytest.approx(100.0 + 30.0 + 5.0 - 10.0)


def test_at_most_one_two_step_gap():
    step = NEUTRON_STEP_DA
    # Gaps of two, one and two nominal steps again; the chain as a whole
    # must not be reported, only its three-member sub-chains (one gap each).
    peaks = [
        (100.0, 100.0),
        (100.0 + 2 * step, 50.0),
        (100.0 + 3 * step, 10.0),
        (100.0 + 5 * step, 5.0),
    ]
    result = detect_isotope_clusters(peaks, (1, 1), 0.005, 3)
    for cluster in result.clusters:
        assert len(cluster.missing_positions) <= 1
        assert len(cluster.member_indices) == 3


def test_no_qualifying_candidate_returns_empty():
    peaks = [(100.0, 1.0), (200.0, 2.0), (300.0, 3.0)]
    result = detect_isotope_clusters(peaks, (1, 3), 0.001, 3)
    assert result.clusters == ()
    assert result.unassigned_indices == (0, 1, 2)


def test_unassigned_peak_between_two_clusters():
    step = NEUTRON_STEP_DA
    peaks = [
        (100.0, 100.0),
        (100.0 + step, 60.0),
        (100.0 + 2 * step, 10.0),
        (150.0, 1.0),
        (200.0, 80.0),
        (200.0 + step, 40.0),
        (200.0 + 2 * step, 5.0),
    ]
    result = detect_isotope_clusters(peaks, (1, 1), 0.005, 3)
    assert [c.member_indices for c in result.clusters] == [(0, 1, 2), (4, 5, 6)]
    assert result.unassigned_indices == (3,)


def test_spacing_outside_tolerance_rejected():
    step = NEUTRON_STEP_DA
    peaks = [
        (100.0, 100.0),
        (100.0 + step + 0.006, 50.0),
        (100.0 + 2 * step + 0.012, 10.0),
    ]
    result = detect_isotope_clusters(peaks, (1, 1), 0.005, 3)
    assert result.clusters == ()


def test_min_peaks_filters_short_chains():
    step = NEUTRON_STEP_DA
    peaks = [(100.0, 100.0), (100.0 + step, 50.0)]
    assert detect_isotope_clusters(peaks, (1, 1), 0.005, 3).clusters == ()
    result = detect_isotope_clusters(peaks, (1, 1), 0.005, 2)
    assert len(result.clusters) == 1
    assert result.clusters[0].member_indices == (0, 1)


def test_overlapping_candidates_choose_higher_score():
    step = NEUTRON_STEP_DA
    # Strong chain 0,1,2 (300); peaks 3,4 cannot attach to index 2 within
    # tolerance, so the only qualifying cluster is the strong start.
    peaks = [
        (100.0, 100.0),
        (100.0 + step, 100.0),
        (100.0 + 2 * step, 100.0),
        (100.0 + 3 * step + 0.003, 10.0),
        (100.0 + 4 * step + 0.003, 10.0),
    ]
    result = detect_isotope_clusters(peaks, (1, 1), 0.001, 3)
    assert [c.member_indices for c in result.clusters] == [(0, 1, 2)]
    assert result.unassigned_indices == (3, 4)


def test_all_zero_intensities_allowed():
    step = NEUTRON_STEP_DA
    peaks = [
        (100.0, 0.0),
        (100.0 + step, 0.0),
        (100.0 + 2 * step, 0.0),
    ]
    result = detect_isotope_clusters(peaks, (1, 1), 0.005, 3)
    assert len(result.clusters) == 1
    assert result.clusters[0].score == pytest.approx(0.0)


# --- API endpoint -----------------------------------------------------------

def test_endpoint_single_cluster(client):
    step = NEUTRON_STEP_DA
    payload = _cluster_payload(
        [
            (100.0, 100.0),
            (100.0 + step, 60.0),
            (100.0 + 2 * step, 10.0),
            (100.0 + 3 * step, 2.0),
            (150.0, 1.0),
        ]
    )
    resp = client.post("/api/v1/isotope-clusters", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["peak_count"] == 5
    assert body["cluster_count"] == 1
    cluster = body["clusters"][0]
    assert cluster["charge"] == 1
    assert cluster["member_indices"] == [0, 1, 2, 3]
    assert cluster["missing_positions"] == []
    assert cluster["score"] == pytest.approx(172.0)
    assert body["unassigned_peaks"] == [4]


def test_endpoint_empty_clusters(client):
    payload = _cluster_payload(
        [(100.0, 100.0), (200.0, 50.0), (300.0, 10.0)], tol=0.001
    )
    resp = client.post("/api/v1/isotope-clusters", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["clusters"] == []
    assert body["cluster_count"] == 0
    assert body["unassigned_peaks"] == [0, 1, 2]


def test_endpoint_two_clusters_with_missing_position(client):
    step = NEUTRON_STEP_DA
    payload = _cluster_payload(
        [
            (100.0, 100.0),
            (100.0 + step, 60.0),
            (100.0 + 2 * step, 30.0),
            (200.0, 100.0),
            (200.0 + 2 * step, 40.0),
            (200.0 + 3 * step, 5.0),
        ]
    )
    resp = client.post("/api/v1/isotope-clusters", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["cluster_count"] == 2
    first, second = body["clusters"]
    assert first["charge"] == 1
    assert first["member_indices"] == [0, 1, 2]
    assert second["member_indices"] == [3, 4, 5]
    assert second["missing_positions"] == [1]
    assert second["score"] == pytest.approx(100.0 + 40.0 + 5.0 - 10.0)
    assert body["unassigned_peaks"] == []


def test_endpoint_non_increasing_mz_422(client):
    payload = _cluster_payload(
        [(101.0, 100.0), (100.0, 50.0), (102.0, 10.0)]
    )
    resp = client.post("/api/v1/isotope-clusters", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_peaks"
    assert detail["index"] == 1


def test_endpoint_duplicate_mz_422(client):
    payload = _cluster_payload(
        [(100.0, 100.0), (100.0, 50.0), (101.0, 10.0)]
    )
    resp = client.post("/api/v1/isotope-clusters", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_peaks"
    assert detail["index"] == 1


def test_endpoint_negative_intensity_422(client):
    payload = _cluster_payload(
        [(100.0, 100.0), (101.0, -1.0), (102.0, 10.0)]
    )
    resp = client.post("/api/v1/isotope-clusters", json=payload)
    assert resp.status_code == 422


def test_endpoint_non_finite_values_422(client):
    # httpx refuses to emit NaN/Infinity itself, so send raw JSON text; the
    # service parses these Python-flavoured literals and must reject them.
    cases = [
        '{"peaks":[{"mz":100.0,"intensity":1.0},{"mz":101.0,"intensity":1.0},{"mz":NaN,"intensity":1.0}],"spacing_tolerance":0.005}',
        '{"peaks":[{"mz":100.0,"intensity":1.0},{"mz":Infinity,"intensity":1.0},{"mz":102.0,"intensity":1.0}],"spacing_tolerance":0.005}',
        '{"peaks":[{"mz":100.0,"intensity":1.0},{"mz":101.0,"intensity":-Infinity},{"mz":102.0,"intensity":1.0}],"spacing_tolerance":0.005}',
        '{"peaks":[{"mz":100.0,"intensity":Infinity},{"mz":101.0,"intensity":1.0},{"mz":102.0,"intensity":1.0}],"spacing_tolerance":0.005}',
        '{"peaks":[{"mz":100.0,"intensity":NaN},{"mz":101.0,"intensity":1.0},{"mz":102.0,"intensity":1.0}],"spacing_tolerance":0.005}',
        '{"peaks":[{"mz":100.0,"intensity":1.0},{"mz":101.0,"intensity":1.0}],"spacing_tolerance":Infinity}',
        '{"peaks":[{"mz":100.0,"intensity":1.0},{"mz":101.0,"intensity":1.0}],"spacing_tolerance":NaN}',
    ]
    for raw in cases:
        body = raw + ',"min_charge":1,"max_charge":10,"min_peaks":2}'
        resp = client.post(
            "/api/v1/isotope-clusters",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, raw


def test_endpoint_parameter_bounds(client):
    base = _cluster_payload([(100.0, 1.0), (101.0, 2.0), (102.0, 3.0)])
    for overrides in (
        {"min_charge": 0},
        {"min_charge": 11},
        {"max_charge": 0},
        {"max_charge": 11},
        {"min_charge": 5, "max_charge": 2},
        {"spacing_tolerance": 0.0},
        {"spacing_tolerance": -0.1},
        {"min_peaks": 1},
        {"min_peaks": 0},
        {"min_peaks": 201},
    ):
        payload = {**base, **overrides}
        resp = client.post("/api/v1/isotope-clusters", json=payload)
        assert resp.status_code == 422, overrides


def test_endpoint_peak_count_bounds(client):
    empty = _cluster_payload([])
    assert client.post("/api/v1/isotope-clusters", json=empty).status_code == 422
    too_many = _cluster_payload([(float(i), 1.0) for i in range(201)])
    assert client.post("/api/v1/isotope-clusters", json=too_many).status_code == 422


def test_endpoint_all_zero_intensity_accepted(client):
    step = NEUTRON_STEP_DA
    payload = _cluster_payload(
        [(100.0, 0.0), (100.0 + step, 0.0), (100.0 + 2 * step, 0.0)]
    )
    resp = client.post("/api/v1/isotope-clusters", json=payload)
    assert resp.status_code == 200
    assert resp.json()["cluster_count"] == 1


def test_endpoint_charge_tie_prefers_lower_charge(client):
    # min_peaks = 2: two peaks exactly one neutron step apart are valid for
    # charge 1; with a loose tolerance no other charge fits that spacing,
    # so the reported charge is 1 (the deterministic smaller one).
    payload = _cluster_payload(
        [(100.0, 100.0), (100.0 + NEUTRON_STEP_DA, 50.0)], min_peaks=2
    )
    resp = client.post("/api/v1/isotope-clusters", json=payload)
    assert resp.status_code == 200
    assert resp.json()["clusters"][0]["charge"] == 1


def test_endpoint_charge_range_limit(client):
    step2 = NEUTRON_STEP_DA / 2
    payload = _cluster_payload(
        [
            (200.0, 100.0),
            (200.0 + step2, 50.0),
            (200.0 + 2 * step2, 10.0),
        ],
        min_charge=1,
        max_charge=1,
    )
    resp = client.post("/api/v1/isotope-clusters", json=payload)
    assert resp.status_code == 200
    assert resp.json()["clusters"] == []

    payload["max_charge"] = 2
    resp = client.post("/api/v1/isotope-clusters", json=payload)
    assert resp.status_code == 200
    assert resp.json()["clusters"][0]["charge"] == 2
