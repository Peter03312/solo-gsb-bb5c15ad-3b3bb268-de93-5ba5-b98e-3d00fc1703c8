import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.identify import predict_ion_candidate
from app.mixture import mix_envelopes
from app.parser import parse_formula


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Isolate each test run with its own SQLite file.
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def _mixed_peaks(formula, first, second, first_fraction, tolerance=0.01):
    """Observed-style peaks of a two-ion mixture, built with the app's own
    envelope/mixing machinery so an exact grid point must score MSE 0."""
    parsed = parse_formula(formula)
    pred_first = predict_ion_candidate(
        parsed, first[0], first[1], settings.merge_tolerance, settings.prune_threshold
    )
    pred_second = predict_ion_candidate(
        parsed, second[0], second[1], settings.merge_tolerance, settings.prune_threshold
    )
    return [
        {"mz": m, "intensity": i}
        for m, i in mix_envelopes(
            pred_first.peaks, pred_second.peaks, first_fraction, tolerance
        )
    ]


def _payload(**overrides):
    payload = {
        "formula": "C2H6O",
        "peaks": [{"mz": 47.049, "intensity": 100.0}],
        "mass_tolerance": 0.01,
        "candidates": [
            {"adduct": "+H", "charge": 1},
            {"adduct": "+Na-H", "charge": 1},
        ],
        "step_percent": 1.0,
    }
    payload.update(overrides)
    return payload


def test_ion_mixture_fit_recovers_ratio(client: TestClient):
    peaks = _mixed_peaks("C2H6O", ("+H", 1), ("+Na-H", 1), 0.6)
    resp = client.post("/api/v1/ion-mixture-fit", json=_payload(peaks=peaks))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["formula"] == "C2H6O"
    assert body["mass_tolerance"] == 0.01
    assert body["step_percent"] == 1.0
    assert body["observed_peak_count"] == len(peaks)
    assert body["valid_candidate_count"] == 2
    # One candidate pair x 101 ratios (0..100 at step 1.0).
    assert body["combination_count"] == 101
    assert len(body["combinations"]) == 101

    best = body["best_combination"]
    assert best["first_index"] == 0
    assert best["second_index"] == 1
    assert best["first_adduct"] == "+H"
    assert best["first_charge"] == 1
    assert best["second_adduct"] == "+Na-H"
    assert best["second_charge"] == 1
    assert best["first_fraction_percent"] == pytest.approx(60.0)
    assert best["second_fraction_percent"] == pytest.approx(40.0)
    assert best["mse"] == pytest.approx(0.0, abs=1e-9)
    assert best["pair_count"] == len(best["pairs"])
    # Every observed peak participates in exactly one serialised pair.
    seen = [p["observed_mz"] for p in best["pairs"] if p["observed_mz"] is not None]
    assert len(seen) == len(peaks)
    assert set(best["pairs"][0]) == {
        "observed_mz",
        "observed_intensity",
        "predicted_mz",
        "predicted_intensity",
    }

    # Combinations are ranked by ascending MSE; rank 1 is the best one.
    mses = [c["mse"] for c in body["combinations"]]
    assert mses == sorted(mses)
    top = body["combinations"][0]
    assert top["rank"] == 1
    assert (top["first_index"], top["second_index"]) == (0, 1)
    assert top["first_fraction_percent"] == pytest.approx(60.0)
    assert top["second_fraction_percent"] == pytest.approx(40.0)
    assert top["mse"] == best["mse"]
    assert [c["rank"] for c in body["combinations"]] == list(range(1, 102))


def test_ion_mixture_fit_tie_breaks_to_index_pair_then_ratio(client: TestClient):
    # Single-isotope Na: the "+Na" envelope is one exact (mz, 100.0) peak,
    # so mixing the two identical "+Na" candidates at ratio 0 or 100 (one
    # side skipped, no scaling arithmetic) reproduces the observed peak
    # bit-exactly, as does the pure-"+Na" endpoint of the other pairs.
    # Those exact MSE-0 ties must resolve to the smallest index pair and
    # then the lowest first-candidate ratio.
    mono_mz = predict_ion_candidate(
        parse_formula("Na"), "+Na", 1, settings.merge_tolerance, settings.prune_threshold
    ).peaks[0][0]
    resp = client.post(
        "/api/v1/ion-mixture-fit",
        json=_payload(
            formula="Na",
            peaks=[{"mz": mono_mz, "intensity": 100.0}],
            candidates=[
                {"adduct": "+Na", "charge": 1},
                {"adduct": "+Na", "charge": 1},
                {"adduct": "+H", "charge": 1},
            ],
        ),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid_candidate_count"] == 3
    # Three pairs x 101 ratios.
    assert body["combination_count"] == 3 * 101

    best = body["best_combination"]
    assert (best["first_index"], best["second_index"]) == (0, 1)
    assert best["first_fraction_percent"] == 0.0
    assert best["second_fraction_percent"] == 100.0
    assert best["mse"] == 0.0

    combos = body["combinations"]
    assert (combos[0]["first_index"], combos[0]["second_index"]) == (0, 1)
    assert combos[0]["first_fraction_percent"] == 0.0
    assert combos[0]["mse"] == 0.0

    # The pure-"+Na" endpoints of every pair are exact MSE-0 ties...
    zero = [c for c in combos if c["mse"] == 0.0]
    keys = {
        (c["first_index"], c["second_index"], c["first_fraction_percent"])
        for c in zero
    }
    assert {(0, 1, 0.0), (0, 1, 100.0), (0, 2, 100.0), (1, 2, 100.0)} <= keys
    # ...and within the tie the ranking follows (index pair, ratio) order.
    zero_keys = [
        (c["first_index"], c["second_index"], c["first_fraction_percent"])
        for c in zero
    ]
    assert zero_keys == sorted(zero_keys)


def test_ion_mixture_fit_invalid_candidates_skip_combinations(client: TestClient):
    peaks = _mixed_peaks("C2H6O", ("+H", 1), ("+Na-H", 1), 0.25)
    resp = client.post(
        "/api/v1/ion-mixture-fit",
        json=_payload(
            peaks=peaks,
            candidates=[
                {"adduct": "+H", "charge": 1},      # valid, index 0
                {"adduct": "-99H", "charge": -1},   # negative H count, index 1
                {"adduct": "+Na-H", "charge": 1},   # valid, index 2
                {"adduct": "+ H", "charge": 1},     # whitespace, index 3
            ],
        ),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid_candidate_count"] == 2
    # Only the (0, 2) pair survives; every combination of the invalid
    # candidates is skipped.
    assert body["combination_count"] == 101
    assert {(c["first_index"], c["second_index"]) for c in body["combinations"]} == {(0, 2)}

    best = body["best_combination"]
    assert (best["first_index"], best["second_index"]) == (0, 2)
    assert best["first_fraction_percent"] == pytest.approx(25.0)
    assert best["second_fraction_percent"] == pytest.approx(75.0)
    assert best["mse"] == pytest.approx(0.0, abs=1e-9)

    # Candidates stay in request order; invalid ones carry the same
    # position-aware adduct error as /api/v1/ion-identify.
    by_index = {c["index"]: c for c in body["candidates"]}
    assert by_index[0]["valid"] is True
    assert by_index[0]["monoisotopic_mz"] == pytest.approx(47.049141, abs=1e-5)
    assert by_index[0]["error"] is None
    assert by_index[2]["valid"] is True

    bad = by_index[1]
    assert bad["valid"] is False
    assert bad["monoisotopic_mz"] is None
    assert bad["error"] == "invalid_adduct"
    assert bad["position"] == 3
    assert "negative" in bad["message"]

    bad = by_index[3]
    assert bad["valid"] is False
    assert bad["error"] == "invalid_adduct"
    assert bad["position"] == 1
    assert "whitespace" in bad["message"]


def test_ion_mixture_fit_fewer_than_two_valid_candidates(client: TestClient):
    # One valid candidate: no pair can be formed.
    resp = client.post(
        "/api/v1/ion-mixture-fit",
        json=_payload(
            candidates=[
                {"adduct": "+H", "charge": 1},
                {"adduct": "-99H", "charge": -1},
            ]
        ),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid_candidate_count"] == 1
    assert body["combination_count"] == 0
    assert body["combinations"] == []
    assert body["best_combination"] is None

    # All candidates invalid.
    resp = client.post(
        "/api/v1/ion-mixture-fit",
        json=_payload(
            candidates=[
                {"adduct": "H", "charge": 1},
                {"adduct": "-99H", "charge": -1},
            ]
        ),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["valid_candidate_count"] == 0
    assert body["combination_count"] == 0
    assert body["combinations"] == []
    assert body["best_combination"] is None
    assert all(c["valid"] is False for c in body["candidates"])


def test_ion_mixture_fit_step_percent_grid(client: TestClient):
    peaks = _mixed_peaks("C2H6O", ("+H", 1), ("+Na-H", 1), 0.3)
    resp = client.post(
        "/api/v1/ion-mixture-fit",
        json=_payload(peaks=peaks, step_percent=10.0),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["combination_count"] == 11
    fractions = sorted(c["first_fraction_percent"] for c in body["combinations"])
    assert fractions == [float(i) for i in range(0, 101, 10)]
    best = body["best_combination"]
    assert best["first_fraction_percent"] == pytest.approx(30.0)
    assert best["mse"] == pytest.approx(0.0, abs=1e-9)


def test_ion_mixture_fit_mixed_charge_states(client: TestClient):
    # A z=2 component mixed with a z=1 component at 50/50.
    peaks = _mixed_peaks("C2H6O", ("+2H", 2), ("+H", 1), 0.5)
    resp = client.post(
        "/api/v1/ion-mixture-fit",
        json=_payload(
            peaks=peaks,
            candidates=[
                {"adduct": "+H", "charge": 1},
                {"adduct": "+2H", "charge": 2},
            ],
        ),
    )
    assert resp.status_code == 200, resp.text
    best = resp.json()["best_combination"]
    assert (best["first_index"], best["second_index"]) == (0, 1)
    assert best["first_fraction_percent"] == pytest.approx(50.0)
    assert best["mse"] == pytest.approx(0.0, abs=1e-9)


def test_ion_mixture_fit_unmatched_peaks_count_on_both_sides(client: TestClient):
    # Observed = pure +H envelope plus one far-away extra peak; the best
    # mixture must score the unmatched observed peak and the unmatched
    # predicted peaks at zero intensity, so the MSE stays clearly positive.
    peaks = _mixed_peaks("C2H6O", ("+H", 1), ("+H", 1), 1.0)
    peaks.append({"mz": peaks[-1]["mz"] + 50.0, "intensity": 40.0})
    resp = client.post("/api/v1/ion-mixture-fit", json=_payload(peaks=peaks))
    assert resp.status_code == 200, resp.text
    best = resp.json()["best_combination"]
    assert best["mse"] > 1.0
    unmatched_obs = [p for p in best["pairs"] if p["predicted_mz"] is None]
    assert len(unmatched_obs) == 1
    assert unmatched_obs[0]["observed_intensity"] == 40.0
    unmatched_pred = [p for p in best["pairs"] if p["observed_mz"] is None]
    assert unmatched_pred  # predicted peaks beyond the observed range


def test_ion_mixture_fit_request_validation_422(client: TestClient):
    # candidate count 2..10
    assert (
        client.post(
            "/api/v1/ion-mixture-fit",
            json=_payload(candidates=[{"adduct": "+H", "charge": 1}]),
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/ion-mixture-fit",
            json=_payload(candidates=[{"adduct": "+H", "charge": 1}] * 11),
        ).status_code
        == 422
    )

    # step_percent bounds
    assert (
        client.post(
            "/api/v1/ion-mixture-fit", json=_payload(step_percent=0.4)
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/ion-mixture-fit", json=_payload(step_percent=10.5)
        ).status_code
        == 422
    )

    # non-positive tolerance
    assert (
        client.post(
            "/api/v1/ion-mixture-fit", json=_payload(mass_tolerance=0.0)
        ).status_code
        == 422
    )

    # charge zero / out of range
    assert (
        client.post(
            "/api/v1/ion-mixture-fit",
            json=_payload(
                candidates=[
                    {"adduct": "+H", "charge": 0},
                    {"adduct": "+H", "charge": 1},
                ]
            ),
        ).status_code
        == 422
    )

    # invalid formula keeps the position-aware 422
    resp = client.post("/api/v1/ion-mixture-fit", json=_payload(formula="C6Xx2"))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_formula"
    assert detail["position"] == 2

    # peak-list validation: empty / not increasing / all-zero intensities
    assert (
        client.post("/api/v1/ion-mixture-fit", json=_payload(peaks=[])).status_code
        == 422
    )
    resp = client.post(
        "/api/v1/ion-mixture-fit",
        json=_payload(
            peaks=[
                {"mz": 48.0, "intensity": 1.0},
                {"mz": 47.0, "intensity": 1.0},
            ]
        ),
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_peaks"
    resp = client.post(
        "/api/v1/ion-mixture-fit",
        json=_payload(peaks=[{"mz": 47.0, "intensity": 0.0}]),
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_peaks"
