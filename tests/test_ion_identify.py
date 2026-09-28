import pytest
from fastapi.testclient import TestClient

from app.config import settings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Isolate each test run with its own SQLite file.
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def _ion_peaks(client, formula, adduct, charge, top_n=10):
    resp = client.post(
        "/api/v1/ion-envelope",
        json={
            "formula": formula,
            "adduct": adduct,
            "charge": charge,
            "top_n": top_n,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    return body["monoisotopic_mz"], [
        {"mz": p["mz"], "intensity": p["intensity"]} for p in body["peaks"]
    ]


def _payload(**overrides):
    payload = {
        "formula": "C6H12O6",
        "peaks": [{"mz": 181.07, "intensity": 100.0}],
        "mass_tolerance": 0.01,
        "candidates": [{"adduct": "+H", "charge": 1}],
    }
    payload.update(overrides)
    return payload


def test_ion_identify_picks_matching_adduct(client: TestClient):
    _, peaks = _ion_peaks(client, "C6H12O6", "+H", 1, top_n=10)
    resp = client.post(
        "/api/v1/ion-identify",
        json=_payload(
            peaks=peaks,
            candidates=[
                {"adduct": "+Na-H", "charge": 1},
                {"adduct": "+H", "charge": 1},
                {"adduct": "+2H", "charge": 2},
            ],
        ),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["observed_peak_count"] == len(peaks)
    assert body["mass_tolerance"] == 0.01

    # Best candidate is +H at request index 1, with a (near-)perfect score
    # (small residual from the top_n truncation/rounding of observed peaks).
    best = body["best_candidate"]
    assert best["index"] == 1
    assert best["adduct"] == "+H"
    assert best["charge"] == 1
    assert best["mse"] == pytest.approx(0.0, abs=1e-3)

    # Candidates are sorted by ascending MSE.
    mses = [c["mse"] for c in body["candidates"]]
    assert mses == sorted(mses)
    winner = body["candidates"][0]
    assert winner["index"] == 1
    assert winner["valid"] is True
    assert winner["adduct"] == "+H"
    assert winner["charge"] == 1
    assert winner["monoisotopic_mz"] == pytest.approx(181.070665, abs=1e-5)
    assert winner["composition"] == {"C": 6, "H": 13, "O": 6}
    assert winner["labels"] == {}
    assert winner["pair_count"] == len(winner["pairs"])
    assert winner["error"] is None

    # Every observed peak participates in exactly one serialised pair.
    observed_mzs = [
        p["observed_mz"] for p in winner["pairs"] if p["observed_mz"] is not None
    ]
    assert len(observed_mzs) == len(peaks)
    # Pair m/z fields (not neutral-mass ones).
    assert set(winner["pairs"][0]) == {
        "observed_mz",
        "observed_intensity",
        "predicted_mz",
        "predicted_intensity",
    }
    # Unmatched predicted peaks null out the observed side.
    unmatched = [p for p in winner["pairs"] if p["observed_mz"] is None]
    assert unmatched and unmatched[0]["predicted_intensity"] > 0


def test_ion_identify_stable_order_on_tie(client: TestClient):
    # Identical adduct/charge specifications predict identical envelopes and
    # therefore tie exactly on MSE; ties must keep the request order rather
    # than reshuffle.
    resp = client.post(
        "/api/v1/ion-identify",
        json=_payload(
            candidates=[
                {"adduct": "+H", "charge": 1},
                {"adduct": "+H", "charge": 1},
                {"adduct": "+Na-H", "charge": 1},
                {"adduct": "+H", "charge": 1},
            ]
        ),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    mses = [c["mse"] for c in body["candidates"]]
    assert mses == sorted(mses)
    # The three +H entries tie and must remain in request-index order.
    tied = [c["index"] for c in body["candidates"] if c["adduct"] == "+H"]
    assert tied == [0, 1, 3]
    # Lowest index wins a tie.
    assert body["best_candidate"]["index"] == 0
    assert body["best_candidate"]["adduct"] == "+H"


def test_ion_identify_doubly_charged_uses_charge_scaled_envelope(client: TestClient):
    mono_mz, peaks = _ion_peaks(client, "C6H12O6", "+2H", 2, top_n=10)
    resp = client.post(
        "/api/v1/ion-identify",
        json=_payload(
            peaks=peaks,
            candidates=[
                {"adduct": "+2H", "charge": 2},
                {"adduct": "+H", "charge": 1},
            ],
        ),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["best_candidate"]["index"] == 0
    winner = body["candidates"][0]
    assert winner["monoisotopic_mz"] == pytest.approx(mono_mz, abs=1e-5)
    assert winner["composition"] == {"C": 6, "H": 14, "O": 6}


def test_ion_identify_invalid_candidates_report_index_and_position(client: TestClient):
    resp = client.post(
        "/api/v1/ion-identify",
        json=_payload(
            candidates=[
                {"adduct": "+H", "charge": 1},        # valid, index 0
                {"adduct": "H", "charge": 1},         # missing leading sign, index 1
                {"adduct": "-99H", "charge": -1},     # negative H count, index 2
                {"adduct": "+ H", "charge": 1},       # whitespace, index 3
            ]
        ),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # The single valid candidate wins regardless of invalid entries.
    assert body["best_candidate"] == {
        "index": 0,
        "adduct": "+H",
        "charge": 1,
        "mse": body["best_candidate"]["mse"],
    }

    by_index = {c["index"]: c for c in body["candidates"]}
    assert by_index[0]["valid"] is True

    bad = by_index[1]
    assert bad["valid"] is False
    assert bad["mse"] is None
    assert bad["error"] == "invalid_adduct"
    assert bad["position"] == 0
    assert "+" in bad["message"] or "-" in bad["message"]

    # '-99H': the fragment starts after the full coefficient run.
    bad = by_index[2]
    assert bad["valid"] is False
    assert bad["error"] == "invalid_adduct"
    assert bad["position"] == 3
    assert "negative" in bad["message"]

    bad = by_index[3]
    assert bad["valid"] is False
    assert bad["error"] == "invalid_adduct"
    assert bad["position"] == 1
    assert "whitespace" in bad["message"]

    # Invalid candidates follow scored ones; invalid entries keep request order.
    tail = body["candidates"][-3:]
    assert [c["index"] for c in tail] == [1, 2, 3]
    assert all(c["valid"] is False for c in tail)


def test_ion_identify_all_invalid_candidates(client: TestClient):
    resp = client.post(
        "/api/v1/ion-identify",
        json=_payload(
            candidates=[
                {"adduct": "H", "charge": 1},
                {"adduct": "-99H", "charge": -1},
            ]
        ),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["best_candidate"] is None
    assert [c["index"] for c in body["candidates"]] == [0, 1]
    assert all(c["valid"] is False for c in body["candidates"])


def test_ion_identify_unmatched_observed_peak_scores_against_zero(client: TestClient):
    # Two observed peaks; the candidate envelope only covers one of them
    # under a tight tolerance, so the other must score at predicted intensity 0.
    mono_mz, _ = _ion_peaks(client, "C6H12O6", "+H", 1, top_n=3)
    peaks = [
        {"mz": mono_mz, "intensity": 100.0},
        {"mz": mono_mz + 5.0, "intensity": 40.0},
    ]
    resp = client.post(
        "/api/v1/ion-identify",
        json=_payload(peaks=peaks, mass_tolerance=0.01),
    )
    assert resp.status_code == 200, resp.text
    winner = resp.json()["candidates"][0]
    # (0^2 + 40^2) / number of pairs >= 1600/pairs; just check it is clearly
    # positive and the unmatched observed pair nulls the predicted side.
    assert winner["mse"] > 1.0
    unmatched = [p for p in winner["pairs"] if p["predicted_mz"] is None]
    assert len(unmatched) == 1
    assert unmatched[0]["observed_intensity"] == 40.0
    assert unmatched[0]["predicted_intensity"] is None


def test_ion_identify_invalid_formula_422(client: TestClient):
    resp = client.post("/api/v1/ion-identify", json=_payload(formula="C6Xx2"))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_formula"
    assert detail["position"] == 2


def test_ion_identify_peak_validation_422(client: TestClient):
    # empty
    resp = client.post("/api/v1/ion-identify", json=_payload(peaks=[]))
    assert resp.status_code == 422

    # not sorted by increasing m/z
    resp = client.post(
        "/api/v1/ion-identify",
        json=_payload(
            peaks=[
                {"mz": 182.0, "intensity": 1.0},
                {"mz": 181.0, "intensity": 1.0},
            ]
        ),
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_peaks"
    assert detail["index"] == 1

    # duplicate m/z
    resp = client.post(
        "/api/v1/ion-identify",
        json=_payload(
            peaks=[
                {"mz": 181.0, "intensity": 100.0},
                {"mz": 181.0, "intensity": 5.0},
            ]
        ),
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_peaks"
    assert detail["index"] == 1

    # all-zero intensities
    resp = client.post(
        "/api/v1/ion-identify",
        json=_payload(peaks=[{"mz": 181.0, "intensity": 0.0}]),
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_peaks"


def test_ion_identify_parameter_bounds_422(client: TestClient):
    # charge out of range / zero is rejected at request validation.
    for charge in (0, 11, -11):
        resp = client.post(
            "/api/v1/ion-identify",
            json=_payload(candidates=[{"adduct": "+H", "charge": charge}]),
        )
        assert resp.status_code == 422, charge

    # non-positive tolerance
    assert (
        client.post(
            "/api/v1/ion-identify", json=_payload(mass_tolerance=0.0)
        ).status_code
        == 422
    )

    # candidate count 1..20
    assert (
        client.post(
            "/api/v1/ion-identify", json=_payload(candidates=[])
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/ion-identify",
            json=_payload(
                candidates=[{"adduct": "+H", "charge": 1}] * 21
            ),
        ).status_code
        == 422
    )

    # peak count 1..200
    too_many = [{"mz": float(i), "intensity": 1.0} for i in range(201)]
    assert (
        client.post(
            "/api/v1/ion-identify", json=_payload(peaks=too_many)
        ).status_code
        == 422
    )

    # negative intensity
    assert (
        client.post(
            "/api/v1/ion-identify",
            json=_payload(peaks=[{"mz": 181.0, "intensity": -1.0}]),
        ).status_code
        == 422
    )
