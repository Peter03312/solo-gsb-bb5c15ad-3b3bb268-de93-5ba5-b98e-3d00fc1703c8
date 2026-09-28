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


def payload(**overrides):
    base = {"formula": "C6H12O6", "adduct": "+H", "charge": 1, "top_n": 10}
    base.update(overrides)
    return base


def test_ion_envelope_protonated_glucose(client):
    resp = client.post("/api/v1/ion-envelope", json=payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # neutral 180.063388 + H 1.007825, minus one electron, charge 1
    assert body["monoisotopic_mz"] == pytest.approx(181.070665, abs=1e-5)
    assert body["charge"] == 1
    assert body["adduct"] == "+H"
    assert body["composition"] == {"C": 6, "H": 13, "O": 6}
    assert body["labels"] == {}
    assert body["peak_count"] == 10
    assert len(body["peaks"]) == 10
    assert max(p["intensity"] for p in body["peaks"]) == pytest.approx(100.0)
    mzs = [p["mz"] for p in body["peaks"]]
    assert mzs == sorted(mzs)
    # peak key is m/z, not neutral mass
    assert set(body["peaks"][0]) == {"mz", "intensity"}


def test_ion_envelope_mz_spacing_for_doubly_charged(client):
    resp = client.post(
        "/api/v1/ion-envelope", json=payload(adduct="+2H", charge=2)
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # (180.063388 + 2 * 1.007825 - 2 * m_e) / 2
    assert body["monoisotopic_mz"] == pytest.approx(91.038971, abs=1e-5)
    assert body["composition"] == {"C": 6, "H": 14, "O": 6}
    # spacing from the monoisotopic peak to the first 13C satellite is ~1/2 Da
    mzs = [p["mz"] for p in body["peaks"]]
    assert mzs[1] - mzs[0] == pytest.approx(0.501677, abs=1e-4)


def test_ion_envelope_negative_charge_sign(client):
    body = client.post(
        "/api/v1/ion-envelope", json=payload(adduct="-H", charge=-1)
    ).json()
    # m/z of [M-H]- = neutral - H + electron
    assert body["monoisotopic_mz"] == pytest.approx(179.056112, abs=1e-5)
    assert body["composition"] == {"C": 6, "H": 11, "O": 6}


def test_sodium_adduct_minus_hydrogen(client):
    body = client.post(
        "/api/v1/ion-envelope", json=payload(adduct="+Na-H", charge=1)
    ).json()
    # 180.063388 + 22.989769 - 1.007825 - m_e
    assert body["monoisotopic_mz"] == pytest.approx(202.044784, abs=1e-5)
    assert body["composition"] == {"C": 6, "H": 11, "O": 6, "Na": 1}


def test_compound_adduct_coefficient_on_whole_fragment(client):
    body = client.post(
        "/api/v1/ion-envelope", json=payload(adduct="+2H-H2O", charge=1)
    ).json()
    assert body["composition"] == {"C": 6, "H": 12, "O": 5}


def test_adduct_with_isotope_label(client):
    body = client.post(
        "/api/v1/ion-envelope",
        json=payload(formula="C6H6", adduct="+[13C]2", charge=1),
    ).json()
    assert body["composition"] == {"C": 6, "H": 6}
    assert body["labels"] == {"[13C]": 2}


def test_negative_count_returns_422_with_fragment_position(client):
    resp = client.post(
        "/api/v1/ion-envelope", json=payload(adduct="+Na-H20")
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_adduct"
    # '+Na-' spans indices 0..3; the 'H20' fragment starts at 4
    assert detail["position"] == 4
    assert "negative" in detail["message"]


def test_negative_count_first_subtractive_fragment(client):
    resp = client.post(
        "/api/v1/ion-envelope",
        json=payload(formula="H2O", adduct="-3H", charge=-1),
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_adduct"
    assert detail["position"] == 2


def test_malformed_adduct_returns_422(client):
    for adduct, position in [
        ("H", 0),       # missing leading sign
        ("+", 1),       # empty fragment
        ("+H-", 3),     # trailing sign / empty fragment
        ("+Na-Xx", 4),  # unknown element in second fragment
        ("+0H", 1),     # zero coefficient
        ("+ H", 1),     # whitespace
    ]:
        resp = client.post("/api/v1/ion-envelope", json=payload(adduct=adduct))
        assert resp.status_code == 422, adduct
        detail = resp.json()["detail"]
        assert detail["error"] == "invalid_adduct"
        assert detail["position"] == position, adduct


def test_invalid_neutral_formula_still_invalid_formula(client):
    resp = client.post("/api/v1/ion-envelope", json=payload(formula="C6Xx2"))
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_formula"
    assert resp.json()["detail"]["position"] == 2


def test_charge_zero_rejected(client):
    resp = client.post("/api/v1/ion-envelope", json=payload(charge=0))
    assert resp.status_code == 422


@pytest.mark.parametrize("charge", [-10, -1, 1, 10])
def test_charge_bounds_accepted(client, charge):
    resp = client.post("/api/v1/ion-envelope", json=payload(charge=charge))
    assert resp.status_code == 200, resp.text


@pytest.mark.parametrize("charge", [-11, 11])
def test_charge_out_of_bounds_rejected(client, charge):
    resp = client.post("/api/v1/ion-envelope", json=payload(charge=charge))
    assert resp.status_code == 422


def test_missing_required_fields(client):
    resp = client.post("/api/v1/ion-envelope", json={"formula": "H2O"})
    assert resp.status_code == 422


def test_top_n_bounds(client):
    assert client.post(
        "/api/v1/ion-envelope", json=payload(top_n=0)
    ).status_code == 422
    assert client.post(
        "/api/v1/ion-envelope", json=payload(top_n=settings.top_n_max + 1)
    ).status_code == 422


def test_merge_tolerance_scales_with_abs_charge(client, monkeypatch):
    from app import main

    used = []
    original = main.isotope_envelope

    def spy(parsed, tolerance, prune):
        used.append(tolerance)
        return original(parsed, tolerance, prune)

    monkeypatch.setattr(main, "isotope_envelope", spy)
    resp = client.post(
        "/api/v1/ion-envelope", json=payload(charge=-3)
    )
    assert resp.status_code == 200
    assert used == [pytest.approx(3 * settings.merge_tolerance)]
