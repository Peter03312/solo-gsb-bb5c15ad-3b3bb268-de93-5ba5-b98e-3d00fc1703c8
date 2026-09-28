import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.enrichment import enriched_distribution
from app.envelope import convolve, isotope_envelope, merge_peaks, power_polynomial, top_peaks
from app.isotopes import ISOTOPES
from app.parser import parse_formula


def synthetic_peaks(formula, symbol, mass_number, fraction, top=10):
    """Observed-style peaks with ``symbol``'s isotope ``mass_number``
    enriched to ``fraction`` (all other elements natural)."""
    parsed = parse_formula(formula)
    dist = enriched_distribution(symbol, mass_number, fraction)
    env = [(0.0, 1.0)]
    for sym, count in parsed.composition.items():
        base = list(dist) if sym == symbol else list(ISOTOPES[sym])
        env = convolve(env, power_polynomial(base, count, 0.0001, 1e-12), 0.0001, 1e-12)
    return top_peaks(merge_peaks(env, 0.0001), top)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Isolate each test run with its own SQLite file.
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_envelope_glucose(client):
    resp = client.post("/api/v1/envelope", json={"formula": "C6H12O6", "top_n": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["monoisotopic_mass"] == pytest.approx(180.063388, abs=1e-5)
    assert body["peak_count"] == 10
    assert len(body["peaks"]) == 10
    assert max(p["intensity"] for p in body["peaks"]) == pytest.approx(100.0)
    assert body["cached"] is False
    masses = [p["mass"] for p in body["peaks"]]
    assert masses == sorted(masses)


def test_envelope_with_label(client):
    resp = client.post("/api/v1/envelope", json={"formula": "C6H6[13C]2", "top_n": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["labels"] == {"[13C]": 2}
    assert body["monoisotopic_mass"] == pytest.approx(
        6 * 12.0 + 6 * 1.00782503223 + 2 * 13.00335483507, abs=1e-5
    )


def test_envelope_cached_on_second_call(client):
    payload = {"formula": "C2H6O", "top_n": 5}
    first = client.post("/api/v1/envelope", json=payload).json()
    second = client.post("/api/v1/envelope", json=payload).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert first["peaks"] == second["peaks"]


def test_invalid_formula_reports_position(client):
    resp = client.post("/api/v1/envelope", json={"formula": "C6Xx2"})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_formula"
    assert detail["position"] == 2
    assert "Xx" in detail["message"]


def test_invalid_label_mass_reports_position(client):
    resp = client.post("/api/v1/envelope", json={"formula": "C6[20C]"})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["position"] == 2
    assert "20" in detail["message"]


def test_top_n_bounds(client):
    resp = client.post("/api/v1/envelope", json={"formula": "H2O", "top_n": 0})
    assert resp.status_code == 422
    resp = client.post(
        "/api/v1/envelope", json={"formula": "H2O", "top_n": settings.top_n_max + 1}
    )
    assert resp.status_code == 422


def test_missing_formula(client):
    resp = client.post("/api/v1/envelope", json={})
    assert resp.status_code == 422


def test_elements_endpoint(client):
    resp = client.get("/api/v1/elements")
    assert resp.status_code == 200
    body = resp.json()
    assert "C" in body and "O" in body
    assert body["C"][0]["mass"] == pytest.approx(12.0)


def _fit_payload(**overrides):
    payload = {
        "formula": "C6",
        "label": "13C",
        "peaks": [
            {"mass": 72.0, "intensity": 100.0},
            {"mass": 73.00335, "intensity": 6.5},
        ],
        "step_percent": 1.0,
        "mass_tolerance": 0.05,
    }
    payload.update(overrides)
    return payload


def test_enrichment_fit_recovers_abundance(client):
    # Natural-abundance glucose envelope -> best 13C abundance near the
    # natural 1.07%, i.e. the 1.0% grid point.
    env = client.post("/api/v1/envelope", json={"formula": "C6H12O6", "top_n": 10})
    peaks = env.json()["peaks"]
    resp = client.post(
        "/api/v1/enrichment-fit",
        json=_fit_payload(formula="C6H12O6", peaks=peaks, step_percent=0.5),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["best_abundance_percent"] == pytest.approx(1.0)
    assert body["search_points"] == 201
    assert body["mse"] >= 0.0
    assert body["pair_count"] == len(body["pairs"])
    # every observed peak appears exactly once in the pairing
    seen = [p["observed_mass"] for p in body["pairs"] if p["observed_mass"] is not None]
    assert len(seen) == len(peaks)


def test_enrichment_fit_perfect_match_zero_mse(client):
    # C6 at 50% 13C: binomial envelope, 7 isotopologue peaks.
    peaks = [
        {"mass": m, "intensity": i}
        for m, i in synthetic_peaks("C6", "C", 13, 0.5, top=7)
    ]
    resp = client.post(
        "/api/v1/enrichment-fit",
        json=_fit_payload(peaks=peaks, step_percent=0.5),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["best_abundance_percent"] == pytest.approx(50.0)
    assert body["mse"] == pytest.approx(0.0, abs=1e-9)


def test_enrichment_fit_search_points_with_coarse_step(client):
    resp = client.post(
        "/api/v1/enrichment-fit", json=_fit_payload(step_percent=10.0)
    )
    assert resp.status_code == 200
    assert resp.json()["search_points"] == 11


def test_enrichment_fit_invalid_label(client):
    for label, position in [("C13", 0), ("13Zz", 2), ("20C", 0), ("[13C]", 0), ("", 0)]:
        resp = client.post("/api/v1/enrichment-fit", json=_fit_payload(label=label))
        assert resp.status_code == 422, label
        detail = resp.json()["detail"]
        assert detail["error"] == "invalid_label"
        assert detail["position"] == position


def test_enrichment_fit_invalid_formula(client):
    resp = client.post("/api/v1/enrichment-fit", json=_fit_payload(formula="C6Xx2"))
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_formula"


def test_enrichment_fit_empty_peaks(client):
    resp = client.post("/api/v1/enrichment-fit", json=_fit_payload(peaks=[]))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_peaks"


def test_enrichment_fit_non_increasing_masses(client):
    peaks = [
        {"mass": 72.0, "intensity": 100.0},
        {"mass": 71.0, "intensity": 5.0},
    ]
    resp = client.post("/api/v1/enrichment-fit", json=_fit_payload(peaks=peaks))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_peaks"
    assert detail["index"] == 1


def test_enrichment_fit_duplicate_masses(client):
    peaks = [
        {"mass": 72.0, "intensity": 100.0},
        {"mass": 72.0, "intensity": 5.0},
    ]
    resp = client.post("/api/v1/enrichment-fit", json=_fit_payload(peaks=peaks))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_peaks"
    assert detail["index"] == 1
    assert "Duplicate" in detail["message"]


def test_enrichment_fit_parameter_bounds(client):
    resp = client.post("/api/v1/enrichment-fit", json=_fit_payload(step_percent=0.4))
    assert resp.status_code == 422
    resp = client.post("/api/v1/enrichment-fit", json=_fit_payload(step_percent=10.5))
    assert resp.status_code == 422
    resp = client.post("/api/v1/enrichment-fit", json=_fit_payload(mass_tolerance=0.0))
    assert resp.status_code == 422
    resp = client.post("/api/v1/enrichment-fit", json=_fit_payload(mass_tolerance=-0.1))
    assert resp.status_code == 422
    peaks = [{"mass": float(m), "intensity": 1.0} for m in range(201)]
    resp = client.post("/api/v1/enrichment-fit", json=_fit_payload(peaks=peaks))
    assert resp.status_code == 422
    peaks = [{"mass": 72.0, "intensity": -1.0}]
    resp = client.post("/api/v1/enrichment-fit", json=_fit_payload(peaks=peaks))
    assert resp.status_code == 422


def _synthetic_dual_payload(formula, enrichments, **overrides):
    parsed = parse_formula(formula)
    distributions = {
        sym: enriched_distribution(sym, mass_number, fraction)
        for sym, (mass_number, fraction) in enrichments.items()
    }
    env = [(0.0, 1.0)]
    for sym, count in parsed.composition.items():
        base = list(distributions.get(sym, ISOTOPES[sym]))
        env = convolve(env, power_polynomial(base, count, 0.0001, 1e-12), 0.0001, 1e-12)
    peaks = [
        {"mass": m, "intensity": i} for m, i in top_peaks(merge_peaks(env, 0.0001), 20)
    ]
    labels = [
        {"label": f"{mass_number}{sym}", "step_percent": 1.0}
        for sym, (mass_number, _) in enrichments.items()
    ]
    payload = {
        "formula": formula,
        "labels": labels,
        "peaks": peaks,
        "mass_tolerance": 0.05,
    }
    payload.update(overrides)
    return payload


def test_dual_enrichment_fit_recovers_both_abundances(client):
    payload = _synthetic_dual_payload(
        "C4N2", {"C": (13, 0.4), "N": (15, 0.6)}
    )
    payload["labels"][0]["step_percent"] = 2.0
    payload["labels"][1]["step_percent"] = 2.0
    resp = client.post("/api/v1/dual-enrichment-fit", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [b["label"] for b in body["best_abundances"]] == ["13C", "15N"]
    assert body["best_abundances"][0]["best_abundance_percent"] == pytest.approx(40.0)
    assert body["best_abundances"][1]["best_abundance_percent"] == pytest.approx(60.0)
    assert body["search_points"] == 51 * 51
    assert body["mse"] == pytest.approx(0.0, abs=1e-6)
    assert body["pair_count"] == len(body["pairs"])
    # every observed peak appears exactly once in the pairing
    seen = [p["observed_mass"] for p in body["pairs"] if p["observed_mass"] is not None]
    assert len(seen) == len(payload["peaks"])


def test_dual_enrichment_fit_endpoint_tie_lexicographic(client):
    # H2O contains neither C nor N: all grid points tie -> (0, 0).
    payload = {
        "formula": "H2O",
        "labels": [
            {"label": "13C", "step_percent": 5.0},
            {"label": "15N", "step_percent": 5.0},
        ],
        "peaks": [
            {"mass": 18.010565, "intensity": 100.0},
            {"mass": 19.01394, "intensity": 0.05},
        ],
        "mass_tolerance": 0.05,
    }
    resp = client.post("/api/v1/dual-enrichment-fit", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["best_abundances"][0]["best_abundance_percent"] == 0.0
    assert body["best_abundances"][1]["best_abundance_percent"] == 0.0
    assert body["search_points"] == 21 * 21


def test_dual_enrichment_fit_duplicate_element_labels_422(client):
    payload = _synthetic_dual_payload("C6", {"C": (13, 0.5)})
    payload["labels"] = [
        {"label": "13C", "step_percent": 5.0},
        {"label": "12C", "step_percent": 5.0},
    ]
    resp = client.post("/api/v1/dual-enrichment-fit", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_labels"
    assert detail["index"] == 1
    assert "different elements" in detail["message"]


def test_dual_enrichment_fit_same_label_twice_422(client):
    payload = _synthetic_dual_payload("C6", {"C": (13, 0.5)})
    payload["labels"] = [
        {"label": "13C", "step_percent": 5.0},
        {"label": "13C", "step_percent": 5.0},
    ]
    resp = client.post("/api/v1/dual-enrichment-fit", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_labels"
    assert detail["index"] == 1


def test_dual_enrichment_fit_malformed_label_422(client):
    # A malformed *second* label is reported via the position-aware
    # invalid_label error rather than the duplicate-element error.
    payload = _synthetic_dual_payload("C6N2", {"C": (13, 0.5), "N": (15, 0.5)})
    payload["labels"][1] = {"label": "20N", "step_percent": 5.0}
    resp = client.post("/api/v1/dual-enrichment-fit", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_label"
    assert detail["position"] == 0


def test_dual_enrichment_fit_wrong_label_count_422(client):
    payload = _synthetic_dual_payload("C6N2", {"C": (13, 0.5), "N": (15, 0.5)})
    payload["labels"] = [payload["labels"][0]]
    resp = client.post("/api/v1/dual-enrichment-fit", json=payload)
    assert resp.status_code == 422
    payload["labels"] = [
        {"label": "13C"}, {"label": "15N"}, {"label": "18O"}
    ]
    resp = client.post("/api/v1/dual-enrichment-fit", json=payload)
    assert resp.status_code == 422


def test_dual_enrichment_fit_step_bounds_and_peak_validation(client):
    payload = _synthetic_dual_payload("C6N2", {"C": (13, 0.5), "N": (15, 0.5)})
    payload["labels"][1]["step_percent"] = 0.4
    assert client.post("/api/v1/dual-enrichment-fit", json=payload).status_code == 422
    payload["labels"][1]["step_percent"] = 10.5
    assert client.post("/api/v1/dual-enrichment-fit", json=payload).status_code == 422

    payload = _synthetic_dual_payload("C6N2", {"C": (13, 0.5), "N": (15, 0.5)})
    payload["mass_tolerance"] = 0.0
    assert client.post("/api/v1/dual-enrichment-fit", json=payload).status_code == 422

    payload = _synthetic_dual_payload("C6N2", {"C": (13, 0.5), "N": (15, 0.5)})
    payload["peaks"] = []
    resp = client.post("/api/v1/dual-enrichment-fit", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_peaks"

    payload = _synthetic_dual_payload("C6N2", {"C": (13, 0.5), "N": (15, 0.5)})
    payload["formula"] = "C6Xx2"
    resp = client.post("/api/v1/dual-enrichment-fit", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_formula"


# --- /api/v1/formula-search -------------------------------------------------

# Full (untruncated) natural-abundance glucose envelope for scoring tests:
# with every predicted peak observed, glucose itself scores MSE ~ 0.
GLUCOSE_FULL_PEAKS = [
    {"mass": round(m, 6), "intensity": round(i, 6)}
    for m, i in isotope_envelope(parse_formula("C6H12O6"), 0.0001, 1e-12)
]


def _formula_search_payload(**overrides):
    payload = {
        "target_mass": 180.063388,
        "observed_peaks": GLUCOSE_FULL_PEAKS,
        "elements": [
            {"symbol": "C", "min_count": 0, "max_count": 12},
            {"symbol": "H", "min_count": 0, "max_count": 30},
            {"symbol": "O", "min_count": 0, "max_count": 8},
        ],
        "mass_tolerance": 0.01,
        "max_results": 10,
    }
    payload.update(overrides)
    return payload


def test_formula_search_finds_glucose(client):
    resp = client.post("/api/v1/formula-search", json=_formula_search_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_hits"] >= 1
    assert body["observed_peak_count"] == len(GLUCOSE_FULL_PEAKS)
    assert body["visited_nodes"] >= 1
    assert body["pruned_branches"] >= 0
    assert len(body["candidates"]) == min(10, body["total_hits"])

    best = body["candidates"][0]
    assert best["rank"] == 1
    assert best["formula"] == "C6H12O6"
    assert best["composition"] == {"C": 6, "H": 12, "O": 6}
    assert best["monoisotopic_mass"] == pytest.approx(180.063388, abs=1e-5)
    assert abs(best["mass_error"]) < 1e-5
    assert best["mse"] == pytest.approx(0.0, abs=1e-6)
    assert best["pair_count"] == len(best["pairs"])
    # Pairs are serialised by mass ascending and reference both sides.
    masses = [
        p["observed_mass"] if p["observed_mass"] is not None else p["predicted_mass"]
        for p in best["pairs"]
    ]
    assert masses == sorted(masses)


def test_formula_search_ranks_by_mse_then_mass_error_then_formula(client):
    body = client.post(
        "/api/v1/formula-search", json=_formula_search_payload(max_results=50)
    ).json()
    keys = [
        (c["mse"], abs(c["mass_error"]), c["formula"]) for c in body["candidates"]
    ]
    assert keys == sorted(keys)
    ranks = [c["rank"] for c in body["candidates"]]
    assert ranks == list(range(1, len(ranks) + 1))


def test_formula_search_no_hits_returns_200_empty_list(client):
    payload = _formula_search_payload(
        target_mass=500.0,
        observed_peaks=[{"mass": 500.0, "intensity": 100.0}],
        elements=[
            {"symbol": "C", "min_count": 0, "max_count": 5},
            {"symbol": "H", "min_count": 0, "max_count": 5},
        ],
    )
    resp = client.post("/api/v1/formula-search", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["candidates"] == []
    assert body["total_hits"] == 0
    assert body["visited_nodes"] == 1
    assert body["pruned_branches"] > 0


def test_formula_search_too_many_visited_nodes_returns_422(client):
    # Wide-open windows over six elements exceed the 1,000,000-node cap.
    payload = _formula_search_payload(
        target_mass=60000.0,
        observed_peaks=[{"mass": 60000.0, "intensity": 100.0}],
        elements=[
            {"symbol": symbol, "min_count": 0, "max_count": 200}
            for symbol in ("C", "H", "O", "N", "S", "P")
        ],
        mass_tolerance=60000.0,
    )
    resp = client.post("/api/v1/formula-search", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "search_limit_exceeded"
    assert detail["visited_nodes"] > 1_000_000
    assert detail["limit"] == 1_000_000


def test_formula_search_duplicate_element_422(client):
    payload = _formula_search_payload()
    payload["elements"].append({"symbol": "C", "min_count": 0, "max_count": 1})
    resp = client.post("/api/v1/formula-search", json=payload)
    assert resp.status_code == 422


def test_formula_search_unknown_element_422(client):
    payload = _formula_search_payload()
    payload["elements"][0]["symbol"] = "Xx"
    resp = client.post("/api/v1/formula-search", json=payload)
    assert resp.status_code == 422


def test_formula_search_min_above_max_422(client):
    payload = _formula_search_payload()
    payload["elements"][0]["min_count"] = 9
    payload["elements"][0]["max_count"] = 5
    resp = client.post("/api/v1/formula-search", json=payload)
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_count", -1),
        ("max_count", 201),
    ],
)
def test_formula_search_count_bounds_422(client, field, value):
    payload = _formula_search_payload()
    payload["elements"][0][field] = value
    assert client.post("/api/v1/formula-search", json=payload).status_code == 422


def test_formula_search_element_count_must_be_2_to_6(client):
    one = _formula_search_payload()
    one["elements"] = one["elements"][:1]
    assert client.post("/api/v1/formula-search", json=one).status_code == 422

    seven = _formula_search_payload(
        elements=[
            {"symbol": symbol, "min_count": 0, "max_count": 1}
            for symbol in ("C", "H", "O", "N", "S", "P", "F")
        ]
    )
    assert client.post("/api/v1/formula-search", json=seven).status_code == 422


def test_formula_search_scalar_bounds_422(client):
    for overrides in (
        {"target_mass": 0.0},
        {"target_mass": -5.0},
        {"mass_tolerance": 0.0},
        {"mass_tolerance": -0.1},
        {"max_results": 0},
        {"max_results": 51},
    ):
        payload = _formula_search_payload(**overrides)
        assert client.post("/api/v1/formula-search", json=payload).status_code == 422


def test_formula_search_unsorted_peaks_422(client):
    payload = _formula_search_payload(
        observed_peaks=[
            {"mass": 182.0, "intensity": 1.0},
            {"mass": 181.0, "intensity": 1.0},
        ]
    )
    resp = client.post("/api/v1/formula-search", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_peaks"


def test_formula_search_all_zero_intensities_422(client):
    payload = _formula_search_payload(
        observed_peaks=[
            {"mass": 180.0, "intensity": 0.0},
            {"mass": 181.0, "intensity": 0.0},
        ]
    )
    resp = client.post("/api/v1/formula-search", json=payload)
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "invalid_peaks"
