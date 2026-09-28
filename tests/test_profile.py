"""Tests for the Gaussian instrument-profile simulation (unit + API)."""

import math

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.envelope import isotope_envelope
from app.parser import parse_formula
from app.profile import (
    FWHM_TO_SIGMA,
    MAX_GRID_POINTS,
    SIGMA_SUPPORT,
    ProfileParameterError,
    simulate_profile,
)

TOL = 0.0001
PRUNE = 1e-12


def sticks_of(formula: str):
    return isotope_envelope(parse_formula(formula), TOL, PRUNE)


def gaussian(mass: float, intensity: float, m: float, sigma: float) -> float:
    delta = m - mass
    return intensity * math.exp(-(delta * delta) / (2.0 * sigma * sigma))



def test_single_stick_default_bounds_span_4_sigma():
    sticks = sticks_of("F")
    mass = sticks[0][0]
    rp = 10000.0
    sigma = FWHM_TO_SIGMA * mass / rp
    res = simulate_profile(sticks, rp, 0.0001)
    assert res.mass_min == pytest.approx(mass - SIGMA_SUPPORT * sigma)
    assert res.mass_max == pytest.approx(mass + SIGMA_SUPPORT * sigma)
    assert res.point_count == len(res.profile)
    assert res.point_count == int((res.mass_max - res.mass_min) / 0.0001 + 1e-9) + 1


def test_single_stick_gaussian_values_and_fwhm():
    sticks = sticks_of("F")
    mass = sticks[0][0]
    rp = 10000.0
    sigma = FWHM_TO_SIGMA * mass / rp
    res = simulate_profile(sticks, rp, 0.00005)
    # The grid is anchored at the +-4 sigma edge, so the stick itself need
    # not be a grid point; compare the normalised *discrete* Gaussian, whose
    # sampled maximum defines the 100 reference.
    expected = [gaussian(mass, 1.0, m, sigma) for m, _ in res.profile]
    base = max(expected)
    for (_, i), e in zip(res.profile, expected):
        assert i == pytest.approx(100.0 * e / base, abs=1e-6)
    assert max(i for _, i in res.profile) == pytest.approx(100.0)
    # Measured FWHM across the discrete grid agrees with m/R (within one
    # grid spacing of quantisation).
    half = [m for m, i in res.profile if i >= 50.0]
    assert half[-1] - half[0] == pytest.approx(mass / rp, abs=0.0001)


def test_profile_is_base_peak_normalised_for_multiple_sticks():
    sticks = sticks_of("C6H12O6")
    res = simulate_profile(sticks, 100000.0, 0.0005)
    assert max(i for _, i in res.profile) == pytest.approx(100.0)
    masses = [m for m, _ in res.profile]
    assert masses == sorted(masses)
    # Uniform grid
    diffs = {round(b - a, 12) for a, b in zip(masses, masses[1:])}
    assert diffs == {0.0005}


def test_explicit_range_is_clipped_but_keeps_inside_contribution():
    sticks = sticks_of("F")
    mass = sticks[0][0]
    rp = 10000.0
    sigma = FWHM_TO_SIGMA * mass / rp
    lo, hi = mass - 0.0005, mass + 0.001
    res = simulate_profile(sticks, rp, 0.0001, mass_min=lo, mass_max=hi)
    assert res.mass_min == lo
    assert res.mass_max == hi
    assert res.point_count == len(res.profile)
    # Kernels are truncated at the window edges but the in-window values are
    # the unrenormalised Gaussian (only the final base-peak scaling to 100
    # applies), so an asymmetric window shifts neither shape nor peak.
    expected = [gaussian(mass, 1.0, m, sigma) for m, _ in res.profile]
    base = max(expected)
    for (_, i), e in zip(res.profile, expected):
        assert i == pytest.approx(100.0 * e / base, abs=1e-6)
    peak_mass = max(res.profile, key=lambda p: p[1])[0]
    assert abs(peak_mass - mass) < 0.0001


def test_one_sided_default_uses_union_bound():
    sticks = sticks_of("F")
    mass = sticks[0][0]
    rp = 50000.0
    step = 0.0001
    auto_lo = mass - SIGMA_SUPPORT * FWHM_TO_SIGMA * mass / rp
    auto_hi = mass + SIGMA_SUPPORT * FWHM_TO_SIGMA * mass / rp
    res_lo = simulate_profile(sticks, rp, step, mass_min=auto_lo)
    assert res_lo.mass_min == pytest.approx(auto_lo)
    assert res_lo.mass_max == pytest.approx(auto_hi)
    res_hi = simulate_profile(sticks, rp, step, mass_max=auto_hi)
    assert res_hi.mass_max == pytest.approx(auto_hi)
    assert res_hi.mass_min == pytest.approx(auto_lo)


def test_grid_limit_allows_exactly_max_points():
    sticks = sticks_of("F")
    step = 0.001
    lo = 1.0
    # R=1 gives a kernel broad enough to reach the whole window.
    res = simulate_profile(
        sticks, 1.0, step, mass_min=lo, mass_max=lo + (MAX_GRID_POINTS - 1) * step
    )
    assert res.point_count == MAX_GRID_POINTS


@pytest.mark.parametrize(
    "kwargs,error,field",
    [
        ({"mass_step": 0.0}, "invalid_mass_step", "mass_step"),
        ({"mass_step": -0.1}, "invalid_mass_step", "mass_step"),
        ({"mass_min": 20.0, "mass_max": 10.0}, "invalid_mass_range", "mass_min"),
        ({"mass_min": 19.0, "mass_max": 19.0}, "invalid_mass_range", "mass_min"),
        (
            {"mass_min": 1.0, "mass_max": 1.0 + MAX_GRID_POINTS * 0.001},
            "grid_too_large",
            "mass_step",
        ),
        # A subnormal step would overflow the point-count arithmetic.
        ({"mass_step": 5e-324}, "grid_too_large", "mass_step"),
        (
            {"mass_min": 100.0, "mass_max": 101.0},
            "no_profile_points",
            "mass_min",
        ),
    ],
)
def test_parameter_errors_are_locatable(kwargs, error, field):
    sticks = sticks_of("F")
    args = {"resolving_power": 10000.0, "mass_step": 0.0001}
    args.update(kwargs)
    with pytest.raises(ProfileParameterError) as exc:
        simulate_profile(sticks, **args)
    assert exc.value.error == error
    assert exc.value.field == field


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_api_neutral_profile_default_range(client):
    resp = client.post(
        "/api/v1/neutral-profile",
        json={"formula": "C6H12O6", "resolving_power": 100000, "mass_step": 0.001},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["formula"] == "C6H12O6"
    assert body["point_count"] == len(body["profile"])
    assert body["point_count"] <= MAX_GRID_POINTS
    masses = [p["mass"] for p in body["profile"]]
    assert masses[0] == pytest.approx(body["mass_min"])
    assert masses == sorted(masses)
    assert max(p["intensity"] for p in body["profile"]) == pytest.approx(100.0)


def test_api_neutral_profile_explicit_range_and_labels(client):
    # C6H6[13C]2 sits around 104 Da; choose a narrow window that clips the
    # kernels but still reaches them.
    lo, hi = 103.5, 105.0
    resp = client.post(
        "/api/v1/neutral-profile",
        json={
            "formula": "C6H6[13C]2",
            "resolving_power": 50000,
            "mass_step": 0.0005,
            "mass_min": lo,
            "mass_max": hi,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mass_min"] == lo
    assert body["mass_max"] == hi
    assert body["point_count"] == len(body["profile"])
    assert max(p["intensity"] for p in body["profile"]) == pytest.approx(100.0)


def test_api_neutral_profile_invalid_formula_422(client):
    resp = client.post(
        "/api/v1/neutral-profile",
        json={"formula": "C6Xx2", "resolving_power": 10000, "mass_step": 0.001},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_formula"
    assert detail["position"] == 2


def test_api_neutral_profile_bad_range_422(client):
    resp = client.post(
        "/api/v1/neutral-profile",
        json={
            "formula": "F",
            "resolving_power": 10000,
            "mass_step": 0.0001,
            "mass_min": 20.0,
            "mass_max": 19.0,
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_mass_range"
    assert detail["field"] == "mass_min"


def test_api_neutral_profile_no_points_422(client):
    resp = client.post(
        "/api/v1/neutral-profile",
        json={
            "formula": "F",
            "resolving_power": 10000,
            "mass_step": 0.0001,
            "mass_min": 100.0,
            "mass_max": 101.0,
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "no_profile_points"


def test_api_neutral_profile_grid_too_large_422(client):
    resp = client.post(
        "/api/v1/neutral-profile",
        json={
            "formula": "F",
            "resolving_power": 10000,
            "mass_step": 0.0001,
            "mass_min": 0.0001,
            "mass_max": 100.0,
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "grid_too_large"
    assert detail["field"] == "mass_step"


@pytest.mark.parametrize("payload", [
    {"formula": "F", "resolving_power": 0, "mass_step": 0.001},
    {"formula": "F", "resolving_power": 10000, "mass_step": 0},
    {"formula": "F", "resolving_power": 10000, "mass_step": -1.0},
    {"formula": "F", "resolving_power": 10000, "mass_step": 0.001, "mass_min": 0},
    {"formula": "F", "resolving_power": 10000, "mass_step": 0.001, "mass_min": "inf"},
    {"formula": "F", "resolving_power": "nan", "mass_step": 0.001},
    {"resolving_power": 10000, "mass_step": 0.001},
])
def test_api_neutral_profile_parameter_validation_422(client, payload):
    resp = client.post("/api/v1/neutral-profile", json=payload)
    assert resp.status_code == 422, resp.text
