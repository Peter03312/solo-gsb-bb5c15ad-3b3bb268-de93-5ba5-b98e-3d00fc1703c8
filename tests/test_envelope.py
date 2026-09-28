import pytest

from app.envelope import (
    ELECTRON_MASS_DA,
    ion_envelope,
    isotope_envelope,
    mass_to_mz,
    merge_peaks,
    monoisotopic_mass,
    top_peaks,
)
from app.parser import parse_formula

TOL = 0.0001
PRUNE = 1e-12


def envelope_of(formula: str):
    return isotope_envelope(parse_formula(formula), TOL, PRUNE)


def test_monoisotopic_mass_water():
    # 2 * 1.00782503223 + 15.99491461957
    assert monoisotopic_mass(parse_formula("H2O")) == pytest.approx(18.01056468, abs=1e-6)


def test_monoisotopic_mass_glucose():
    assert monoisotopic_mass(parse_formula("C6H12O6")) == pytest.approx(180.0633881, abs=1e-6)


def test_monoisotopic_mass_with_label():
    # natural C6H6 plus two pure 13C atoms
    expected = monoisotopic_mass(parse_formula("C6H6")) + 2 * 13.00335483507
    assert monoisotopic_mass(parse_formula("C6H6[13C]2")) == pytest.approx(expected, abs=1e-9)


def test_probabilities_sum_to_one():
    env = envelope_of("C6H12O6")
    assert sum(i for _, i in env) == pytest.approx(1.0, abs=1e-9)


def test_peaks_merged_beyond_tolerance():
    env = envelope_of("C60")
    masses = [m for m, _ in env]
    gaps = [b - a for a, b in zip(masses, masses[1:])]
    assert all(g >= TOL - 1e-12 for g in gaps)


def test_merge_peaks_weighted_mean():
    merged = merge_peaks([(100.0, 1.0), (100.00005, 3.0), (101.0, 2.0)], TOL)
    assert len(merged) == 2
    mass, inten = merged[0]
    assert inten == pytest.approx(4.0)
    assert mass == pytest.approx((100.0 * 1.0 + 100.00005 * 3.0) / 4.0)
    assert merged[1] == (101.0, 2.0)


def test_base_peak_normalised_to_100():
    peaks = top_peaks(envelope_of("C6H12O6"), 10)
    assert max(i for _, i in peaks) == pytest.approx(100.0)


def test_top_n_limits_and_sorts_by_mass():
    peaks = top_peaks(envelope_of("C6H12O6"), 5)
    assert len(peaks) == 5
    masses = [m for m, _ in peaks]
    assert masses == sorted(masses)


def test_first_peak_is_monoisotopic():
    peaks = top_peaks(envelope_of("C6H12O6"), 10)
    mono = monoisotopic_mass(parse_formula("C6H12O6"))
    assert peaks[0][0] == pytest.approx(mono, abs=TOL)


def test_label_shifts_envelope():
    natural = envelope_of("C6H6")
    labelled = envelope_of("C6H6[13C]2")
    shift = 2 * 13.00335483507
    assert labelled[0][0] == pytest.approx(natural[0][0] + shift, abs=TOL)


def test_single_element_envelope_matches_abundances():
    env = envelope_of("Cl")
    assert len(env) == 2
    masses = {round(m, 6): i for m, i in env}
    assert masses[round(34.968852682, 6)] == pytest.approx(0.7576)
    assert masses[round(36.965902602, 6)] == pytest.approx(0.2424)


def test_mass_to_mz_positive_charge_removes_electrons():
    # singly charged: (M - m_e) / 1
    assert mass_to_mz(100.0, 1) == pytest.approx(100.0 - ELECTRON_MASS_DA)
    # doubly charged: (M - 2 * m_e) / 2
    assert mass_to_mz(100.0, 2) == pytest.approx(50.0 - ELECTRON_MASS_DA)


def test_mass_to_mz_negative_charge_adds_electrons():
    # singly negative: (M - (-1) * m_e) / 1 = M + m_e
    assert mass_to_mz(100.0, -1) == pytest.approx(100.0 + ELECTRON_MASS_DA)
    assert mass_to_mz(100.0, -2) == pytest.approx(50.0 + ELECTRON_MASS_DA)


def test_ion_envelope_preserves_order_and_intensity():
    env = [(18.0, 0.9), (19.0, 0.1)]
    mz_peaks = ion_envelope(env, 2)
    assert [i for _, i in mz_peaks] == [0.9, 0.1]
    assert mz_peaks[0][0] == pytest.approx(mass_to_mz(18.0, 2))
    assert mz_peaks[1][0] == pytest.approx(mass_to_mz(19.0, 2))
    assert mz_peaks[1][0] > mz_peaks[0][0]
