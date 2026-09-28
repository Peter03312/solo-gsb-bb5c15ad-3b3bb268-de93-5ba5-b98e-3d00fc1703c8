"""Tests for the molecular-formula reverse search (branch and bound)."""

import gc
import weakref

import pytest

from app.envelope import isotope_envelope, top_peaks
from app.formula_search import (
    MAX_VISITED_NODES,
    ElementSpec,
    FormulaHit,
    SearchLimitExceeded,
    _BoundedCandidates,
    enumerate_compositions,
    formula_string,
    iter_compositions,
    search_formulas,
)
from app.isotopes import most_abundant_isotope_mass
from app.parser import parse_formula

TOL = 0.0001
PRUNE = 1e-12


def _natural_peaks(formula: str, top: int = 10):
    env = isotope_envelope(parse_formula(formula), TOL, PRUNE)
    return top_peaks(env, top)


# --- formula_string ---------------------------------------------------------

def test_formula_string_lexicographic_symbols():
    assert formula_string({"C": 6, "H": 12, "O": 6}) == "C6H12O6"
    # Pure lexicographic order: "C" is a prefix of "Cl".
    assert formula_string({"C": 1, "Cl": 2, "H": 3}) == "CCl2H3"
    assert formula_string({"O": 1}) == "O"
    assert formula_string({"H": 0, "O": 1}) == "O"
    assert formula_string({}) == ""


# --- enumeration ------------------------------------------------------------

def test_enumerate_glucose_is_the_only_mass_hit():
    raw, visited, pruned = enumerate_compositions(
        180.063388,
        0.005,
        [
            ElementSpec("C", 0, 20),
            ElementSpec("H", 0, 40),
            ElementSpec("O", 0, 10),
        ],
    )
    assert len(raw) == 1
    counts, mass = raw[0]
    # Visit order is heaviest monoisotopic mass first: O, C, H.
    assert dict(zip(("O", "C", "H"), counts)) == {"O": 6, "C": 6, "H": 12}
    assert mass == pytest.approx(180.063388, abs=1e-5)
    assert visited >= 1
    assert pruned > 0


def test_enumerate_matches_brute_force():
    specs = [
        ElementSpec("C", 0, 4),
        ElementSpec("H", 0, 6),
        ElementSpec("N", 0, 3),
    ]
    target, tolerance = 42.0, 3.0
    raw, _, _ = enumerate_compositions(target, tolerance, specs)
    order = sorted(
        specs, key=lambda e: (-most_abundant_isotope_mass(e.symbol), e.symbol)
    )
    got = {frozenset(zip([e.symbol for e in order], c)) for c, _ in raw}

    expected = set()
    for c_c in range(5):
        for c_h in range(7):
            for c_n in range(4):
                mass = (
                    c_c * most_abundant_isotope_mass("C")
                    + c_h * most_abundant_isotope_mass("H")
                    + c_n * most_abundant_isotope_mass("N")
                )
                if mass > 0 and abs(mass - target) <= tolerance:
                    expected.add(frozenset(zip("CHN", (c_c, c_h, c_n))))
    assert got == expected


def test_enumerate_pruned_counter_accounts_for_skipped_counts():
    # Only count 1 is feasible for a single O at mass ~16: 5 of the 6
    # candidate count values are pruned; the root plus the one leaf are the
    # visited nodes.
    raw, visited, pruned = enumerate_compositions(
        most_abundant_isotope_mass("O"), 0.5, [ElementSpec("O", 0, 5)]
    )
    assert len(raw) == 1
    assert visited == 2
    assert pruned == 5


def test_enumerate_unreachable_window_prunes_everything_at_root():
    raw, visited, pruned = enumerate_compositions(
        500.0, 0.01, [ElementSpec("C", 0, 5), ElementSpec("H", 0, 5)]
    )
    assert raw == []
    assert visited == 1
    assert pruned > 0


def test_enumerate_respects_min_and_max_counts():
    # H2O would hit 18.01, but min_count H=3 forbids it.
    raw, _, _ = enumerate_compositions(
        18.01, 0.01, [ElementSpec("H", 3, 5), ElementSpec("O", 1, 1)]
    )
    assert raw == []

    # min_count forces atoms into every hit.
    raw, _, _ = enumerate_compositions(
        50.0, 5.0, [ElementSpec("N", 2, 4), ElementSpec("C", 0, 4)]
    )
    assert raw
    assert all(c[0] >= 2 for c, _ in raw)  # visit order: N first
    assert all(c[0] <= 4 and c[1] <= 4 for c, _ in raw)


def test_enumerate_excludes_all_zero_composition_under_huge_tolerance():
    raw, _, _ = enumerate_compositions(
        0.0, 1000.0, [ElementSpec("C", 0, 2), ElementSpec("H", 0, 2)]
    )
    assert raw
    assert all(any(c > 0 for c in counts) for counts, _ in raw)


def test_enumerate_zero_tolerance_needs_exact_mass():
    m_water = 2 * most_abundant_isotope_mass("H") + most_abundant_isotope_mass("O")
    raw, _, _ = enumerate_compositions(
        m_water, 0.0, [ElementSpec("H", 0, 5), ElementSpec("O", 0, 5)]
    )
    assert len(raw) == 1

    raw, _, _ = enumerate_compositions(
        m_water + 0.001, 0.0, [ElementSpec("H", 0, 5), ElementSpec("O", 0, 5)]
    )
    assert raw == []


def test_enumerate_tolerance_window_boundaries_are_inclusive():
    m_water = 2 * most_abundant_isotope_mass("H") + most_abundant_isotope_mass("O")
    for target in (m_water + 0.01, m_water - 0.01):
        raw, _, _ = enumerate_compositions(
            target, 0.01, [ElementSpec("H", 0, 5), ElementSpec("O", 0, 5)]
        )
        assert any(abs(m - m_water) < 1e-9 for _, m in raw)


def test_enumerate_independent_of_request_element_order():
    specs_a = [ElementSpec("N", 0, 3), ElementSpec("C", 0, 4), ElementSpec("H", 0, 6)]
    specs_b = list(reversed(specs_a))
    raw_a, va, pa = enumerate_compositions(42.0, 3.0, specs_a)
    raw_b, vb, pb = enumerate_compositions(42.0, 3.0, specs_b)

    def normalise(raw, specs):
        order = sorted(
            specs, key=lambda e: (-most_abundant_isotope_mass(e.symbol), e.symbol)
        )
        return {
            frozenset(zip([e.symbol for e in order], counts)): round(mass, 9)
            for counts, mass in raw
        }

    assert normalise(raw_a, specs_a) == normalise(raw_b, specs_b)
    assert va == vb and pa == pb


def test_enumerate_aborts_after_one_million_visited_nodes():
    # Windows wide open on six 0..200 elements: far beyond 1e6 nodes.
    with pytest.raises(SearchLimitExceeded) as excinfo:
        enumerate_compositions(
            60000.0,
            60000.0,
            [ElementSpec(symbol, 0, 200) for symbol in "CHONSP"],
        )
    assert excinfo.value.visited_nodes == MAX_VISITED_NODES + 1


# --- scoring/ranking --------------------------------------------------------

def test_search_scores_glucose_best_from_its_natural_envelope():
    observed = _natural_peaks("C6H12O6", top=8)
    result = search_formulas(
        180.063388,
        observed,
        [
            ElementSpec("C", 0, 12),
            ElementSpec("H", 0, 30),
            ElementSpec("N", 0, 3),
            ElementSpec("O", 0, 8),
        ],
        mass_tolerance=0.01,
        max_results=20,
        merge_tolerance=TOL,
        prune=PRUNE,
    )
    assert result.hits
    winner = result.hits[0]
    assert winner.formula == "C6H12O6"
    assert winner.composition == {"C": 6, "H": 12, "O": 6}
    # The observed list carries only the top 8 predicted peaks; every other
    # predicted peak is unmatched and counts at zero intensity (same MSE
    # convention as the enrichment/identify endpoints), so the winner is not
    # exactly zero but must be far better than any near-isobaric formula.
    assert winner.mass_error == pytest.approx(0.0, abs=1e-5)
    assert winner.mse == pytest.approx(0.0, abs=1e-3)
    assert result.hits[1].mse > winner.mse * 100
    # Ranking key: (mse, |mass error|, formula).
    keys = [(h.mse, abs(h.mass_error), h.formula) for h in result.hits]
    assert keys == sorted(keys)


def test_search_caps_returned_hits_but_reports_total():
    result = search_formulas(
        42.0,
        [(42.0, 100.0)],
        [ElementSpec("C", 0, 4), ElementSpec("H", 0, 6), ElementSpec("N", 0, 3)],
        mass_tolerance=3.0,
        max_results=2,
        merge_tolerance=TOL,
        prune=PRUNE,
    )
    assert result.total_hits == 18
    assert len(result.hits) == 2


def test_search_empty_hit_list_when_no_composition_in_window():
    result = search_formulas(
        500.0,
        [(500.0, 100.0)],
        [ElementSpec("C", 0, 5), ElementSpec("H", 0, 5)],
        mass_tolerance=0.01,
        max_results=10,
        merge_tolerance=TOL,
        prune=PRUNE,
    )
    assert result.hits == []
    assert result.total_hits == 0
    assert result.visited_nodes == 1
    assert result.pruned_branches > 0


def test_search_hit_pairs_one_to_one_with_observed_peaks():
    observed = _natural_peaks("C6H12O6", top=5)
    result = search_formulas(
        180.063388,
        observed,
        [ElementSpec("C", 4, 8), ElementSpec("H", 8, 16), ElementSpec("O", 4, 8)],
        mass_tolerance=0.01,
        max_results=5,
        merge_tolerance=TOL,
        prune=PRUNE,
    )
    hit = result.hits[0]
    # Every observed and every predicted peak participates in at least one pair.
    obs_indices = {i for i, _ in hit.pairs if i is not None}
    pred_indices = {j for _, j in hit.pairs if j is not None}
    assert obs_indices == set(range(len(observed)))
    assert len(pred_indices) <= len(hit.predicted)
    assert len(hit.pairs) >= 1


# --- streaming enumeration --------------------------------------------------

def _drain(stream):
    hits = []
    try:
        while True:
            hits.append(next(stream))
    except StopIteration as done:
        stats = done.value
    return hits, stats


def test_iter_compositions_matches_list_entry_and_returns_counters():
    specs = [
        ElementSpec("C", 0, 4),
        ElementSpec("H", 0, 6),
        ElementSpec("N", 0, 3),
    ]
    target, tolerance = 42.0, 3.0
    streamed, (stream_visited, stream_pruned) = _drain(
        iter_compositions(target, tolerance, specs)
    )
    listed, visited, pruned = enumerate_compositions(target, tolerance, specs)
    # Same hits, in the same traversal order, same counters.
    assert streamed == listed
    assert (stream_visited, stream_pruned) == (visited, pruned)


def test_iter_compositions_emits_in_depth_first_order():
    # Visit order is heaviest first (O, C, H); within an element counts
    # ascend, so the stream order is the DFS leaf order.
    stream = iter_compositions(
        42.0,
        3.0,
        [
            ElementSpec("C", 0, 4),
            ElementSpec("H", 0, 6),
            ElementSpec("O", 0, 3),
        ],
    )
    streamed, _ = _drain(stream)
    assert streamed == sorted(streamed)  # tuples ascend in (O, C, H) counts


def test_iter_compositions_does_not_abort_when_consumer_stops_early():
    # A consumer taking only the first hit must be able to close the
    # generator without the node limit ever being observed.
    stream = iter_compositions(
        180.063388,
        0.005,
        [
            ElementSpec("C", 0, 20),
            ElementSpec("H", 0, 40),
            ElementSpec("O", 0, 10),
        ],
    )
    first = next(stream)
    assert dict(zip(("O", "C", "H"), first[0])) == {"O": 6, "C": 6, "H": 12}
    stream.close()


def test_iter_compositions_carries_counters_when_limit_exceeded():
    stream = iter_compositions(
        60000.0,
        60000.0,
        [ElementSpec(symbol, 0, 200) for symbol in "CHONSP"],
    )
    with pytest.raises(SearchLimitExceeded) as excinfo:
        _drain(stream)
    assert excinfo.value.visited_nodes == MAX_VISITED_NODES + 1


# --- bounded candidate structure --------------------------------------------

def _hit(formula, mse, mass_error=0.0):
    return FormulaHit(
        formula=formula,
        composition={},
        monoisotopic_mass=mass_error,
        mass_error=mass_error,
        mse=mse,
        pairs=[],
        predicted=[],
    )


def test_bounded_candidates_keeps_best_by_full_ranking_key():
    bounded = _BoundedCandidates(3)
    offered = [
        _hit("C", mse=5.0),  # seq 0
        _hit("A", mse=1.0),  # seq 1
        _hit("B", mse=3.0),  # seq 2
        _hit("D", mse=3.0),  # seq 3: worse formula than B, evicted
        _hit("E", mse=2.0),  # seq 4: pushes C (mse 5) out
    ]
    for seq, hit in enumerate(offered):
        bounded.offer(hit, seq)
    kept = bounded.ranked_hits()
    assert [h.formula for h in kept] == ["A", "E", "B"]


def test_bounded_candidates_full_key_tie_keeps_first_traversed():
    bounded = _BoundedCandidates(1)
    first = _hit("X", mse=1.0, mass_error=0.0)
    second = _hit("X", mse=1.0, mass_error=0.0)
    bounded.offer(first, seq=0)
    bounded.offer(second, seq=1)
    (kept,) = bounded.ranked_hits()
    assert kept is first


def test_bounded_candidates_evicted_candidate_is_released():
    # Eviction must not retain the displaced hit: a weak reference to it
    # must clear once a strictly better candidate takes its slot.
    bounded = _BoundedCandidates(1)
    worse = _hit("worse", mse=10.0)
    bounded.offer(worse, seq=0)
    ref = weakref.ref(worse)
    del worse
    bounded.offer(_hit("better", mse=1.0), seq=1)
    gc.collect()
    assert ref() is None
    assert [h.formula for h in bounded.ranked_hits()] == ["better"]


def test_bounded_candidates_capacity_is_the_hard_retention_limit():
    bounded = _BoundedCandidates(2)
    for seq, mse in enumerate((4.0, 3.0, 2.0, 1.0)):
        bounded.offer(_hit(f"f{seq}", mse=mse), seq)
    # No more than capacity hits retained, regardless of offers.
    assert len(bounded.ranked_hits()) == 2
    assert [h.mse for h in bounded.ranked_hits()] == [1.0, 2.0]


# --- streaming production search --------------------------------------------

def test_search_topk_matches_collect_all_then_sort():
    specs = [
        ElementSpec("C", 0, 5),
        ElementSpec("H", 0, 8),
        ElementSpec("N", 0, 4),
        ElementSpec("O", 0, 3),
    ]
    kwargs = dict(
        target_mass=60.0,
        observed=[(60.0, 100.0), (61.0, 5.0)],
        elements=specs,
        mass_tolerance=3.0,
        merge_tolerance=TOL,
        prune=PRUNE,
    )
    capped = search_formulas(max_results=3, **kwargs)
    full = search_formulas(max_results=1000, **kwargs)
    assert capped.total_hits == full.total_hits
    assert (
        capped.visited_nodes,
        capped.pruned_branches,
    ) == (full.visited_nodes, full.pruned_branches)
    assert len(capped.hits) == min(3, full.total_hits)
    expected = sorted(
        full.hits, key=lambda h: (h.mse, abs(h.mass_error), h.formula)
    )[:3]
    assert [h.formula for h in capped.hits] == [h.formula for h in expected]


def test_search_releases_every_evicted_candidate_envelope(monkeypatch):
    # The production path must score one hit at a time and drop losers:
    # after a bounded search, weak references to scored hits show only the
    # retained candidates stay alive.
    specs = [
        ElementSpec("C", 0, 5),
        ElementSpec("H", 0, 8),
        ElementSpec("N", 0, 4),
        ElementSpec("O", 0, 3),
    ]
    result = search_formulas(
        target_mass=60.0,
        observed=[(60.0, 100.0), (61.0, 5.0)],
        elements=specs,
        mass_tolerance=3.0,
        max_results=2,
        merge_tolerance=TOL,
        prune=PRUNE,
    )
    assert result.total_hits > 2
    refs = []
    seen = {}

    from app import formula_search as fs

    original = fs._score_hit

    def tracking(*args, **kwargs):
        hit = original(*args, **kwargs)
        seen[id(hit)] = hit
        refs.append(weakref.ref(hit))
        return hit

    monkeypatch.setattr(fs, "_score_hit", tracking)
    bounded = search_formulas(
        target_mass=60.0,
        observed=[(60.0, 100.0), (61.0, 5.0)],
        elements=specs,
        mass_tolerance=3.0,
        max_results=2,
        merge_tolerance=TOL,
        prune=PRUNE,
    )
    del seen
    gc.collect()
    alive = [obj for ref in refs if (obj := ref()) is not None]
    retained = list(bounded.hits)
    # Exactly the retained hits survive; every evicted hit was released.
    assert len(alive) == len(retained)
    assert {id(hit) for hit in alive} == {id(hit) for hit in retained}
    assert len(retained) == 2
