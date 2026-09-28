"""Molecular-formula reverse search by branch-and-bound enumeration.

Given a target monoisotopic mass, observed isotope peaks and per-element
count ranges (2-6 distinct elements, each count between a ``min_count`` and
a ``max_count``), every composition whose monoisotopic mass falls within the
mass tolerance is enumerated depth-first. Elements are visited from the
heaviest monoisotopic mass down; at every node the minimum and maximum mass
still obtainable from the not-yet-assigned elements restricts the current
element's integer count window, so counts that cannot possibly land inside
the tolerance window are pruned without recursion.

The enumeration itself is a *streaming* generator
(:func:`iter_compositions`) yielding one reachable composition at a time.
The production search (:func:`search_formulas`) first drains one cheap
stream purely to settle the node/prune counters and the node-limit guard
(no envelopes are built, no hit data retained), then walks a second scored
stream that scores every hit as it is emitted and keeps only the best
``max_results`` candidates in a bounded heap: besides the candidate
currently being scored, neither the composition, the predicted envelope
nor the peak pairing of any displaced candidate is retained, so resident
memory is independent of the number of hits. The pre-pass means an
over-wide request still aborts at the node limit before isotope envelopes
are computed. :func:`enumerate_compositions` stays available as the legacy
list-return compatibility entry, built on the same streaming traversal.

Each hit reuses the existing natural-abundance isotope convolution,
base-peak normalisation and one-to-one peak matching (unmatched peaks
counted at zero intensity); candidates rank by MSE, then absolute mass
error, then the element-symbol-lexicographic formula string, with an
exact-key tie keeping whichever candidate was traversed first. Searches
visiting more than :data:`MAX_VISITED_NODES` nodes abort with
:class:`SearchLimitExceeded` so an overly wide request cannot exhaust the
worker.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Generator

from .enrichment import normalise_base_peak, pair_and_score
from .envelope import Peak, isotope_envelope
from .isotopes import most_abundant_isotope_mass
from .parser import ParsedFormula

# Hard cap on enumeration nodes (the root counts as one). Exceeding it
# aborts the search and the endpoint answers HTTP 422.
MAX_VISITED_NODES = 1_000_000

# Absolute guard (Da) against ordinary floating-point round-off when
# deriving integer count windows and deciding window membership. It only
# ever keeps boundary-neighbour counts (which are re-checked at the leaf);
# at 1e-10 Da it is far smaller than any physically meaningful mass
# tolerance and cannot include an off-by-one-atom neighbour (the lightest
# element, H, shifts the mass by ~1 Da per count).
_ROUND_EPS = 1e-10


class SearchLimitExceeded(Exception):
    """Enumeration exceeded :data:`MAX_VISITED_NODES` visited nodes."""

    def __init__(self, visited_nodes: int, pruned_branches: int) -> None:
        super().__init__(
            f"Formula search visited {visited_nodes} nodes; the limit is "
            f"{MAX_VISITED_NODES}"
        )
        self.visited_nodes = visited_nodes
        self.pruned_branches = pruned_branches


@dataclass(frozen=True)
class ElementSpec:
    """One search element with its inclusive atom-count range."""

    symbol: str
    min_count: int
    max_count: int


@dataclass
class FormulaHit:
    """One within-tolerance composition scored against the observed peaks."""

    formula: str
    composition: dict[str, int]
    monoisotopic_mass: float
    mass_error: float  # predicted monoisotopic mass - target mass (Da)
    mse: float
    pairs: list[tuple[int | None, int | None]]
    predicted: list[Peak]  # base-peak normalised envelope


@dataclass
class FormulaSearchResult:
    """Ranked hits (capped at the requested ``max_results``) plus search stats."""

    hits: list[FormulaHit]
    total_hits: int
    visited_nodes: int
    pruned_branches: int


def formula_string(composition: dict[str, int]) -> str:
    """Canonical formula: element symbols in lexicographic order, counts of
    1 omitted and zero-count elements skipped."""
    return "".join(
        symbol + (str(count) if count > 1 else "")
        for symbol, count in sorted(composition.items())
        if count > 0
    )


@dataclass
class _Frame:
    """Explicit-stack state for one depth-first node.

    ``partial`` is the mass already committed by ancestor counts; the
    feasible count window ``lo..hi`` is computed lazily on first visit so
    node/prune accounting happens exactly once per entered node.
    """

    index: int
    partial: float
    lo: int = 0
    hi: int = -1
    next_count: int = 0
    initialised: bool = False


def iter_compositions(
    target_mass: float,
    mass_tolerance: float,
    elements: list[ElementSpec],
) -> Generator[tuple[tuple[int, ...], float], None, tuple[int, int]]:
    """Streaming branch-and-bound enumeration of compositions inside the
    mass window.

    Elements are visited heaviest-mononisotopic-mass first (ties by symbol)
    so the suffix mass bounds tighten as early as possible. Yields each raw
    hit as ``(counts in visit order, monoisotopic mass)`` as soon as its
    leaf is reached; no hit list is accumulated, so the caller scores each
    composition as it arrives and bounds retained memory independently of
    the total hit count. On clean exhaustion the generator returns
    ``(visited_nodes, pruned_branches)`` (read from the ``value`` attribute
    of the raised :class:`StopIteration`); raises
    :class:`SearchLimitExceeded` past :data:`MAX_VISITED_NODES` visits.
    """
    specs = sorted(
        elements,
        key=lambda e: (-most_abundant_isotope_mass(e.symbol), e.symbol),
    )
    masses = [most_abundant_isotope_mass(s.symbol) for s in specs]
    n = len(specs)

    # suffix_min[i]/suffix_max[i]: smallest/largest mass the elements
    # i..n-1 can still contribute given their count ranges.
    suffix_min = [0.0] * (n + 1)
    suffix_max = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_min[i] = suffix_min[i + 1] + specs[i].min_count * masses[i]
        suffix_max[i] = suffix_max[i + 1] + specs[i].max_count * masses[i]

    visited_nodes = 0
    pruned_branches = 0
    # Counts committed along the current root-to-node path; mirrors the
    # recursive visit's mutable counts list.
    counts: list[int] = []
    # Explicit depth-first stack; frames process counts ascending, which is
    # the same traversal order as the original recursive visit.
    stack: list[_Frame] = [_Frame(0, 0.0)]

    while stack:
        frame = stack[-1]
        index = frame.index

        if not frame.initialised:
            frame.initialised = True
            visited_nodes += 1
            if visited_nodes > MAX_VISITED_NODES:
                raise SearchLimitExceeded(visited_nodes, pruned_branches)

            if index == n:
                # Leaf: re-check membership with a pure round-off guard,
                # since count-window widening at the parent may have reached
                # boundary neighbours.
                if (
                    frame.partial > 0.0
                    and abs(frame.partial - target_mass)
                    <= mass_tolerance + _ROUND_EPS
                ):
                    yield tuple(counts), frame.partial
                stack.pop()
                if counts:
                    counts.pop()
                continue

            mass_i = masses[index]
            # For count c of this element to leave a feasible completion:
            #   partial + c*m_i + suffix_max[i+1] >= target - tolerance
            #   partial + c*m_i + suffix_min[i+1] <= target + tolerance
            low = (
                target_mass
                - mass_tolerance
                - frame.partial
                - suffix_max[index + 1]
            )
            high = (
                target_mass
                + mass_tolerance
                - frame.partial
                - suffix_min[index + 1]
            )
            window_lo = math.ceil((low - _ROUND_EPS) / mass_i)
            window_hi = math.floor((high + _ROUND_EPS) / mass_i)

            frame.lo = max(specs[index].min_count, window_lo)
            frame.hi = min(specs[index].max_count, window_hi)
            frame.next_count = frame.lo
            candidate_count = specs[index].max_count - specs[index].min_count + 1
            visited_count = max(0, frame.hi - frame.lo + 1)
            # Every count value outside the feasible window is one pruned
            # branch (the empty-window case included, via this subtraction).
            pruned_branches += candidate_count - visited_count

        if frame.next_count <= frame.hi:
            count = frame.next_count
            frame.next_count += 1
            counts.append(count)
            stack.append(
                _Frame(index + 1, frame.partial + count * masses[index])
            )
        else:
            # Feasible window exhausted: back out of this node.
            stack.pop()
            if counts:
                counts.pop()

    return visited_nodes, pruned_branches


def enumerate_compositions(
    target_mass: float,
    mass_tolerance: float,
    elements: list[ElementSpec],
) -> tuple[list[tuple[tuple[int, ...], float]], int, int]:
    """Branch-and-bound enumeration of compositions inside the mass window.

    Legacy list-returning compatibility entry over the same streaming
    traversal as :func:`iter_compositions` (which the production search
    uses). Elements are visited heaviest-mononisotopic-mass first (ties by
    symbol). Returns the raw hits as
    ``((counts in visit order, monoisotopic mass), ...)`` in traversal
    order plus the visited-node and pruned-branch counters. Raises
    :class:`SearchLimitExceeded` past :data:`MAX_VISITED_NODES` visits.
    """
    raw_hits: list[tuple[tuple[int, ...], float]] = []
    stream = iter_compositions(target_mass, mass_tolerance, elements)
    try:
        while True:
            raw_hits.append(next(stream))
    except StopIteration as done:
        visited_nodes, pruned_branches = done.value
    return raw_hits, visited_nodes, pruned_branches


@dataclass(frozen=True)
class _InvertedStr:
    """String wrapper whose ordering is lexicographically inverted, so a
    later formula compares as smaller (worse) inside the min-heap."""

    value: str

    def __lt__(self, other: "_InvertedStr") -> bool:
        return self.value > other.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _InvertedStr) and self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass(order=True)
class _RankedHit:
    """Heap entry ordering candidates by the ranking key.

    The bounded structure is a min-heap whose root is the *worst* retained
    candidate. "Worse" means, in order: higher MSE, larger absolute mass
    error or lexicographically later formula; on an identical ranking key
    the later-traversed candidate (larger ``seq``) is worse, so eviction on
    full capacity keeps the first traversal. The numeric keys and the
    traversal index are negated (``-mse`` is smaller when ``mse`` is
    larger); formula goes through :class:`_InvertedStr`.
    """

    neg_mse: float
    neg_abs_error: float
    neg_formula: _InvertedStr
    neg_seq: int
    hit: FormulaHit = field(compare=False)


class _BoundedCandidates:
    """Capacity-limited collection of the best-scoring hits.

    At most ``capacity`` fully scored :class:`FormulaHit` objects (with
    their compositions, envelopes and pairings) ever exist at once: a
    candidate displaced by a better one is removed from the heap and
    dropped immediately. Insertion ranking is
    ``(mse, |mass error|, formula, -traversal index)`` with exact-key ties
    refusing the newcomer, preserving first-traversal order.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._heap: list[_RankedHit] = []

    def offer(
        self,
        hit: FormulaHit,
        seq: int,
    ) -> None:
        # The inverted string makes a lexicographically later formula
        # compare as smaller, i.e. "worse" in the min-heap ordering.
        entry = _RankedHit(
            neg_mse=-hit.mse,
            neg_abs_error=-abs(hit.mass_error),
            neg_formula=_InvertedStr(hit.formula),
            neg_seq=-seq,
            hit=hit,
        )
        if len(self._heap) < self._capacity:
            heapq.heappush(self._heap, entry)
        elif entry > self._heap[0]:
            # Strictly better (or an earlier traversal on the same key) than
            # the current worst: replace it; the evicted hit's envelope and
            # pairs are released as the heap slot is overwritten.
            heapq.heapreplace(self._heap, entry)

    def ranked_hits(self) -> list[FormulaHit]:
        """Return the retained candidates in final ranking order."""
        ordered = sorted(
            self._heap,
            key=lambda e: (
                -e.neg_mse,
                -e.neg_abs_error,
                e.neg_formula.value,
                -e.neg_seq,
            ),
        )
        return [e.hit for e in ordered]


def _score_hit(
    counts: tuple[int, ...],
    mono_mass: float,
    target_mass: float,
    specs: list[ElementSpec],
    observed: list[Peak],
    mass_tolerance: float,
    merge_tolerance: float,
    prune: float,
) -> FormulaHit:
    composition = {
        specs[i].symbol: count for i, count in enumerate(counts) if count > 0
    }
    parsed = ParsedFormula(composition=composition, labels={})
    envelope = isotope_envelope(parsed, merge_tolerance, prune)
    predicted = normalise_base_peak(envelope)
    pairs, mse = pair_and_score(observed, predicted, mass_tolerance)
    return FormulaHit(
        formula=formula_string(composition),
        composition=composition,
        monoisotopic_mass=mono_mass,
        mass_error=mono_mass - target_mass,
        mse=mse,
        pairs=pairs,
        predicted=predicted,
    )


def search_formulas(
    target_mass: float,
    observed: list[Peak],
    elements: list[ElementSpec],
    mass_tolerance: float,
    max_results: int,
    merge_tolerance: float,
    prune: float,
) -> FormulaSearchResult:
    """Stream every within-tolerance composition, scoring each one as it is
    enumerated and retaining only the best ``max_results`` in a bounded
    heap. Hits rank by ascending MSE, then absolute mass error, then
    lexicographic formula; two candidates equal on the whole key keep the
    one traversed first. The full reachable space is still enumerated for
    ``total_hits`` and the node/prune counters, but rejected candidates'
    compositions, envelopes and pairings are dropped on eviction.

    A first cheap traversal (no per-hit scoring, no retained hit data) walks
    the identical tree purely to settle the node/prune accounting and the
    1,000,000-node guard, so an over-limit request aborts before any
    envelope is built -- exactly like the previous enumerate-then-score
    pipeline -- and the response counters match the traversal one-to-one.
    The scored traversal that follows cannot exceed the budget on the same
    inputs."""
    obs = normalise_base_peak(observed)

    specs = sorted(
        elements,
        key=lambda e: (-most_abundant_isotope_mass(e.symbol), e.symbol),
    )

    # Budget probe: drain the stream while retaining nothing (the generator
    # itself only holds the depth path), so an over-wide search raises
    # SearchLimitExceeded before a single isotope envelope is computed.
    probe = iter_compositions(target_mass, mass_tolerance, elements)
    try:
        while True:
            next(probe)
    except StopIteration as done:
        visited_nodes, pruned_branches = done.value

    best = _BoundedCandidates(max_results)
    total_hits = 0
    stream = iter_compositions(target_mass, mass_tolerance, elements)
    while True:
        try:
            counts, mono_mass = next(stream)
        except StopIteration:
            # Same inputs, same traversal as the probe: this pass settles
            # within the node budget by construction.
            break
        # One hit scored at a time; an evicted candidate's envelope and
        # pairs are released as soon as the heap slot is replaced.
        hit = _score_hit(
            counts,
            mono_mass,
            target_mass,
            specs,
            obs,
            mass_tolerance,
            merge_tolerance,
            prune,
        )
        best.offer(hit, total_hits)
        total_hits += 1

    return FormulaSearchResult(
        hits=best.ranked_hits(),
        total_hits=total_hits,
        visited_nodes=visited_nodes,
        pruned_branches=pruned_branches,
    )


__all__ = [
    "MAX_VISITED_NODES",
    "ElementSpec",
    "FormulaHit",
    "FormulaSearchResult",
    "SearchLimitExceeded",
    "enumerate_compositions",
    "formula_string",
    "iter_compositions",
    "search_formulas",
]
