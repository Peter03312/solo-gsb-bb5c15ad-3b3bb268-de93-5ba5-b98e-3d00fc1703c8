"""De novo detection of unknown isotope clusters in centroid m/z peaks.

Unlike the formula-driven endpoints, this module needs no molecular
formula: an isotope cluster of charge *z* is recognised purely by the
regular ~1.003354835 / *z* m/z spacing of its members (the 13C-12C neutron
mass difference split over *z* charges). For every charge in a requested
1..10 range, candidate chains are grown from every peak:

* adjacent members sit one or two nominal isotope steps apart
  (``abs(measured_gap - d * step) <= spacing_tolerance`` for ``d`` in 1, 2);
* a two-step jump means exactly one isotope position is missing, and each
  candidate chain may contain at most one such gap.

Observed intensities are normalised to the spectrum base peak (100). A
candidate's score is the sum of its member intensities minus 10 per missing
isotope position. A weighted-interval dynamic program then chooses the
mutually index-non-overlapping candidates with the highest total score
(ties resolve to more members, then the lexicographically smaller
monoisotopic-m/z tuple, then the smaller charge tuple). Peaks left unused
by every chosen cluster are reported as unassigned.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

# Mass difference 13C-12C (Da); isotope peaks of a z-fold charged ion are
# spaced by NEUTRON_STEP_DA / z along the m/z axis.
NEUTRON_STEP_DA = 1.003354835
# Missing isotope positions cost this many normalised intensity points.
MISSING_PENALTY = 10.0
# Scores within this absolute distance count as equal (float round-off guard).
SCORE_EPS = 1e-9
# Safety stop for pathological enumeration (e.g. an absurdly large spacing
# tolerance that turns every peak into a reachable successor).
MAX_CANDIDATES = 100_000

ChargeRange = tuple[int, int]


class IsotopeClusterError(ValueError):
    """Validation error for the isotope-cluster request.

    Carries a machine-readable ``error`` code, the offending request
    ``field`` and a human-readable ``message``.
    """

    def __init__(self, error: str, field: str, message: str) -> None:
        super().__init__(message)
        self.error = error
        self.field = field
        self.message = message


@dataclass(frozen=True)
class ClusterCandidate:
    """One isotope cluster found at a single charge state."""

    charge: int
    # 0-based indices into the input peak list, m/z increasing.
    members: tuple[int, ...]
    # Nominal isotope positions (0-based offsets from the monoisotopic
    # member) that are absent because of a two-step gap.
    missing_positions: tuple[int, ...]
    monoisotopic_mz: float
    score: float


@dataclass(frozen=True)
class IsotopeCluster:
    charge: int
    monoisotopic_mz: float
    score: float
    member_indices: tuple[int, ...]
    missing_positions: tuple[int, ...]


@dataclass(frozen=True)
class ClusterDetectionResult:
    charge_min: int
    charge_max: int
    spacing_tolerance: float
    min_peaks: int
    clusters: tuple[IsotopeCluster, ...]
    # 0-based indices of peaks used by no chosen cluster.
    unassigned_indices: tuple[int, ...]


def _grow_candidates_for_charge(
    mz: list[float],
    intensity: list[float],
    charge: int,
    spacing_tolerance: float,
    min_peaks: int,
    budget: list[int],
) -> list[ClusterCandidate]:
    """Enumerate every valid chain at ``charge`` via depth-first growth.

    Each starting peak is a potential monoisotopic member. From the current
    tail a successor may be one or two nominal steps away (within
    ``spacing_tolerance``); a chain records itself once it has at least
    ``min_peaks`` members and may carry at most one two-step gap. Chains that
    only differ by the route taken to the same member sequence are merged
    (the missing position is then identical).
    """
    step = NEUTRON_STEP_DA / charge
    candidates: list[ClusterCandidate] = []
    # Keyed by (members, missing_positions) to defend against routes with
    # identical membership entering the same window twice.
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    n = len(mz)

    def check_budget() -> None:
        if budget[0] < 0:
            raise IsotopeClusterError(
                "too_many_candidates",
                "spacing_tolerance",
                f"Isotope-cluster enumeration produced more than "
                f"{MAX_CANDIDATES:,} candidates; reduce spacing_tolerance "
                f"or increase min_peaks",
            )

    def record(
        chain_members: list[int], missing: list[int], intensity_sum: float
    ) -> None:
        key = (tuple(chain_members), tuple(missing))
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            ClusterCandidate(
                charge=charge,
                members=key[0],
                missing_positions=key[1],
                monoisotopic_mz=mz[chain_members[0]],
                score=intensity_sum - MISSING_PENALTY * len(missing),
            )
        )
        budget[0] -= 1
        check_budget()

    def dfs(
        tail: int,
        members: list[int],
        missing: list[int],
        gap_used: bool,
        intensity_sum: float,
        nominal: int,
    ) -> None:
        check_budget()
        if len(members) >= min_peaks:
            record(members, missing, intensity_sum)
        mz_tail = mz[tail]
        deltas = (1, 2) if not gap_used else (1,)
        for delta in deltas:
            expected = mz_tail + delta * step
            lo = bisect_left(mz, expected - spacing_tolerance, lo=tail + 1)
            hi = bisect_right(mz, expected + spacing_tolerance, lo=lo)
            for nxt in range(lo, hi):
                members.append(nxt)
                if delta == 2:
                    missing.append(nominal + 1)
                dfs(
                    nxt,
                    members,
                    missing,
                    gap_used or delta == 2,
                    intensity_sum + intensity[nxt],
                    nominal + delta,
                )
                members.pop()
                if delta == 2:
                    missing.pop()

    for start in range(n):
        dfs(start, [start], [], False, intensity[start], 0)
    return candidates


def _pair_ties(score_a: float, score_b: float) -> bool:
    return abs(score_a - score_b) <= SCORE_EPS


def _select_clusters(
    candidates: list[ClusterCandidate], peak_count: int
) -> list[ClusterCandidate]:
    """Choose the optimal mutually non-overlapping candidate set.

    A candidate occupies the inclusive index interval ``[first, last]``; two
    candidates are compatible when ``last_a < first_b``. The optimisation runs
    in two phases over suffixes:

    1. ``f[p]`` is the maximum ``(score, member count)`` pair attainable with
       clusters whose first member is at index ``>= p`` (taking a candidate
       starting at ``p`` continues at ``last + 1``). Scalar additive keys
       make this recurrence exact.
    2. ``g[p]`` is the lexicographically smallest *full* cluster sequence
       attaining ``f[p]``: whenever any candidate starting at ``p`` reaches
       the optimum, taking one is lexicographically forced (its monoisotopic
       m/z is smaller than every later start), and ties between such takes
       are settled by comparing the complete suffix sequences
       (monoisotopic-m/z tuples, then charge tuples). Comparing finished
       suffix tuples wholesale avoids the prefix-order flip that an
       incremental single-state DP would suffer.
    """
    by_first: list[list[ClusterCandidate]] = [[] for _ in range(peak_count)]
    for cand in candidates:
        by_first[cand.members[0]].append(cand)

    # f[p] = optimal (score, member_count) from suffix p; sentinel f[n]=(0,0).
    best_score = [0.0] * (peak_count + 1)
    best_members = [0] * (peak_count + 1)
    # g[p] = lexicographically smallest optimum of suffix p as
    # (chosen candidates, monoisotopic-m/z tuple, charge tuple).
    chosen: list[tuple[ClusterCandidate, ...]] = [()] * (peak_count + 1)
    mono_tuple: list[tuple[float, ...]] = [()] * (peak_count + 1)
    charge_tuple: list[tuple[int, ...]] = [()] * (peak_count + 1)

    for p in range(peak_count - 1, -1, -1):
        # Option 1: leave peak p free of a cluster starting there.
        opt_score = best_score[p + 1]
        opt_members = best_members[p + 1]
        # Option 2: take a candidate whose first member is p.
        feasible_takes: list[ClusterCandidate] = []
        for cand in by_first[p]:
            q = cand.members[-1] + 1
            score = cand.score + best_score[q]
            members = len(cand.members) + best_members[q]
            if score > opt_score + SCORE_EPS:
                opt_score = score
                opt_members = members
                feasible_takes = [cand]
            elif _pair_ties(score, opt_score):
                if members > opt_members:
                    opt_members = members
                    feasible_takes = [cand]
                elif members == opt_members:
                    feasible_takes.append(cand)

        best_score[p] = opt_score
        best_members[p] = opt_members

        if not feasible_takes:
            # Skipping p (strictly) wins; carry the suffix optimum over.
            chosen[p] = chosen[p + 1]
            mono_tuple[p] = mono_tuple[p + 1]
            charge_tuple[p] = charge_tuple[p + 1]
            continue

        # A take reaches the optimum and always beats the skip option on the
        # monoisotopic tuple (mz[p] is smaller than any later start's m/z);
        # the skip option could only tie here with zero more members, which a
        # take with >= min_peaks members cannot reproduce.
        def take_key(cand: ClusterCandidate):
            q = cand.members[-1] + 1
            return mono_tuple[q], (cand.charge,) + charge_tuple[q]

        winner = min(feasible_takes, key=take_key)
        q = winner.members[-1] + 1
        chosen[p] = (winner,) + chosen[q]
        mono_tuple[p] = (winner.monoisotopic_mz,) + mono_tuple[q]
        charge_tuple[p] = (winner.charge,) + charge_tuple[q]

    return list(chosen[0])


def detect_isotope_clusters(
    peaks: list[tuple[float, float]],
    charge_range: ChargeRange,
    spacing_tolerance: float,
    min_peaks: int,
) -> ClusterDetectionResult:
    """Detect non-overlapping isotope clusters in sorted centroid peaks.

    ``peaks`` must be strictly m/z-increasing with non-negative intensities
    (validated by the endpoint before calling). Returns the
    score-optimal set of clusters plus the indices of unassigned peaks.
    """
    charge_min, charge_max = charge_range
    mz = [m for m, _ in peaks]
    raw_intensity = [i for _, i in peaks]
    base = max(raw_intensity)
    # An all-zero spectrum is legal: leave intensities at zero so every
    # candidate score is zero (clusters can still be found by spacing).
    intensity = [100.0 * i / base if base > 0.0 else 0.0 for i in raw_intensity]

    budget = [MAX_CANDIDATES]
    candidates: list[ClusterCandidate] = []
    for charge in range(charge_min, charge_max + 1):
        candidates.extend(
            _grow_candidates_for_charge(
                mz, intensity, charge, spacing_tolerance, min_peaks, budget
            )
        )

    chosen = _select_clusters(candidates, len(peaks))
    # Report clusters in spectrum (first member index) order.
    chosen.sort(
        key=lambda c: (
            c.members[0],
            c.monoisotopic_mz,
            c.charge,
        )
    )

    used = {idx for cand in chosen for idx in cand.members}
    unassigned = tuple(i for i in range(len(peaks)) if i not in used)

    clusters = tuple(
        IsotopeCluster(
            charge=cand.charge,
            monoisotopic_mz=cand.monoisotopic_mz,
            score=cand.score,
            member_indices=cand.members,
            missing_positions=cand.missing_positions,
        )
        for cand in chosen
    )
    return ClusterDetectionResult(
        charge_min=charge_min,
        charge_max=charge_max,
        spacing_tolerance=spacing_tolerance,
        min_peaks=min_peaks,
        clusters=clusters,
        unassigned_indices=unassigned,
    )
