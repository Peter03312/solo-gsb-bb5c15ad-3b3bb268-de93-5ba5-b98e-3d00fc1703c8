"""Centroid conversion of equally-spaced continuous mass/intensity profiles.

The profile endpoints (e.g. :func:`app.profile.simulate_profile`) emit
continuous spectra sampled on a uniform mass grid. Real centroid (stick)
data keeps one point per local maximum instead. This module performs that
conversion:

1. The grid is validated: strictly increasing masses with a constant step
   (every gap must agree with the first gap to within a relative error of
   :data:`STEP_REL_TOLERANCE`), non-negative finite intensities, at least
   one positive intensity.
2. Intensities are normalised to the base peak (100) -- always forming the
   ratio ``intensity / base`` before the *100 scale so values near the
   float maximum cannot overflow -- and points below
   ``relative_threshold`` are removed from consideration.
3. Every flat-topped local-maximum interval (a maximal run of equal
   intensities strictly above its immediate neighbours) becomes one
   candidate whose position is the midpoint of the plateau; the midpoint
   is formed as ``left/2 + right/2`` so two endpoints near the float
   maximum cannot sum to infinity (which would make distinct plateaus
   coincide in the NMS distance and nearest-candidate steps below).
4. Candidates undergo ``min_separation`` non-maximum suppression in
   descending intensity order (equal intensity: lower mass first); a
   candidate closer than ``min_separation`` to an already kept candidate is
   discarded.
5. Every above-threshold point is assigned to the nearest kept candidate
   (ties go to the lower-mass one); each peak's centroid mass is the
   intensity-weighted mean mass of its assigned points. The weighting uses
   the original (unnormalised) intensities but accumulates them scaled by
   the base peak, which keeps every weight in [0, 1] and every running sum
   finite for legal inputs up to the float maximum; the share-of-total
   percentage is the ratio of the same scaled sums. Peaks are returned by
   ascending mass.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass

# Accepted profile length bounds.
MIN_POINTS = 3
MAX_POINTS = 20000
# Every grid gap must agree with the first gap within this relative error.
STEP_REL_TOLERANCE = 1e-6


class CentroidError(ValueError):
    """Locatable validation error for the centroid request.

    Carries a machine-readable ``error`` code, the offending request
    ``field`` and, where applicable, the 0-based ``index`` of the offending
    profile point (``None`` when not index-specific).
    """

    def __init__(
        self, error: str, field: str, message: str, index: int | None = None
    ) -> None:
        super().__init__(message)
        self.error = error
        self.field = field
        self.message = message
        self.index = index


@dataclass
class CentroidPeak:
    mass: float
    intensity: float
    total_intensity_percent: float


@dataclass
class CentroidResult:
    point_count: int
    relative_threshold: float
    min_separation: float
    candidate_count: int
    peaks: list[CentroidPeak]


def _midpoint(left: float, right: float) -> float:
    """Overflow-free mean of two finite masses.

    ``(left + right) / 2`` overflows to ``inf`` when both endpoints sit
    near the float maximum (two legal plateaus at ~1.7e308 can both carry
    inf positions, after which NMS measures ``abs(inf - inf) == NaN`` and
    treats them as coincident no matter how far apart they really are, and
    nearest-candidate assignment collapses all points onto one winner).
    Halving each endpoint before adding bounds both summands at half a
    finite value, so the result is always finite for finite inputs; the
    rounding error is the same single-ULP bound as the direct mean.
    """
    return left / 2.0 + right / 2.0


def _find_candidates(
    masses: list[float],
    intensities: list[float],
    kept: list[bool],
    base: float,
) -> list[tuple[float, float, int, int]]:
    """Flat-topped local maxima of the thresholded profile.

    Returns tuples ``(position, normalised_intensity, start, end)`` where
    ``start``/``end`` delimit the plateau (inclusive grid indices) and
    ``position`` is the plateau midpoint. A plateau is a maximal run of
    consecutive above-threshold points with identical intensity; it is a
    local maximum when both immediate neighbours (missing edges count as
    infinitely low) are strictly lower. Filtered-out neighbours are lower
    by construction, and equal kept neighbours would have extended the run.
    """
    n = len(masses)
    candidates: list[tuple[float, float, int, int]] = []
    i = 0
    while i < n:
        if not kept[i]:
            i += 1
            continue
        level = intensities[i]
        j = i
        while j + 1 < n and kept[j + 1] and intensities[j + 1] == level:
            j += 1
        left_lower = i == 0 or intensities[i - 1] < level
        right_lower = j == n - 1 or intensities[j + 1] < level
        if left_lower and right_lower:
            # Divide by the base peak before the *100 scale: a level near
            # the float maximum would make 100*level overflow before the
            # same-magnitude division rescaled it. The midpoint is formed
            # without adding the endpoints, since two masses near the float
            # maximum would sum to infinity and collapse the plateau's
            # position onto inf for sorting, NMS and nearest-point assignment.
            candidates.append(
                (_midpoint(masses[i], masses[j]), 100.0 * (level / base), i, j)
            )
        i = j + 1
    return candidates


def _centroid_mass(
    members: list[tuple[float, float]],
    weighted_mass_sum: float,
    weight_sum: float,
) -> float:
    """Intensity-weighted mean mass of one peak's assigned points.

    ``members`` holds ``(mass, scaled_intensity)`` pairs with intensities
    already divided by the profile base peak, so every weight lies in
    [0, 1]; the running sums are then bounded by the point count and stay
    finite whenever the masses themselves are moderate. The single rescue
    path below is only reached when ``weighted_mass_sum`` still overflowed
    (i.e. the masses themselves approach the float maximum): masses are
    additionally divided by the largest member mass, which bounds every
    summand at 1 while preserving the same ratio.
    """
    if math.isfinite(weighted_mass_sum):
        return weighted_mass_sum / weight_sum
    mass_scale = max(abs(mass) for mass, _ in members)
    scaled_numerator = sum(
        weight * (mass / mass_scale) for mass, weight in members
    )
    scaled_denominator = sum(weight for _, weight in members)
    return mass_scale * (scaled_numerator / scaled_denominator)


def centroid_profile(
    points: list[tuple[float, float]],
    relative_threshold: float,
    min_separation: float,
) -> CentroidResult:
    """Convert a uniform-grid continuous profile into centroid peaks.

    See the module docstring for the full algorithm. Raises
    :class:`CentroidError` for an illegal mass grid (non-increasing masses
    or gaps whose relative deviation from the first gap exceeds
    :data:`STEP_REL_TOLERANCE`), an all-zero profile, or a threshold
    filtering that leaves no candidate.
    """
    n = len(points)
    if not MIN_POINTS <= n <= MAX_POINTS:
        raise CentroidError(
            "invalid_profile",
            "profile",
            f"profile must contain between {MIN_POINTS} and {MAX_POINTS} "
            f"points, got {n}",
        )

    masses = [m for m, _ in points]
    intensities = [value for _, value in points]

    for i, (mass, value) in enumerate(points):
        if not math.isfinite(mass):
            raise CentroidError(
                "invalid_mass_grid",
                "profile",
                f"mass at index {i} must be a finite number, got {mass}",
                i,
            )
        if not math.isfinite(value):
            raise CentroidError(
                "invalid_profile",
                "profile",
                f"intensity at index {i} must be a finite number, got {value}",
                i,
            )
        if value < 0.0:
            raise CentroidError(
                "invalid_profile",
                "profile",
                f"intensity at index {i} must be non-negative, got {value}",
                i,
            )

    # Strictly increasing masses on a uniform grid; the first gap is the
    # reference spacing every later gap is compared against.
    reference_step = masses[1] - masses[0]
    if reference_step <= 0.0:
        raise CentroidError(
            "invalid_mass_grid",
            "profile",
            f"Masses must be strictly increasing, but the mass at index 1 "
            f"({masses[1]}) does not exceed the mass at index 0 ({masses[0]})",
            1,
        )
    for i in range(2, n):
        gap = masses[i] - masses[i - 1]
        if gap <= 0.0:
            raise CentroidError(
                "invalid_mass_grid",
                "profile",
                f"Masses must be strictly increasing, but the mass at "
                f"index {i} ({masses[i]}) does not exceed the mass at "
                f"index {i - 1} ({masses[i - 1]})",
                i,
            )
        rel_error = abs(gap - reference_step) / reference_step
        if rel_error > STEP_REL_TOLERANCE:
            raise CentroidError(
                "invalid_mass_grid",
                "profile",
                f"Grid spacing must be uniform within a relative error of "
                f"{STEP_REL_TOLERANCE:g}; gap at index {i} is {gap:g} but "
                f"the reference spacing is {reference_step:g} "
                f"(relative error {rel_error:g})",
                i,
            )

    base = max(intensities)
    if base <= 0.0:
        raise CentroidError(
            "zero_profile",
            "profile",
            "All profile intensities are zero; at least one must be positive",
        )

    # Keep points at or above the base-peak-relative threshold. The ratio
    # is formed before the *100 scale: a value near the float maximum
    # (up to ~1.8e308) would overflow in 100*value before the same-order
    # division by base rescaled it. The tiny nudge protects the 100.0
    # base-peak level from binary round-off in the normalisation
    # (100*x/x need not be exactly 100.0).
    kept = [
        100.0 * (value / base) + 1e-9 >= relative_threshold
        for value in intensities
    ]

    candidates = _find_candidates(masses, intensities, kept, base)
    if not candidates:
        raise CentroidError(
            "no_candidates",
            "relative_threshold",
            "No local maximum remains above the relative threshold; lower "
            "relative_threshold",
        )

    # Non-maximum suppression: strongest first, equal strength prefers the
    # lower mass. A candidate strictly closer than min_separation to any
    # already kept candidate is dropped; exactly min_separation is allowed.
    candidates.sort(key=lambda c: (-c[1], c[0]))
    winners: list[tuple[float, float, int, int]] = []
    for candidate in candidates:
        position = candidate[0]
        # Exactly min_separation (within binary round-off) is allowed.
        if all(
            abs(position - kept_candidate[0]) >= min_separation * (1.0 - 1e-12)
            for kept_candidate in winners
        ):
            winners.append(candidate)

    # Assign every above-threshold point to the nearest winner; equidistant
    # points go to the lower-mass (left) candidate. Intensities are scaled
    # by the base peak as they accumulate: each weight is then in [0, 1] and
    # neither the weight sums nor the mass-weighted products can overflow
    # for original intensities as large as the float maximum.
    winners_by_mass = sorted(winners, key=lambda c: c[0])
    positions = [c[0] for c in winners_by_mass]
    weight_sum = [0.0] * len(winners_by_mass)
    weighted_mass_sum = [0.0] * len(winners_by_mass)
    members: list[list[tuple[float, float]]] = [
        [] for _ in winners_by_mass
    ]
    for k in range(n):
        if not kept[k]:
            continue
        mass = masses[k]
        right = bisect_left(positions, mass)
        if right == 0:
            owner = 0
        elif right == len(positions):
            owner = right - 1
        else:
            left = right - 1
            owner = left if mass - positions[left] <= positions[right] - mass else right
        weight = intensities[k] / base
        weight_sum[owner] += weight
        weighted_mass_sum[owner] += weight * mass
        members[owner].append((mass, weight))

    # Scaled sums total the same proportion as the original intensity sum,
    # so the share-of-total percentage is unaffected by the rescaling (and
    # its numerator/denominator can no longer overflow to inf/inf = NaN).
    total_assigned = sum(weight_sum)
    peaks: list[CentroidPeak] = []
    for owner, candidate in enumerate(winners_by_mass):
        _, level, _, _ = candidate
        peaks.append(
            CentroidPeak(
                mass=_centroid_mass(
                    members[owner],
                    weighted_mass_sum[owner],
                    weight_sum[owner],
                ),
                intensity=level,
                total_intensity_percent=100.0 * weight_sum[owner] / total_assigned,
            )
        )
    peaks.sort(key=lambda peak: peak.mass)

    return CentroidResult(
        point_count=n,
        relative_threshold=relative_threshold,
        min_separation=min_separation,
        candidate_count=len(candidates),
        peaks=peaks,
    )
