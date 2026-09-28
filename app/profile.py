"""Instrument-profile simulation for neutral-molecule isotope stick spectra.

The isotope envelope produced by :mod:`app.envelope` is a list of infinitely
narrow (mass, intensity) sticks. A real instrument broadens every stick into a
Gaussian whose full width at half maximum is set by its resolving power::

    FWHM(m) = m / resolving_power
    sigma   = FWHM / (2 * sqrt(2 * ln 2))

Each stick's Gaussian kernel is sampled onto one uniform mass grid within
+-4 sigma of the stick mass (the range outside a requested window is cut, but
the contribution inside is kept); all kernels add into the same grid, whose
profile is then base-peak normalised to 100.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .envelope import Peak

# Gaussian FWHM-to-sigma factor: sigma = FWHM / (2 sqrt(2 ln 2)).
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
# Kernels extend this many standard deviations either side of a stick.
SIGMA_SUPPORT = 4.0
# Largest allowed uniform grid.
MAX_GRID_POINTS = 20000


class ProfileParameterError(ValueError):
    """Locatable validation error for the profile request parameters.

    Carries a machine-readable ``error`` code and the offending request
    ``field`` (``"mass_step"`` or ``"mass_min"``).
    """

    def __init__(self, error: str, field: str, message: str) -> None:
        super().__init__(message)
        self.error = error
        self.field = field
        self.message = message


@dataclass
class ProfileResult:
    mass_min: float
    mass_max: float
    mass_step: float
    point_count: int
    profile: list[Peak]


def _stick_support(
    sticks: list[Peak], resolving_power: float
) -> tuple[float, float]:
    """Union of every stick's +-4 sigma window."""
    lower = min(m - SIGMA_SUPPORT * FWHM_TO_SIGMA * m / resolving_power
                for m, _ in sticks)
    upper = max(m + SIGMA_SUPPORT * FWHM_TO_SIGMA * m / resolving_power
                for m, _ in sticks)
    return lower, upper


def simulate_profile(
    sticks: list[Peak],
    resolving_power: float,
    mass_step: float,
    mass_min: float | None = None,
    mass_max: float | None = None,
) -> ProfileResult:
    """Broaden stick peaks into a Gaussian instrument profile on a grid.

    Each stick at mass ``m`` contributes a Gaussian with
    ``FWHM = m / resolving_power`` at every grid point within ``+-4 sigma``
    of ``m``. Kernels are clipped to the ``[mass_min, mass_max]`` window but
    their in-window contribution is kept unrenormalised. When either bound is
    omitted it defaults to the corresponding edge of the union of all
    sticks' ``+-4 sigma`` support. The summed grid is normalised so its
    highest intensity is 100.

    Raises :class:`ProfileParameterError` for a non-positive step, an empty
    or backwards range, a grid larger than :data:`MAX_GRID_POINTS`, or a
    window on which no kernel reaches any grid point.
    """
    if mass_step <= 0.0:
        raise ProfileParameterError(
            "invalid_mass_step",
            "mass_step",
            f"mass_step must be greater than zero, got {mass_step}",
        )

    auto_min, auto_max = _stick_support(sticks, resolving_power)
    lo = auto_min if mass_min is None else mass_min
    hi = auto_max if mass_max is None else mass_max
    if not hi > lo:
        raise ProfileParameterError(
            "invalid_mass_range",
            "mass_min",
            f"mass range must be a non-empty interval with mass_min < "
            f"mass_max, got mass_min={lo} and mass_max={hi}",
        )

    span = hi - lo
    ratio = span / mass_step
    # point_count = floor(ratio) + 1 (with a float-tolerance nudge), so the
    # largest allowed ratio is MAX_GRID_POINTS - 1; a non-finite ratio (e.g.
    # a subnormal step) is rejected up front so floor() cannot overflow.
    if not math.isfinite(ratio) or ratio > MAX_GRID_POINTS - 1 + 1e-9:
        raise ProfileParameterError(
            "grid_too_large",
            "mass_step",
            f"Mass grid would contain more than {MAX_GRID_POINTS} points "
            f"(at most {MAX_GRID_POINTS} are allowed; widen mass_step or "
            f"narrow the mass range)",
        )
    point_count = int(math.floor(ratio + 1e-9)) + 1

    grid = [lo + k * mass_step for k in range(point_count)]
    intensities = [0.0] * point_count
    for mass, intensity in sticks:
        fwhm = mass / resolving_power
        sigma = FWHM_TO_SIGMA * fwhm
        radius = SIGMA_SUPPORT * sigma
        # Clamp to a finite range before ceil/floor: a subnormal step makes
        # the raw ratio infinite (which int() rejects) even when the stick
        # lies inside the window.
        lower_ratio = max(-1e300, (mass - radius - lo) / mass_step)
        upper_ratio = min(1e300, (mass + radius - lo) / mass_step)
        k_lo = max(0, int(math.ceil(lower_ratio - 1e-9)))
        k_hi = min(point_count - 1, int(math.floor(upper_ratio + 1e-9)))
        # Stick falls entirely outside the grid window.
        if k_lo > k_hi:
            continue
        denom = 2.0 * sigma * sigma
        for k in range(k_lo, k_hi + 1):
            delta = grid[k] - mass
            intensities[k] += intensity * math.exp(-(delta * delta) / denom)

    base = max(intensities)
    if base <= 0.0:
        raise ProfileParameterError(
            "no_profile_points",
            "mass_min",
            "No stick kernel reaches any grid point in the requested mass "
            "range; widen the range or use a smaller mass_step",
        )

    profile = [(grid[k], 100.0 * intensities[k] / base) for k in range(point_count)]
    return ProfileResult(lo, hi, mass_step, point_count, profile)
