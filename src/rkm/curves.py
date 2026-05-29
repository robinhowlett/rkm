"""Velocity curve fitting — linear deceleration model.

Model: v(d) = v0 - decay_rate × d

Where:
  v0 = initial velocity at race start (ft/s)
  decay_rate = velocity lost per foot of distance (ft/s per ft)

Over racing distances (1000-7000 ft), deceleration is effectively linear.
The two parameters directly capture:
  - v0: raw speed / acceleration ability (anaerobic)
  - decay_rate: stamina / energy depletion rate (aerobic efficiency)
"""

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

MIN_VELOCITY_POINTS = 12
MAX_BEATEN_LENGTHS = 30
MIN_VELOCITY = 30.0  # ft/s — anything below is not a real performance
MAX_VELOCITY = 70.0  # ft/s — anything above is a data error

# A positive slope means the horse accelerates over distance — physically
# implausible at career scale. This module REJECTS such fits (returns None);
# form.py CLAMPS to flat instead. The asymmetry is intentional: a career
# curve fit on many races with a positive slope strongly indicates bad
# data; a recent-form curve fit on few weighted observations is more
# likely to show spurious positive slope from noise, where rejecting
# would discard usable form coverage. Both modules share this threshold.
POSITIVE_SLOPE_CLAMP_THRESHOLD = 0.001


@dataclass
class CurveFit:
    v0: float           # initial velocity (ft/s)
    decay_rate: float   # velocity loss per 1000ft (ft/s per 1000ft)
    n_observations: int
    n_races: int
    residual_std: float
    first_race: str | None
    last_race: str | None


def compute_segment_velocities(race_points: list[dict]) -> list[tuple[float, float]]:
    """Convert (feet, millis) pairs from a single race into (distance_midpoint, velocity) pairs."""
    sorted_points = sorted(race_points, key=lambda p: p["feet"])
    velocities = []

    for i in range(len(sorted_points) - 1):
        d1, t1 = sorted_points[i]["feet"], sorted_points[i]["millis"]
        d2, t2 = sorted_points[i + 1]["feet"], sorted_points[i + 1]["millis"]

        dt = t2 - t1
        dd = d2 - d1

        if dt <= 0 or dd <= 0:
            continue

        v = dd / dt * 1000.0
        midpoint = (d1 + d2) / 2.0

        if v < MIN_VELOCITY or v > MAX_VELOCITY:
            continue

        velocities.append((midpoint, v))

    return velocities


def fit_horse_curve(all_race_points: list[list[dict]],
                    beaten_lengths: list[float | None] = None) -> CurveFit | None:
    """Fit linear deceleration model to a horse's career observations on one surface.

    Returns CurveFit or None if insufficient data.
    """
    distances = []
    velocities = []
    race_dates = []
    n_races_used = 0

    for i, race_points in enumerate(all_race_points):
        if beaten_lengths and i < len(beaten_lengths):
            if beaten_lengths[i] is not None and beaten_lengths[i] > MAX_BEATEN_LENGTHS:
                continue

        seg_vels = compute_segment_velocities(race_points)
        if not seg_vels:
            continue

        for d, v in seg_vels:
            distances.append(d)
            velocities.append(v)

        n_races_used += 1
        if race_points:
            race_dates.append(race_points[0].get("race_date"))

    if len(distances) < MIN_VELOCITY_POINTS:
        return None

    d_arr = np.array(distances)
    v_arr = np.array(velocities)

    # Robust linear fit using numpy polyfit (degree 1)
    # For robustness: iteratively reweight (simple IRLS with Huber weights)
    coeffs = np.polyfit(d_arr, v_arr, 1)
    predicted = np.polyval(coeffs, d_arr)
    residuals = v_arr - predicted

    # One round of Huber reweighting for robustness
    mad = np.median(np.abs(residuals - np.median(residuals)))
    sigma = mad * 1.4826  # robust std estimate
    if sigma > 0:
        huber_delta = 1.5 * sigma
        weights = np.where(np.abs(residuals) <= huber_delta, 1.0,
                           huber_delta / np.abs(residuals))
        coeffs = np.polyfit(d_arr, v_arr, 1, w=weights)
        predicted = np.polyval(coeffs, d_arr)
        residuals = v_arr - predicted

    slope = coeffs[0]       # negative = deceleration
    intercept = coeffs[1]   # v0

    # Sanity: v0 should be physically plausible and slope should be non-positive
    if intercept < 40 or intercept > 85:  # 85 ft/s ≈ 58 mph, beyond any horse
        return None
    if slope > POSITIVE_SLOPE_CLAMP_THRESHOLD:
        # Horse accelerating overall — unusual at career scale, likely bad
        # data or sprinter quirk. Reject the fit (form.py clamps instead;
        # see POSITIVE_SLOPE_CLAMP_THRESHOLD docstring for the asymmetry).
        return None

    decay_rate = -slope * 1000  # convert to ft/s per 1000ft (positive number)
    residual_std = float(np.std(residuals))

    valid_dates = [d for d in race_dates if d is not None]

    return CurveFit(
        v0=round(float(intercept), 2),
        decay_rate=round(float(decay_rate), 4),
        n_observations=len(distances),
        n_races=n_races_used,
        residual_std=round(residual_std, 3),
        first_race=str(min(valid_dates)) if valid_dates else None,
        last_race=str(max(valid_dates)) if valid_dates else None,
    )
