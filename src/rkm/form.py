"""Time-weighted velocity curves for current form estimation."""

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

DECAY_FACTOR = 0.90  # per 30 days — half-life ≈ 6.6 months
MIN_PRIOR_RACES = 1  # lowered from 2 → 1 for broader coverage (Phase 3)
MIN_VELOCITY = 30.0
MAX_VELOCITY = 85.0


@dataclass
class FormSnapshot:
    current_v0: float
    current_decay: float
    career_v0: float
    career_decay: float
    v0_trend: float
    n_recent_races: int  # races in last 180 days
    days_since_last: int


def compute_form_at_date(prior_observations: list[dict], race_date, career_v0: float,
                         career_decay: float) -> FormSnapshot | None:
    """Compute time-weighted curve from prior observations.

    Args:
        prior_observations: list of {race_date, distances: [float], velocities: [float]}
            Each entry represents one prior race's velocity points.
        race_date: the date we're computing form FOR (entering this race)
        career_v0, career_decay: the static career curve parameters

    Returns FormSnapshot or None if insufficient data.
    """
    if len(prior_observations) < MIN_PRIOR_RACES:
        return None

    all_distances = []
    all_velocities = []
    all_weights = []
    n_recent = 0
    days_since_last = None

    for obs in prior_observations:
        obs_date = obs["race_date"]
        days_ago = (race_date - obs_date).days
        if days_ago <= 0:
            continue  # can't use current or future races

        if days_since_last is None or days_ago < days_since_last:
            days_since_last = days_ago

        if days_ago <= 180:
            n_recent += 1

        # Time weight: exponential decay
        weight = DECAY_FACTOR ** (days_ago / 30.0)

        for d, v in zip(obs["distances"], obs["velocities"]):
            if v < MIN_VELOCITY or v > MAX_VELOCITY:
                continue
            all_distances.append(d)
            all_velocities.append(v)
            all_weights.append(weight)

    if len(all_distances) < 4:  # lowered from 8 → 4 (1 race has 4-6 points)
        return None

    d_arr = np.array(all_distances)
    v_arr = np.array(all_velocities)
    w_arr = np.array(all_weights)

    # Weighted linear regression
    try:
        coeffs = np.polyfit(d_arr, v_arr, 1, w=w_arr)
    except (np.linalg.LinAlgError, ValueError):
        return None

    slope = coeffs[0]
    intercept = coeffs[1]

    # Sanity checks
    if intercept < 40 or intercept > 85:
        return None
    if slope > 0.001:  # shouldn't be accelerating overall
        slope = 0.0  # clamp to flat

    current_v0 = round(float(intercept), 2)
    current_decay = round(float(-slope * 1000), 4)  # per 1000ft, positive
    v0_trend = round(current_v0 - career_v0, 2)

    return FormSnapshot(
        current_v0=current_v0,
        current_decay=current_decay,
        career_v0=career_v0,
        career_decay=career_decay,
        v0_trend=v0_trend,
        n_recent_races=n_recent,
        days_since_last=days_since_last if days_since_last is not None else 9999,
    )
