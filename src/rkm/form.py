"""Time-weighted velocity curves for current form estimation."""

import logging
from dataclasses import dataclass

import numpy as np

from rkm.curves import POSITIVE_SLOPE_CLAMP_THRESHOLD

log = logging.getLogger(__name__)

DECAY_FACTOR = 0.90  # per 30 days — half-life ≈ 6.6 months
MIN_PRIOR_RACES = 1  # lowered from 2 → 1 for broader coverage (Phase 3)
MIN_VELOCITY = 30.0
MAX_VELOCITY = 70.0  # observation filter — above 70 ft/s is a timing artifact
                     # (empirical: 0.016% of indiv_fractionals exceed, max
                     # observed 2,671 ft/s, clearly bad data). Aligned with
                     # curves.py:MAX_VELOCITY per RKM #6.


@dataclass
class FormSnapshot:
    current_v0: float
    current_decay: float
    career_v0: float
    career_decay: float
    v0_trend: float
    n_recent_races: int  # races in last 180 days
    days_since_last: int


def compute_form_at_date(prior_observations: list[dict], race_date) -> FormSnapshot | None:
    """Compute trailing-career and recent-weighted curves from prior observations.

    Both fits use ONLY prior_observations (races strictly before race_date).
    They differ only in their weighting profile:

      * current_v0 / current_decay: weighted polyfit, exponential decay by
        days-ago (DECAY_FACTOR ** days/30). Reflects recent form.
      * career_v0 / career_decay: unweighted polyfit on the same data.
        Reflects the horse's career-to-date.

    v0_trend = current_v0 - career_v0 captures "recent vs career-to-date"
    rather than the prior implementation's "recent vs full-career" — which
    leaked future races into the baseline. See RKM-T1.4 for context.

    Args:
        prior_observations: list of {race_date, distances: [float], velocities: [float]}
            Each entry represents one prior race's velocity points.
        race_date: the date we're computing form FOR (entering this race)

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

    # Recent-weighted regression (current form).
    try:
        recent_coeffs = np.polyfit(d_arr, v_arr, 1, w=w_arr)
    except (np.linalg.LinAlgError, ValueError):
        return None

    # Unweighted regression on the same observations (career-to-date).
    # Same data, only the weighting differs — the trailing aggregate the
    # RKM-T1.4 audit recommended. Strictly point-in-time-safe because every
    # observation is already from a race before race_date.
    try:
        career_coeffs = np.polyfit(d_arr, v_arr, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None

    recent_slope, recent_intercept = recent_coeffs[0], recent_coeffs[1]
    career_slope, career_intercept = career_coeffs[0], career_coeffs[1]

    # Sanity checks on the recent fit (the leading metric we surface)
    if recent_intercept < 40 or recent_intercept > 85:
        return None
    if recent_slope > POSITIVE_SLOPE_CLAMP_THRESHOLD:
        # Shouldn't be accelerating overall — clamp to flat. Recent-form
        # fits use few weighted observations, so spurious positive slope
        # from noise is plausible; rejecting (as curves.py does) would
        # drop usable form coverage. See curves.py:POSITIVE_SLOPE_CLAMP_THRESHOLD
        # for the rationale behind the asymmetric handling.
        recent_slope = 0.0
    # Same clamp for the career fit. Don't reject — career and recent
    # share the same data distribution; if recent passed the intercept
    # gate the career intercept is bounded similarly in practice.
    if career_slope > POSITIVE_SLOPE_CLAMP_THRESHOLD:
        career_slope = 0.0

    current_v0 = round(float(recent_intercept), 2)
    current_decay = round(float(-recent_slope * 1000), 4)
    career_v0 = round(float(career_intercept), 2)
    career_decay = round(float(-career_slope * 1000), 4)
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
