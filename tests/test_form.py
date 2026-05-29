"""Unit tests for rkm.form.compute_form_at_date.

Specifically targets RKM-T1.4: career_v0 / career_decay must be computed
strictly from observations BEFORE the snapshot date — never from the
horse's full-career curve, which would leak future races into the
baseline.
"""

import sys
from datetime import date
from pathlib import Path

# Make the source tree importable without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from rkm.form import compute_form_at_date


def _race(d: date, distances: list[float], velocities: list[float]) -> dict:
    """Build a single prior-race observation dict."""
    return {"race_date": d, "distances": distances, "velocities": velocities}


def _flat_race(d: date, v0: float) -> dict:
    """Build a 4-point observation that fits exactly to v(d) = v0."""
    return _race(d, [200.0, 400.0, 600.0, 800.0], [v0, v0, v0, v0])


def test_point_in_time_safety_career_uses_only_prior_observations():
    """RKM-T1.4: snapshot at race_date must be invariant under any future
    observations the caller might have access to. The function takes only
    prior_observations, so we verify that two calls with identical priors
    return identical career_v0 regardless of what we do with later data."""
    priors = [
        _flat_race(date(2014, 1, 1), 60.0),
        _flat_race(date(2014, 3, 1), 60.0),
        _flat_race(date(2014, 5, 1), 60.0),
    ]
    # The snapshot at 2014-06-01 should be derived from these 3 priors only.
    snap = compute_form_at_date(priors, date(2014, 6, 1))
    assert snap is not None
    assert snap.career_v0 == 60.0
    # Recomputing with the same priors yields the same career_v0.
    snap2 = compute_form_at_date(priors, date(2014, 6, 1))
    assert snap2.career_v0 == snap.career_v0


def test_career_baseline_does_not_react_to_post_date_changes():
    """If we add a future race to the SAME observation list (which would be
    a caller bug — race_date filters happen earlier), the function still
    correctly ignores the future race via its own days_ago > 0 check."""
    priors_only = [
        _flat_race(date(2014, 1, 1), 60.0),
        _flat_race(date(2014, 3, 1), 60.0),
        _flat_race(date(2014, 5, 1), 60.0),
    ]
    snap_clean = compute_form_at_date(priors_only, date(2014, 6, 1))

    # Same priors plus a future "race" the caller forgot to filter.
    priors_with_future = priors_only + [_flat_race(date(2014, 9, 1), 50.0)]
    snap_robust = compute_form_at_date(priors_with_future, date(2014, 6, 1))

    # The future race must not influence career_v0 (the days_ago<=0 check
    # excludes it before regression).
    assert snap_clean.career_v0 == snap_robust.career_v0


def test_v0_trend_recent_higher_than_career():
    """A horse running steadily then suddenly faster should produce
    current_v0 > career_v0 → positive v0_trend."""
    priors = [
        _flat_race(date(2013, 1, 1), 58.0),  # 1.5 years ago — heavily decayed weight
        _flat_race(date(2013, 6, 1), 58.0),
        _flat_race(date(2014, 5, 15), 64.0),  # ~2 weeks ago — high weight
    ]
    snap = compute_form_at_date(priors, date(2014, 6, 1))
    assert snap is not None
    # Recent fit should sit closer to 64; career fit is the unweighted
    # mean of the three, so the trend is meaningfully positive.
    assert snap.current_v0 > snap.career_v0
    assert snap.v0_trend > 0


def test_v0_trend_recent_lower_than_career():
    """Mirror case: recent races slower than the horse's earlier form."""
    priors = [
        _flat_race(date(2013, 1, 1), 64.0),
        _flat_race(date(2013, 6, 1), 64.0),
        _flat_race(date(2014, 5, 15), 58.0),
    ]
    snap = compute_form_at_date(priors, date(2014, 6, 1))
    assert snap is not None
    assert snap.current_v0 < snap.career_v0
    assert snap.v0_trend < 0


def test_single_prior_race_yields_zero_trend():
    """With only one prior race, recent and career fits are identical, so
    v0_trend must be 0 — no spurious signal from a single observation."""
    priors = [_flat_race(date(2014, 5, 15), 60.0)]
    snap = compute_form_at_date(priors, date(2014, 6, 1))
    assert snap is not None
    # Both fits collapse to the same constant; trend is 0.
    assert snap.v0_trend == 0


def test_no_prior_observations_returns_none():
    assert compute_form_at_date([], date(2014, 6, 1)) is None


def test_too_few_velocity_points_returns_none():
    """Three total velocity points (one race × 3 points) is below the
    4-point regression floor."""
    priors = [_race(date(2014, 5, 15), [200.0, 400.0, 600.0],
                    [60.0, 60.0, 60.0])]
    assert compute_form_at_date(priors, date(2014, 6, 1)) is None


def test_implausible_intercept_returns_none():
    """Intercept outside [40, 85] should be rejected via the recent-fit
    sanity check — same gate as before the RKM-T1.4 refactor."""
    # All 4 points at v=20 → intercept ≈ 20, below the 40 floor.
    priors = [_race(date(2014, 5, 15), [200.0, 400.0, 600.0, 800.0],
                    [20.0, 20.0, 20.0, 20.0])]
    # 20 is below MIN_VELOCITY (30) so points are filtered before regression
    # and we get None for "too few points" rather than the intercept gate.
    # Use velocities at 32 (above MIN_VELOCITY but yielding intercept < 40).
    priors = [_race(date(2014, 5, 15), [200.0, 400.0, 600.0, 800.0],
                    [32.0, 32.0, 32.0, 32.0])]
    assert compute_form_at_date(priors, date(2014, 6, 1)) is None


def test_signature_no_longer_takes_career_args():
    """The pre-RKM-T1.4 signature took career_v0 / career_decay as arguments.
    The fix dropped those — calling with the old signature should be a
    TypeError. This pins the contract change."""
    priors = [_flat_race(date(2014, 5, 15), 60.0)]
    with pytest.raises(TypeError):
        compute_form_at_date(priors, date(2014, 6, 1), 60.0, 5.0)
