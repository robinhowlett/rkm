"""Per-race performance analysis — surprise, pace context, interesting detection."""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

INTERESTING_THRESHOLD = 1.5  # standard deviations from own baseline
PACE_V0_SPREAD = 1.5  # ft/s above median to count as "speed type"


def predict_velocity(distance_ft: float, v0: float, decay_rate: float) -> float:
    """Predict velocity at a given distance from the horse's curve."""
    return v0 - decay_rate * (distance_ft / 1000.0)


def classify_pace(field_v0s: list[float]) -> str:
    """Classify pace scenario from field v0 distribution."""
    if len(field_v0s) < 3:
        return "UNKNOWN"
    median_v0 = np.median(field_v0s)
    speed_count = sum(1 for v in field_v0s if v > median_v0 + PACE_V0_SPREAD)
    if speed_count >= 3:
        return "CONTESTED"
    elif speed_count >= 2:
        return "PRESSURED"
    return "CONTROLLED"


def compute_race_performances(race_starters: pd.DataFrame, curves: dict,
                              race_furlongs: float) -> list[dict]:
    """Compute performance metrics for all starters in one race.

    Args:
        race_starters: DataFrame with columns [starter_id, horse_key, race_id,
                       velocity_points: list of (distance_midpoint, velocity)]
        curves: dict of horse_key → {v0, decay_rate, residual_std, distance_zone}
        race_furlongs: race distance in furlongs
    """
    race_id = race_starters["race_id"].iloc[0] if len(race_starters) > 0 else None
    zone = "route" if race_furlongs > 6.5 else "sprint"

    # Look up curves for each starter
    starters_with_curves = []
    field_v0s = []

    for _, row in race_starters.iterrows():
        hkey = row["horse_key"]
        curve = curves.get((hkey, zone))
        if curve is None:
            continue
        starters_with_curves.append({
            "starter_id": row["starter_id"],
            "horse_key": hkey,
            "race_id": row["race_id"],
            "vel_points": row["velocity_points"],
            "curve_v0": curve["v0"],
            "curve_decay": curve["decay_rate"],
            "curve_residual": curve["residual_std"],
        })
        field_v0s.append(curve["v0"])

    if len(starters_with_curves) < 3:
        return []

    # Pace scenario
    pace = classify_pace(field_v0s)

    # Field strength: average predicted velocity at race midpoint
    race_mid_ft = race_furlongs * 660 / 2  # midpoint in feet
    field_strength = np.mean([
        predict_velocity(race_mid_ft, s["curve_v0"], s["curve_decay"])
        for s in starters_with_curves
    ])

    # First pass: compute raw surprise for each starter
    raw_results = []
    for s in starters_with_curves:
        vel_points = s["vel_points"]
        if not vel_points:
            continue

        surprises = []
        actual_vels = []
        for dist, actual_v in vel_points:
            predicted_v = predict_velocity(dist, s["curve_v0"], s["curve_decay"])
            surprises.append(actual_v - predicted_v)
            actual_vels.append(actual_v)

        if not surprises:
            continue

        raw_results.append({
            **s,
            "raw_surprise": np.mean(surprises),
            "actual_v_avg": np.mean(actual_vels),
        })

    if not raw_results:
        return []

    # Extract race-level variant (shared residual — track speed that day)
    race_variant = np.mean([r["raw_surprise"] for r in raw_results])

    # Second pass: compute variant-adjusted surprise and flag interesting
    median_decay = np.median([s["curve_decay"] for s in starters_with_curves])
    results = []

    for r in raw_results:
        adjusted_surprise = r["raw_surprise"] - race_variant
        actual_v_avg = r["actual_v_avg"]
        predicted_v_avg = actual_v_avg - adjusted_surprise

        residual = r["curve_residual"] if r["curve_residual"] and r["curve_residual"] > 0 else 1.5
        surprise_pct = adjusted_surprise / residual

        is_interesting = abs(surprise_pct) > INTERESTING_THRESHOLD
        interest_reason = None
        if is_interesting:
            if surprise_pct > INTERESTING_THRESHOLD:
                if pace == "CONTESTED" and r["curve_decay"] < median_decay:
                    interest_reason = "Big run aided by contested pace (closer benefited)"
                else:
                    interest_reason = "Significantly above baseline — form improvement?"
            else:
                if pace == "CONTROLLED" and r["curve_decay"] < median_decay:
                    interest_reason = "Underperformed in pace that hurt closing style"
                else:
                    interest_reason = "Significantly below baseline — form decline or trouble?"

        results.append({
            "starter_id": int(r["starter_id"]),
            "race_id": int(r["race_id"]),
            "horse_key": r["horse_key"],
            "predicted_v_avg": round(float(predicted_v_avg), 2),
            "actual_v_avg": round(float(actual_v_avg), 2),
            "surprise": round(float(adjusted_surprise), 2),
            "surprise_pct": round(float(surprise_pct), 2),
            "pace_scenario": pace,
            "field_strength": round(float(field_strength), 2),
            "field_size": len(starters_with_curves),
            "is_interesting": is_interesting,
            "interest_reason": interest_reason,
        })

    return results
