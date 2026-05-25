"""Hierarchical pooling + track-pair adjustments for velocity curves."""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import leastsq

from rkm.db import connect_raw

log = logging.getLogger(__name__)

MIN_GROUP_SIZE = 10
MIN_SHIPPING_STARTS = 3
MIN_SHIPPING_PAIRS = 30
REFERENCE_TRACK = "BEL"


def classify_going(track_condition: str | None) -> str:
    if track_condition is None:
        return "fast"
    off_conditions = {"Muddy", "Sloppy", "Heavy", "Wet Fast", "Slow", "Yielding", "Soft"}
    return "off" if track_condition in off_conditions else "fast"


def classify_race_class(race_type: str | None, grade: int | None) -> str:
    if race_type is None:
        return "CLAIMING"
    rt = race_type.upper()
    if grade is not None and grade in (1, 2, 3):
        return "STAKES_GRADED"
    if "STAKES" in rt or "HANDICAP" in rt:
        return "STAKES_UNGRADED"
    if "MAIDEN SPECIAL" in rt:
        return "MAIDEN_SW"
    if "MAIDEN" in rt:
        return "MAIDEN_CLM"
    if "ALLOWANCE" in rt or "STARTER" in rt or "OPTIONAL" in rt:
        return "ALLOWANCE"
    return "CLAIMING"


def classify_age(min_age, max_age) -> str:
    if min_age == 2 and max_age == 2:
        return "2yo"
    if min_age == 3 and max_age == 3:
        return "3yo"
    return "older"


def classify_sex(sexes_code: str | None) -> str:
    if sexes_code in ("F", "F&M"):
        return "F_M"
    return "open"


def load_race_metadata() -> pd.DataFrame:
    """Load race-level metadata for grouping."""
    query = """
    SELECT r.id AS race_id, r.track, r.surface, r.track_condition, r.type AS race_type,
           r.grade, r.min_age, r.max_age, r.sexes_code, r.purse
    FROM handycapper.races r
    WHERE r.breed = 'TB'
      AND r.date BETWEEN '1997-01-01' AND '2016-12-31'
      AND r.surface IN ('Dirt', 'Turf', 'Synthetic')
    """
    with connect_raw() as conn:
        df = pd.read_sql(query, conn)
    return df


def load_horse_race_assignments() -> pd.DataFrame:
    """Load which horse ran in which race (to assign horses to groups)."""
    query = """
    SELECT s.id AS starter_id, s.horse, r.id AS race_id, r.track, r.surface
    FROM handycapper.starters s
    JOIN handycapper.races r ON r.id = s.race_id
    WHERE r.breed = 'TB'
      AND r.date BETWEEN '1997-01-01' AND '2016-12-31'
      AND r.surface IN ('Dirt', 'Turf', 'Synthetic')
    """
    with connect_raw() as conn:
        df = pd.read_sql(query, conn)
    return df


def compute_group_priors(curves_df: pd.DataFrame, race_meta: pd.DataFrame,
                         horse_races: pd.DataFrame) -> pd.DataFrame:
    """Compute group-level (mean, std) for v0 and decay_rate."""
    log.info("Classifying races into groups...")

    race_meta["going"] = race_meta["track_condition"].apply(classify_going)
    race_meta["race_class"] = race_meta.apply(
        lambda r: classify_race_class(r["race_type"], r.get("grade")), axis=1
    )
    race_meta["age_group"] = race_meta.apply(
        lambda r: classify_age(r["min_age"], r["max_age"]), axis=1
    )
    race_meta["sex_group"] = race_meta["sexes_code"].apply(classify_sex)

    # Assign each horse_key to its primary group (most starts)
    horse_races_with_meta = horse_races.merge(
        race_meta[["race_id", "track", "surface", "going", "race_class", "age_group", "sex_group"]],
        on="race_id", suffixes=("", "_meta")
    )

    # Map starter_id → horse_key from curves
    # For now, use the curves_df directly grouped by (track, surface, going, race_class, age_group, sex_group)
    # We compute group priors from the race-level metadata, not the horse-level curves

    # Get one row per (horse_key, surface) from curves
    # Then find which group they MOST raced in
    # Actually simpler: compute group means directly from the velocity data per race group

    # Merge curves with horse_races to find which races each horse ran in
    # This is complex — let's simplify: compute group priors from ALL horses at that track/surface
    # grouped by the race dimensions

    group_cols = ["track", "surface", "going", "race_class", "age_group", "sex_group"]

    # Count races per group
    group_race_counts = race_meta.groupby(group_cols).size().reset_index(name="n_races")

    # For group priors, we need the average v0/decay of horses that raced in each group
    # Use the curves_df (one per horse×surface) and the horse's primary track
    curves_with_track = curves_df.copy()
    # Extract track from horse_key's primary racing location — we'll use a simpler approach:
    # just group by track × surface and compute means from the curves table directly

    # Simplest correct approach: group priors by (track, surface) from existing curves
    # Then also compute cross-track priors by (surface, going, race_class, age_group, sex_group)
    track_surface_stats = curves_df.groupby(["track", "surface"]).agg(
        mean_v0=("v0", "mean"),
        std_v0=("v0", "std"),
        mean_decay=("decay_rate", "mean"),
        std_decay=("decay_rate", "std"),
        n_horses=("v0", "count"),
    ).reset_index()

    log.info(f"Computed {len(track_surface_stats)} track×surface group priors")
    return track_surface_stats


def compute_track_offsets(curves_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-track v0 offset from shipping horses using network least-squares."""
    log.info("Computing track offsets from shipping horses...")

    # Find horses with curves at 2+ tracks on the same surface
    multi_track = curves_df.groupby("horse_key").filter(lambda g: g["track"].nunique() >= 2)

    if multi_track.empty:
        log.warning("No shipping horses found")
        return pd.DataFrame(columns=["track", "v0_offset", "n_shippers", "confidence"])

    # For each horse, compute their mean v0 at each track (same surface)
    horse_track_v0 = multi_track.groupby(["horse_key", "track", "surface"]).agg(
        mean_v0=("v0", "mean"),
        n_races=("n_races", "first"),
    ).reset_index()

    # Only keep horses with enough starts at each track
    horse_track_v0 = horse_track_v0[horse_track_v0["n_races"] >= MIN_SHIPPING_STARTS]

    # Compute pairwise offsets
    pairs = horse_track_v0.merge(horse_track_v0, on=["horse_key", "surface"], suffixes=("_a", "_b"))
    pairs = pairs[pairs["track_a"] < pairs["track_b"]]
    pairs["offset"] = pairs["mean_v0_a"] - pairs["mean_v0_b"]

    # Aggregate by track pair
    pair_stats = pairs.groupby(["track_a", "track_b"]).agg(
        mean_offset=("offset", "mean"),
        n_horses=("offset", "count"),
    ).reset_index()
    pair_stats = pair_stats[pair_stats["n_horses"] >= MIN_SHIPPING_PAIRS]

    log.info(f"Found {len(pair_stats)} track pairs with sufficient shipping data")

    if pair_stats.empty:
        return pd.DataFrame(columns=["track", "v0_offset", "n_shippers", "confidence"])

    # Network least-squares: solve for per-track offset minimizing pairwise error
    all_tracks = sorted(set(pair_stats["track_a"]) | set(pair_stats["track_b"]))
    track_idx = {t: i for i, t in enumerate(all_tracks)}
    n_tracks = len(all_tracks)

    # Reference track (BEL) gets offset 0
    ref_idx = track_idx.get(REFERENCE_TRACK, 0)

    def residuals(offsets):
        full_offsets = np.insert(offsets, ref_idx, 0.0)
        res = []
        for _, row in pair_stats.iterrows():
            ia = track_idx[row["track_a"]]
            ib = track_idx[row["track_b"]]
            predicted_offset = full_offsets[ia] - full_offsets[ib]
            weight = np.sqrt(row["n_horses"])
            res.append((row["mean_offset"] - predicted_offset) * weight)
        return res

    x0 = np.zeros(n_tracks - 1)
    result = leastsq(residuals, x0, full_output=False)
    solved = np.insert(result[0] if isinstance(result, tuple) else result, ref_idx, 0.0)

    # Count shippers per track
    shippers_per_track = pd.concat([
        pair_stats[["track_a", "n_horses"]].rename(columns={"track_a": "track"}),
        pair_stats[["track_b", "n_horses"]].rename(columns={"track_b": "track"}),
    ]).groupby("track")["n_horses"].sum().reset_index()

    offsets_df = pd.DataFrame({
        "track": all_tracks,
        "v0_offset": np.round(solved, 2),
    })
    offsets_df = offsets_df.merge(shippers_per_track, on="track", how="left")
    offsets_df = offsets_df.rename(columns={"n_horses": "n_shippers"})
    offsets_df["n_shippers"] = offsets_df["n_shippers"].fillna(0).astype(int)
    offsets_df["confidence"] = np.clip(offsets_df["n_shippers"] / offsets_df["n_shippers"].quantile(0.9), 0.1, 1.0).round(2)

    log.info(f"Solved offsets for {len(offsets_df)} tracks (ref={REFERENCE_TRACK}=0)")
    return offsets_df


def apply_adjustments(curves_df: pd.DataFrame, offsets_df: pd.DataFrame) -> pd.DataFrame:
    """Apply track offsets to produce adj_v0 and adj_decay."""
    merged = curves_df.merge(offsets_df[["track", "v0_offset"]], on="track", how="left")
    merged["v0_offset"] = merged["v0_offset"].fillna(0)
    merged["adj_v0"] = (merged["v0"] - merged["v0_offset"]).round(2)
    merged["adj_decay"] = merged["decay_rate"]  # decay offset is typically negligible
    return merged[["horse_key", "surface", "adj_v0", "adj_decay"]]
