#!/usr/bin/env python3
"""Compute time-weighted current form curves for all starters."""

import csv
import logging
import sys
import tempfile
from collections import defaultdict
from datetime import date, datetime

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from rkm.curves import compute_segment_velocities
from rkm.db import connect, connect_raw
from rkm.form import compute_form_at_date
from rkm.identity import build_horse_keys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_QUERY = """
SELECT
    s.id AS starter_id,
    s.horse,
    r.id AS race_id,
    r.date AS race_date,
    r.furlongs,
    r.surface,
    inf.feet,
    inf.millis
FROM handycapper.races r
JOIN handycapper.starters s ON s.race_id = r.id
JOIN handycapper.indiv_fractionals inf ON inf.starter_id = s.id
WHERE r.breed = 'TB'
  AND r.date BETWEEN '1997-01-01' AND '2016-12-31'
  AND r.number_of_runners >= 4
  AND r.surface IN ('Dirt', 'Turf', 'Synthetic')
  AND inf.millis IS NOT NULL AND inf.feet IS NOT NULL
  AND inf.feet > 0 AND inf.millis > 0
ORDER BY s.horse, r.date, inf.feet
"""


def main():
    log.info("Building horse keys...")
    horse_keys_by_starter = build_horse_keys()

    log.info("Loading career curves...")
    with connect_raw() as conn:
        career_curves = pd.read_sql("""
            SELECT horse_key, surface, distance_zone, adj_v0 AS career_v0, decay_rate AS career_decay
            FROM handycapper.rkm_velocity_curves
            WHERE adj_v0 IS NOT NULL
        """, conn)

    career_lookup = {}
    for _, row in career_curves.iterrows():
        key = (row["horse_key"], row["surface"], row["distance_zone"])
        career_lookup[key] = (float(row["career_v0"]), float(row["career_decay"]))

    log.info(f"Loaded {len(career_lookup):,} career curves")

    log.info("Loading fractional data...")
    with connect_raw() as conn:
        df = pd.read_sql(DATA_QUERY, conn)

    log.info(f"Loaded {len(df):,} observations")

    for col in ["feet", "millis", "furlongs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["starter_id"] = df["starter_id"].astype(int)
    df["horse_key"] = df["starter_id"].map(horse_keys_by_starter)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.dropna(subset=["horse_key", "feet", "millis"])

    # Classify distance zone per race
    race_max_feet = df.groupby("race_id")["feet"].max().reset_index()
    race_max_feet["distance_zone"] = race_max_feet["feet"].apply(
        lambda f: "sprint" if f <= 4290 else "route"
    )
    df = df.merge(race_max_feet[["race_id", "distance_zone"]], on="race_id")

    # Pre-filter: only keep data for horses that have career curves
    valid_keys = set(career_lookup.keys())
    df["group_key"] = list(zip(df["horse_key"], df["surface"], df["distance_zone"]))
    df = df[df["group_key"].isin(valid_keys)].copy()
    df = df.drop(columns=["group_key"])
    log.info(f"After filtering to horses with curves: {len(df):,} observations")

    # Process per horse × surface × zone
    log.info("Computing time-weighted form snapshots...")
    results = []
    groups = df.groupby(["horse_key", "surface", "distance_zone"])
    total_groups = len(groups)
    processed = 0

    for (horse_key, surface, zone), group_df in groups:
        career = career_lookup.get((horse_key, surface, zone))
        if career is None:
            continue

        career_v0, career_decay = career

        # Group into per-race observations
        race_obs = []
        race_starters = []  # (starter_id, race_id, race_date)

        for race_id, race_df in group_df.groupby("race_id", sort=False):
            race_date = race_df["race_date"].iloc[0]
            points = [{"feet": int(r.feet), "millis": int(r.millis)} for r in race_df.itertuples()]
            vel_points = compute_segment_velocities(points)

            if vel_points:
                distances = [vp[0] for vp in vel_points]
                velocities = [vp[1] for vp in vel_points]
                race_obs.append({
                    "race_date": race_date.date() if hasattr(race_date, 'date') else race_date,
                    "distances": distances,
                    "velocities": velocities,
                })
                race_starters.append((
                    int(race_df["starter_id"].iloc[0]),
                    int(race_id),
                    race_date.date() if hasattr(race_date, 'date') else race_date,
                ))

        # Sort chronologically
        sorted_indices = sorted(range(len(race_obs)), key=lambda i: race_obs[i]["race_date"])
        race_obs = [race_obs[i] for i in sorted_indices]
        race_starters = [race_starters[i] for i in sorted_indices]

        # For each race (starting from the 3rd), compute form from prior races
        for i in range(2, len(race_obs)):
            starter_id, race_id, race_date = race_starters[i]
            prior = race_obs[:i]  # all races before this one

            snapshot = compute_form_at_date(prior, race_date, career_v0, career_decay)
            if snapshot is None:
                continue

            results.append({
                "starter_id": starter_id,
                "race_id": race_id,
                "horse_key": horse_key,
                "current_v0": snapshot.current_v0,
                "current_decay": snapshot.current_decay,
                "career_v0": snapshot.career_v0,
                "career_decay": snapshot.career_decay,
                "v0_trend": snapshot.v0_trend,
                "n_recent_races": snapshot.n_recent_races,
                "days_since_last": min(snapshot.days_since_last, 32000),
            })

        processed += 1
        if processed % 10000 == 0:
            log.info(f"  {processed:,}/{total_groups:,} horses processed, {len(results):,} snapshots")

    log.info(f"Complete: {len(results):,} form snapshots from {processed:,} horses")

    # Write via COPY
    log.info("Writing results...")
    csv_path = tempfile.mktemp(suffix=".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for r in results:
            writer.writerow([
                r["starter_id"], r["race_id"], r["horse_key"],
                r["current_v0"], r["current_decay"],
                r["career_v0"], r["career_decay"],
                r["v0_trend"], r["n_recent_races"], r["days_since_last"],
            ])

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE handycapper.rkm_current_form")
            with open(csv_path, "r") as f:
                with cur.copy(
                    "COPY handycapper.rkm_current_form "
                    "(starter_id, race_id, horse_key, current_v0, current_decay, "
                    "career_v0, career_decay, v0_trend, n_recent_races, days_since_last) "
                    "FROM STDIN WITH (FORMAT csv)"
                ) as copy:
                    for line in f:
                        copy.write(line)
        conn.commit()

    import os
    os.unlink(csv_path)
    log.info(f"Done. {len(results):,} form snapshots written.")


if __name__ == "__main__":
    main()
