#!/usr/bin/env python3
"""Compute per-race performance metrics (surprise, pace, field strength, interesting flags)."""

import csv
import logging
import sys
import tempfile

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from rkm.curves import compute_segment_velocities
from rkm.db import connect, connect_raw
from rkm.identity import build_horse_keys
from rkm.performance import compute_race_performances

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_QUERY = """
SELECT
    s.id AS starter_id,
    s.horse,
    r.id AS race_id,
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
ORDER BY r.id, s.id, inf.feet
"""


def main():
    log.info("Building horse keys...")
    horse_keys_by_starter = build_horse_keys()

    log.info("Loading velocity curves...")
    with connect_raw() as conn:
        curves_df = pd.read_sql("""
            SELECT horse_key, distance_zone, adj_v0 AS v0, decay_rate, residual_std
            FROM handycapper.rkm_velocity_curves
            WHERE adj_v0 IS NOT NULL
        """, conn)

    # Build curve lookup: (horse_key, zone) → curve params
    curves = {}
    for _, row in curves_df.iterrows():
        key = (row["horse_key"], row["distance_zone"])
        curves[key] = {
            "v0": float(row["v0"]),
            "decay_rate": float(row["decay_rate"]),
            "residual_std": float(row["residual_std"]) if pd.notna(row["residual_std"]) else 1.5,
            "distance_zone": row["distance_zone"],
        }
    log.info(f"Loaded {len(curves):,} curve profiles")

    log.info("Loading race fractional data...")
    with connect_raw() as conn:
        df = pd.read_sql(DATA_QUERY, conn)
    log.info(f"Loaded {len(df):,} fractional observations")

    for col in ["feet", "millis", "furlongs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["starter_id"] = df["starter_id"].astype(int)
    df["horse_key"] = df["starter_id"].map(horse_keys_by_starter)
    df = df.dropna(subset=["horse_key", "feet", "millis"])

    # Group by race, compute velocities per starter, then analyze
    all_results = []
    races_processed = 0

    for race_id, race_df in df.groupby("race_id", sort=False):
        furlongs = race_df["furlongs"].iloc[0]
        surface = race_df["surface"].iloc[0]

        # Compute velocity points for each starter in this race
        starter_data = []
        for starter_id, starter_df in race_df.groupby("starter_id", sort=False):
            points = [{"feet": int(r.feet), "millis": int(r.millis)} for r in starter_df.itertuples()]
            vel_points = compute_segment_velocities(points)
            if vel_points:
                starter_data.append({
                    "starter_id": int(starter_id),
                    "horse_key": starter_df["horse_key"].iloc[0],
                    "race_id": int(race_id),
                    "velocity_points": vel_points,
                })

        if len(starter_data) < 3:
            continue

        race_starters_df = pd.DataFrame(starter_data)
        results = compute_race_performances(race_starters_df, curves, float(furlongs))
        all_results.extend(results)

        races_processed += 1
        if races_processed % 50000 == 0:
            log.info(f"  {races_processed:,} races processed, {len(all_results):,} results")

    log.info(f"Processing complete: {races_processed:,} races, {len(all_results):,} performance records")

    # Write via COPY
    log.info("Writing results...")
    csv_path = tempfile.mktemp(suffix=".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for r in all_results:
            writer.writerow([
                r["starter_id"], r["race_id"], r["horse_key"],
                r["predicted_v_avg"], r["actual_v_avg"], r["surprise"], r["surprise_pct"],
                r["pace_scenario"], r["field_strength"], r["field_size"],
                r["is_interesting"], r["interest_reason"],
            ])

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE handycapper.rkm_race_performance")
            with open(csv_path, "r") as f:
                with cur.copy(
                    "COPY handycapper.rkm_race_performance "
                    "(starter_id, race_id, horse_key, predicted_v_avg, actual_v_avg, "
                    "surprise, surprise_pct, pace_scenario, field_strength, field_size, "
                    "is_interesting, interest_reason) "
                    "FROM STDIN WITH (FORMAT csv, NULL '')"
                ) as copy:
                    for line in f:
                        copy.write(line)
        conn.commit()

    import os
    os.unlink(csv_path)

    interesting_count = sum(1 for r in all_results if r["is_interesting"])
    log.info(f"Done. {len(all_results):,} records written ({interesting_count:,} interesting = {100*interesting_count/max(len(all_results),1):.1f}%)")


if __name__ == "__main__":
    main()
