#!/usr/bin/env python3
"""Compute velocity curves for all horses (1997-2016)."""

import csv
import logging
import sys
import tempfile
from collections import defaultdict
from datetime import date

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from rkm.curves import fit_horse_curve
from rkm.db import connect, connect_raw
from rkm.identity import build_horse_keys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_QUERY = """
SELECT
    s.id AS starter_id,
    s.horse,
    r.id AS race_id,
    r.date AS race_date,
    r.surface,
    inf.point,
    inf.feet,
    inf.millis,
    (SELECT MAX(poc.tot_len_bhd) FROM handycapper.points_of_call poc
     WHERE poc.starter_id = s.id) AS max_len_bhd
FROM handycapper.races r
JOIN handycapper.starters s ON s.race_id = r.id
JOIN handycapper.indiv_fractionals inf ON inf.starter_id = s.id
WHERE r.breed = 'TB'
  AND r.date BETWEEN '1997-01-01' AND '2016-12-31'
  AND r.number_of_runners >= 4
  AND r.surface IN ('Dirt', 'Turf', 'Synthetic')
  AND inf.millis IS NOT NULL
  AND inf.feet IS NOT NULL
  AND inf.feet > 0
  AND inf.millis > 0
ORDER BY s.horse, r.date, inf.feet
"""


def main():
    log.info("Building horse keys...")
    horse_keys_by_starter = build_horse_keys()

    log.info("Loading fractional data (1997-2016)...")
    with connect_raw() as conn:
        df = pd.read_sql(DATA_QUERY, conn)

    log.info(f"Loaded {len(df):,} fractional observations")

    # Type coercion
    for col in ["feet", "millis", "max_len_bhd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["starter_id"] = df["starter_id"].astype(int)

    # Map starter_id to horse_key
    df["horse_key"] = df["starter_id"].map(horse_keys_by_starter)
    df = df.dropna(subset=["horse_key", "feet", "millis"])

    log.info(f"After mapping: {len(df):,} rows with horse keys")

    # Classify each race as sprint or route based on max feet in that race
    race_max_feet = df.groupby("race_id")["feet"].max().reset_index()
    race_max_feet["distance_zone"] = race_max_feet["feet"].apply(
        lambda f: "sprint" if f <= 4290 else "route"  # 6.5f = 4290 feet
    )
    df = df.merge(race_max_feet[["race_id", "distance_zone"]], on="race_id")

    # Group by (horse_key, surface, distance_zone)
    log.info("Grouping by horse × surface × distance zone...")
    results = []
    groups = df.groupby(["horse_key", "surface", "distance_zone"])
    total_groups = len(groups)
    fitted = 0
    skipped = 0

    for i, ((horse_key, surface, zone), group_df) in enumerate(groups):
        races = []
        beaten_lengths = []

        for race_id, race_df in group_df.groupby("race_id", sort=False):
            points = [
                {"feet": int(r.feet), "millis": int(r.millis), "race_date": str(r.race_date)}
                for r in race_df.itertuples()
            ]
            races.append(points)
            max_bhd = race_df["max_len_bhd"].max()
            beaten_lengths.append(max_bhd if pd.notna(max_bhd) else None)

        fit = fit_horse_curve(races, beaten_lengths)
        if fit is None:
            skipped += 1
            continue

        results.append({
            "horse_key": horse_key,
            "surface": surface,
            "distance_zone": zone,
            "v0": fit.v0,
            "decay_rate": fit.decay_rate,
            "n_observations": fit.n_observations,
            "n_races": fit.n_races,
            "residual_std": fit.residual_std,
            "first_race": fit.first_race,
            "last_race": fit.last_race,
        })
        fitted += 1

        if (i + 1) % 10000 == 0:
            log.info(f"  {i+1:,}/{total_groups:,} groups processed ({fitted:,} fitted, {skipped:,} skipped)")

    log.info(f"Fitting complete: {fitted:,} curves fitted, {skipped:,} skipped (insufficient data)")

    # Write via CSV + COPY
    log.info("Writing results via COPY...")
    csv_path = tempfile.mktemp(suffix=".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for r in results:
            writer.writerow([
                r["horse_key"], r["surface"], r["distance_zone"],
                r["v0"], r["decay_rate"],
                r["n_observations"], r["n_races"],
                r["residual_std"], r["first_race"], r["last_race"],
            ])

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE handycapper.rkm_velocity_curves")
            with open(csv_path, "r") as f:
                with cur.copy(
                    "COPY handycapper.rkm_velocity_curves "
                    "(horse_key, surface, distance_zone, v0, decay_rate, "
                    "n_observations, n_races, residual_std, first_race, last_race) "
                    "FROM STDIN WITH (FORMAT csv)"
                ) as copy:
                    for line in f:
                        copy.write(line)
        conn.commit()

    import os
    os.unlink(csv_path)

    log.info(f"Done. {fitted:,} velocity curves written to rkm_velocity_curves.")


if __name__ == "__main__":
    main()
