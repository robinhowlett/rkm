#!/usr/bin/env python3
"""Compute hierarchical group priors + track offsets, apply to velocity curves."""

import csv
import logging
import sys
import tempfile

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from rkm.adjustments import compute_track_offsets, apply_adjustments
from rkm.db import connect, connect_raw

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    log.info("Loading velocity curves...")
    with connect_raw() as conn:
        curves_df = pd.read_sql(
            "SELECT horse_key, surface, v0, decay_rate, n_races FROM handycapper.rkm_velocity_curves",
            conn,
        )

    log.info(f"Loaded {len(curves_df):,} curves")

    # We need to know each horse's primary track for the offset computation
    log.info("Loading horse-track assignments...")
    with connect_raw() as conn:
        horse_tracks = pd.read_sql("""
            SELECT vc.horse_key, vc.surface, r.track, COUNT(*) AS starts
            FROM handycapper.rkm_velocity_curves vc
            JOIN handycapper.starters s ON s.horse = SPLIT_PART(vc.horse_key, '|', 1)
            JOIN handycapper.races r ON r.id = s.race_id
            WHERE r.surface = vc.surface
              AND r.date BETWEEN '1997-01-01' AND '2016-12-31'
            GROUP BY vc.horse_key, vc.surface, r.track
        """, conn)

    log.info(f"Loaded {len(horse_tracks):,} horse-track-surface combinations")

    # Assign primary track (most starts)
    primary_tracks = horse_tracks.sort_values("starts", ascending=False).drop_duplicates(
        subset=["horse_key", "surface"], keep="first"
    )[["horse_key", "surface", "track"]]

    curves_df = curves_df.merge(primary_tracks, on=["horse_key", "surface"], how="left")

    for col in ["v0", "decay_rate"]:
        curves_df[col] = pd.to_numeric(curves_df[col], errors="coerce")

    # Compute track offsets from shipping horses
    offsets_df = compute_track_offsets(curves_df)

    if not offsets_df.empty:
        log.info(f"Track offsets (top 10 positive = inflated):")
        top = offsets_df.nlargest(10, "v0_offset")
        for _, row in top.iterrows():
            log.info(f"  {row['track']:5s}: +{row['v0_offset']:.2f} ({row['n_shippers']} shippers)")

        log.info(f"Track offsets (top 5 negative = deflated):")
        bottom = offsets_df.nsmallest(5, "v0_offset")
        for _, row in bottom.iterrows():
            log.info(f"  {row['track']:5s}: {row['v0_offset']:.2f} ({row['n_shippers']} shippers)")

    # Apply adjustments
    adjusted = apply_adjustments(curves_df, offsets_df)

    # Write track offsets
    log.info("Writing track offsets...")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE handycapper.rkm_track_offsets")
            for _, row in offsets_df.iterrows():
                cur.execute(
                    "INSERT INTO handycapper.rkm_track_offsets (track, v0_offset, n_shippers, confidence) "
                    "VALUES (%s, %s, %s, %s)",
                    (row["track"], float(row["v0_offset"]), int(row["n_shippers"]), float(row["confidence"])),
                )
        conn.commit()

    # Update velocity_curves with adjusted values
    log.info("Updating velocity curves with adjustments...")
    csv_path = tempfile.mktemp(suffix=".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for _, row in adjusted.iterrows():
            if pd.notna(row["adj_v0"]):
                writer.writerow([row["horse_key"], row["surface"], row["adj_v0"], row["adj_decay"]])

    with connect() as conn:
        with conn.cursor() as cur:
            # Create temp table, load, then update
            cur.execute("""
                CREATE TEMP TABLE tmp_adj (
                    horse_key varchar(60), surface varchar(12),
                    adj_v0 decimal(5,2), adj_decay decimal(6,4)
                )
            """)
            with open(csv_path, "r") as f:
                with cur.copy("COPY tmp_adj FROM STDIN WITH (FORMAT csv)") as copy:
                    for line in f:
                        copy.write(line)
            cur.execute("""
                UPDATE handycapper.rkm_velocity_curves vc
                SET adj_v0 = t.adj_v0, adj_decay = t.adj_decay
                FROM tmp_adj t
                WHERE vc.horse_key = t.horse_key AND vc.surface = t.surface
            """)
        conn.commit()

    import os
    os.unlink(csv_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
