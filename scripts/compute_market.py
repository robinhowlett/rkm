#!/usr/bin/env python3
"""Compute market analysis — model vs odds, Benter combination."""

import csv
import logging
import sys
import tempfile

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from rkm.db import connect, connect_raw
from rkm.market import (
    compute_combined_probabilities,
    compute_model_probabilities,
    compute_odds_probabilities,
    fit_benter_logit,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RACE_QUERY = """
SELECT
    s.id AS starter_id,
    r.id AS race_id,
    r.date AS race_date,
    r.furlongs,
    r.surface,
    s.horse,
    s.odds,
    s.winner
FROM handycapper.races r
JOIN handycapper.starters s ON s.race_id = r.id
WHERE r.breed = 'TB'
  AND r.date BETWEEN '1997-01-01' AND '2016-12-31'
  AND r.number_of_runners >= 5
  AND r.surface IN ('Dirt', 'Turf', 'Synthetic')
  AND s.odds IS NOT NULL AND s.odds > 0
ORDER BY r.id, s.id
"""


def main():
    log.info("Loading velocity curves...")
    with connect_raw() as conn:
        curves_df = pd.read_sql("""
            SELECT horse_key, surface, distance_zone, adj_v0, decay_rate
            FROM handycapper.rkm_velocity_curves
            WHERE adj_v0 IS NOT NULL
        """, conn)

    # Build curve lookup: horse_name → (surface, zone) → curve
    curves_df["horse_name"] = curves_df["horse_key"].str.split("|").str[0]
    curves_df["adj_v0"] = pd.to_numeric(curves_df["adj_v0"], errors="coerce")
    curves_df["decay_rate"] = pd.to_numeric(curves_df["decay_rate"], errors="coerce")

    curve_lookup = {}
    for _, row in curves_df.iterrows():
        key = (row["horse_name"], row["surface"], row["distance_zone"])
        curve_lookup[key] = {"adj_v0": float(row["adj_v0"]), "decay_rate": float(row["decay_rate"])}

    log.info(f"Loaded {len(curve_lookup):,} curve entries")

    log.info("Loading race data...")
    with connect_raw() as conn:
        races_df = pd.read_sql(RACE_QUERY, conn)

    log.info(f"Loaded {len(races_df):,} starters")
    races_df["odds"] = pd.to_numeric(races_df["odds"], errors="coerce")
    races_df["furlongs"] = pd.to_numeric(races_df["furlongs"], errors="coerce")

    # Process by race
    all_results = []
    model_probs_train = []
    odds_probs_train = []
    winners_train = []
    races_processed = 0
    races_skipped = 0

    for race_id, race_df in races_df.groupby("race_id", sort=False):
        furlongs = race_df["furlongs"].iloc[0]
        surface = race_df["surface"].iloc[0]
        race_date = race_df["race_date"].iloc[0]
        zone = "route" if furlongs > 6.5 else "sprint"
        distance_ft = furlongs * 660

        # Look up curves for each starter
        starters = []
        odds_list = []
        winner_idx = -1

        for i, (_, row) in enumerate(race_df.iterrows()):
            curve = curve_lookup.get((row["horse"], surface, zone))
            if curve is None:
                starters.append(None)
            else:
                starters.append(curve)
            odds_list.append(float(row["odds"]))
            if row["winner"]:
                winner_idx = i

        # Need at least 60% of field with curves
        has_curve = [s is not None for s in starters]
        if sum(has_curve) < len(starters) * 0.6:
            races_skipped += 1
            continue
        if winner_idx < 0:
            races_skipped += 1
            continue

        # For starters without curves, use field average
        avg_v0 = np.mean([s["adj_v0"] for s in starters if s is not None])
        avg_decay = np.mean([s["decay_rate"] for s in starters if s is not None])
        filled_starters = [
            s if s is not None else {"adj_v0": avg_v0, "decay_rate": avg_decay}
            for s in starters
        ]

        # Compute probabilities
        model_probs = compute_model_probabilities(filled_starters, distance_ft)
        odds_probs = compute_odds_probabilities(odds_list)

        # Store for Benter fit (training partition: 1997-2014)
        if str(race_date) < "2015-01-01":
            model_probs_train.append(model_probs)
            odds_probs_train.append(odds_probs)
            winners_train.append(winner_idx)

        # Store results
        for i, (_, row) in enumerate(race_df.iterrows()):
            all_results.append({
                "starter_id": int(row["starter_id"]),
                "race_id": int(race_id),
                "horse_key": row["horse"],
                "model_prob": round(model_probs[i], 4),
                "odds_prob": round(odds_probs[i], 4),
                "edge": round(model_probs[i] - odds_probs[i], 4),
                "won": bool(row["winner"]),
            })

        races_processed += 1
        if races_processed % 50000 == 0:
            log.info(f"  {races_processed:,} races processed")

    log.info(f"Processed {races_processed:,} races ({races_skipped:,} skipped for insufficient curves)")

    # Fit Benter logit on training data
    log.info(f"Fitting Benter logit on {len(model_probs_train):,} training races...")
    alpha, beta = fit_benter_logit(model_probs_train, odds_probs_train, winners_train)

    # Compute combined probabilities for all results
    log.info("Computing combined probabilities...")
    race_groups = {}
    for r in all_results:
        race_groups.setdefault(r["race_id"], []).append(r)

    for race_id, starters in race_groups.items():
        m_probs = [s["model_prob"] for s in starters]
        o_probs = [s["odds_prob"] for s in starters]
        combined = compute_combined_probabilities(m_probs, o_probs, alpha, beta)
        for i, s in enumerate(starters):
            s["combined_prob"] = round(combined[i], 4)

    # Write to DB
    log.info("Writing results...")
    csv_path = tempfile.mktemp(suffix=".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for r in all_results:
            writer.writerow([
                r["starter_id"], r["race_id"], r["horse_key"],
                r["model_prob"], r["odds_prob"], r["combined_prob"],
                r["edge"], r["won"],
            ])

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE handycapper.rkm_market_analysis")
            with open(csv_path, "r") as f:
                with cur.copy(
                    "COPY handycapper.rkm_market_analysis "
                    "(starter_id, race_id, horse_key, model_prob, odds_prob, "
                    "combined_prob, edge, won) "
                    "FROM STDIN WITH (FORMAT csv)"
                ) as copy:
                    for line in f:
                        copy.write(line)
        conn.commit()

    import os
    os.unlink(csv_path)

    # Quick validation
    log.info("=== VALIDATION ===")
    results_df = pd.DataFrame(all_results)
    winners = results_df[results_df["won"] == True]

    log.info(f"Model top pick wins: {(results_df.groupby('race_id')['model_prob'].idxmax().isin(winners.index)).mean()*100:.1f}%")
    log.info(f"Odds top pick wins: {(results_df.groupby('race_id')['odds_prob'].idxmax().isin(winners.index)).mean()*100:.1f}%")
    log.info(f"Combined top pick wins: {(results_df.groupby('race_id')['combined_prob'].idxmax().isin(winners.index)).mean()*100:.1f}%")

    # Calibration by bins
    log.info("\nModel calibration:")
    results_df["model_bin"] = pd.cut(results_df["model_prob"], bins=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0])
    cal = results_df.groupby("model_bin", observed=True).agg(
        n=("won", "count"), wins=("won", "sum"), avg_pred=("model_prob", "mean")
    )
    cal["actual_pct"] = cal["wins"] / cal["n"]
    log.info(cal[["n", "avg_pred", "actual_pct"]].to_string())

    log.info(f"\nDone. {len(all_results):,} records written. α={alpha:.4f}, β={beta:.4f}")


if __name__ == "__main__":
    main()
