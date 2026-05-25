#!/usr/bin/env python3
"""Compute race situation analysis — identifies ITP-style betting opportunities."""

import csv
import logging
import sys
import tempfile

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from rkm.db import connect, connect_raw
from rkm.situations import analyze_race_situation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

QUERY = """
SELECT
    ma.race_id,
    ma.starter_id,
    ma.horse_key,
    ma.model_prob,
    ma.odds_prob,
    ma.edge,
    ma.won,
    s.odds,
    s.choice,
    s.finish_position,
    r.furlongs,
    r.number_of_runners,
    rp.pace_scenario,
    vc.adj_v0 AS v0,
    vc.decay_rate
FROM handycapper.rkm_market_analysis ma
JOIN handycapper.starters s ON s.id = ma.starter_id
JOIN handycapper.races r ON r.id = ma.race_id
LEFT JOIN handycapper.rkm_race_performance rp ON rp.starter_id = ma.starter_id
LEFT JOIN handycapper.rkm_velocity_curves vc ON SPLIT_PART(vc.horse_key, '|', 1) = ma.horse_key
    AND vc.surface = r.surface
    AND vc.distance_zone = CASE WHEN r.furlongs > 6.5 THEN 'route' ELSE 'sprint' END
WHERE r.number_of_runners >= 7
  AND r.date BETWEEN '2010-01-01' AND '2016-12-31'
  AND s.choice IS NOT NULL
ORDER BY ma.race_id, s.choice
"""

EXOTICS_QUERY = """
SELECT race_id,
    MAX(CASE WHEN name = 'Exacta' THEN payoff END) AS exacta_payoff,
    MAX(CASE WHEN name = 'Trifecta' THEN payoff END) AS trifecta_payoff,
    MAX(CASE WHEN name = 'Superfecta' THEN payoff END) AS super_payoff
FROM handycapper.exotics
WHERE payoff > 0
  AND race_id IN (SELECT DISTINCT race_id FROM handycapper.rkm_market_analysis ma
                  JOIN handycapper.races r ON r.id = ma.race_id
                  WHERE r.date BETWEEN '2010-01-01' AND '2016-12-31')
GROUP BY race_id
"""


def main():
    log.info("Loading race + market + curve data...")
    with connect_raw() as conn:
        df = pd.read_sql(QUERY, conn)
    log.info(f"Loaded {len(df):,} starters")

    log.info("Loading exotic payoffs...")
    with connect_raw() as conn:
        exotics = pd.read_sql(EXOTICS_QUERY, conn)
    log.info(f"Loaded exotic data for {len(exotics):,} races")

    exotics_lookup = exotics.set_index("race_id").to_dict("index")

    # Type coercion
    for col in ["odds", "model_prob", "odds_prob", "edge", "v0", "decay_rate", "furlongs",
                "choice", "finish_position", "number_of_runners"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Process race by race
    results = []
    races_processed = 0

    for race_id, race_df in df.groupby("race_id", sort=False):
        starters = []
        for _, row in race_df.iterrows():
            starters.append({
                "horse_key": row["horse_key"],
                "odds": row["odds"],
                "choice": int(row["choice"]) if pd.notna(row["choice"]) else 99,
                "finish_position": int(row["finish_position"]) if pd.notna(row["finish_position"]) else None,
                "v0": float(row["v0"]) if pd.notna(row["v0"]) else None,
                "decay_rate": float(row["decay_rate"]) if pd.notna(row["decay_rate"]) else None,
                "model_prob": float(row["model_prob"]) if pd.notna(row["model_prob"]) else None,
                "odds_prob": float(row["odds_prob"]) if pd.notna(row["odds_prob"]) else None,
                "edge": float(row["edge"]) if pd.notna(row["edge"]) else None,
            })

        exotic_data = exotics_lookup.get(int(race_id), {})

        race_data = {
            "race_id": int(race_id),
            "furlongs": float(race_df["furlongs"].iloc[0]),
            "field_size": int(race_df["number_of_runners"].iloc[0]),
            "starters": starters,
            "pace_scenario": race_df["pace_scenario"].iloc[0] if pd.notna(race_df["pace_scenario"].iloc[0]) else "UNKNOWN",
            "exacta_payoff": exotic_data.get("exacta_payoff"),
            "trifecta_payoff": exotic_data.get("trifecta_payoff"),
            "super_payoff": exotic_data.get("super_payoff"),
        }

        situation = analyze_race_situation(race_data)
        if situation:
            results.append(situation)

        races_processed += 1
        if races_processed % 50000 == 0:
            log.info(f"  {races_processed:,} races processed")

    log.info(f"Analysis complete: {len(results):,} races classified")

    # Summary stats
    if not results:
        log.warning("No races classified — check data joins")
        return
    types = pd.DataFrame(results)["situation_type"].value_counts()
    log.info(f"Situation distribution:\n{types.to_string()}")

    vuln_fav = [r for r in results if r["has_vulnerable_fav"]]
    if vuln_fav:
        hit_board = [r for r in vuln_fav if not r["fav_missed_board"]]
        missed_board = [r for r in vuln_fav if r["fav_missed_board"]]
        log.info(f"\nVulnerable favorites: {len(vuln_fav)} races")
        log.info(f"  Fav hit board: {len(hit_board)} ({100*len(hit_board)/len(vuln_fav):.1f}%)")
        log.info(f"  Fav missed board: {len(missed_board)} ({100*len(missed_board)/len(vuln_fav):.1f}%)")

        tri_when_hit = [r["trifecta_payoff"] for r in hit_board if r["trifecta_payoff"]]
        tri_when_missed = [r["trifecta_payoff"] for r in missed_board if r["trifecta_payoff"]]
        if tri_when_hit and tri_when_missed:
            import numpy as np
            log.info(f"  Avg trifecta when fav hits: ${np.mean(tri_when_hit):.2f}")
            log.info(f"  Avg trifecta when fav misses: ${np.mean(tri_when_missed):.2f}")
            log.info(f"  Multiplier: {np.mean(tri_when_missed)/np.mean(tri_when_hit):.1f}×")

    # Write via COPY
    log.info("Writing results...")
    csv_path = tempfile.mktemp(suffix=".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for r in results:
            writer.writerow([
                r["race_id"], r["has_vulnerable_fav"], r["fav_horse_key"],
                r["fav_v0"], r["fav_decay_rate"], r["fav_edge"],
                r["fav_finish_position"], r["fav_missed_board"],
                r["pace_scenario"], r["field_depth"], r["field_size"],
                r["usage_concentration"], r["separation_available"],
                r["n_positive_edge"],
                r["exacta_payoff"], r["trifecta_payoff"], r["super_payoff"],
                r["situation_type"],
            ])

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE handycapper.rkm_race_situations")
            with open(csv_path, "r") as f:
                with cur.copy(
                    "COPY handycapper.rkm_race_situations "
                    "(race_id, has_vulnerable_fav, fav_horse_key, "
                    "fav_v0, fav_decay_rate, fav_edge, "
                    "fav_finish_position, fav_missed_board, "
                    "pace_scenario, field_depth, field_size, "
                    "usage_concentration, separation_available, n_positive_edge, "
                    "exacta_payoff, trifecta_payoff, super_payoff, "
                    "situation_type) "
                    "FROM STDIN WITH (FORMAT csv, NULL '')"
                ) as copy:
                    for line in f:
                        copy.write(line)
        conn.commit()

    import os
    os.unlink(csv_path)
    log.info(f"Done. {len(results):,} race situations written.")


if __name__ == "__main__":
    main()
