"""Horse identity resolution — disambiguate reused names across decades."""

import logging
from collections import defaultdict

import pandas as pd

from rkm.db import connect_raw

log = logging.getLogger(__name__)

IDENTITY_QUERY = """
SELECT
    s.id AS starter_id,
    s.horse,
    r.date AS race_date,
    b.foaling_date
FROM handycapper.starters s
JOIN handycapper.races r ON r.id = s.race_id
LEFT JOIN handycapper.breeding b ON b.starter_id = s.id
WHERE s.horse IS NOT NULL
  AND r.breed = 'TB'
ORDER BY s.horse, r.date
"""

GAP_YEARS = 5


def build_horse_keys() -> dict[int, str]:
    """Build mapping of starter_id → horse_key for all starters.

    horse_key format: "HorseName|YYYY" where YYYY is estimated birth year.
    Same name with 5+ year gap between last and next race = different horse.
    """
    log.info("Loading starter/breeding data for identity resolution...")
    with connect_raw() as conn:
        df = pd.read_sql(IDENTITY_QUERY, conn)

    log.info(f"Loaded {len(df):,} starters for identity resolution")

    df["race_date"] = pd.to_datetime(df["race_date"])
    df["foaling_date"] = pd.to_datetime(df["foaling_date"], errors="coerce")

    horse_keys = {}
    grouped = df.groupby("horse", sort=False)

    for horse_name, group in grouped:
        group = group.sort_values("race_date")
        rows = group.to_dict("records")

        # Split into separate "horses" based on gaps
        segments = []
        current_segment = [rows[0]]

        for i in range(1, len(rows)):
            prev_date = rows[i - 1]["race_date"]
            curr_date = rows[i]["race_date"]
            gap_days = (curr_date - prev_date).days

            if gap_days > GAP_YEARS * 365:
                segments.append(current_segment)
                current_segment = [rows[i]]
            else:
                current_segment.append(rows[i])

        segments.append(current_segment)

        # Assign birth year for each segment
        for segment in segments:
            # Try foaling_date first
            foaling = next(
                (r["foaling_date"] for r in segment if pd.notna(r["foaling_date"])),
                None
            )
            if foaling is not None:
                birth_year = foaling.year
            else:
                first_race = segment[0]["race_date"]
                birth_year = first_race.year - 2

            key = f"{horse_name}|{birth_year}"
            for row in segment:
                horse_keys[int(row["starter_id"])] = key

    log.info(f"Resolved {len(horse_keys):,} starters into {len(set(horse_keys.values())):,} unique horses")
    return horse_keys
