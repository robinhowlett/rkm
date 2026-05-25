"""Extract segment times (S1/S2/S3) from individual fractionals."""

import logging

import pandas as pd

from rkm.db import connect_raw

log = logging.getLogger(__name__)

SEGMENTS_QUERY = """
SELECT
    s.id AS starter_id,
    r.id AS race_id,
    r.date AS race_date,
    r.furlongs,
    r.surface,
    r.number_of_runners,
    s.horse,
    MAX(CASE WHEN inf.point = 2 THEN inf.millis END) AS millis_p2,
    MAX(CASE WHEN inf.point = 3 THEN inf.millis END) AS millis_p3,
    MAX(CASE WHEN inf.point = 4 THEN inf.millis END) AS millis_p4,
    MAX(CASE WHEN inf.point = 6 THEN inf.millis END) AS millis_p6
FROM handycapper.races r
JOIN handycapper.starters s ON s.race_id = r.id
JOIN handycapper.indiv_fractionals inf ON inf.starter_id = s.id
WHERE r.breed = 'TB'
  AND r.furlongs IS NOT NULL
  AND r.number_of_runners >= 4
  AND r.surface IN ('Dirt', 'Turf', 'Synthetic')
  AND inf.millis IS NOT NULL
  AND r.date BETWEEN %(start_date)s AND %(end_date)s
GROUP BY s.id, r.id, r.date, r.furlongs, r.surface, r.number_of_runners, s.horse
HAVING MAX(CASE WHEN inf.point = 6 THEN inf.millis END) IS NOT NULL
   AND MAX(CASE WHEN inf.point = 2 THEN inf.millis END) IS NOT NULL
ORDER BY r.date, r.id, s.id
"""


def load_segments(start_date, end_date) -> pd.DataFrame:
    """Load and compute segment times for all starters in date range.

    Returns DataFrame with columns:
        starter_id, race_id, race_date, furlongs, surface, number_of_runners,
        horse, s1_ms, s2_ms, s3_ms, is_route
    """
    log.info(f"Loading segment data for {start_date} to {end_date}...")
    with connect_raw() as conn:
        df = pd.read_sql(SEGMENTS_QUERY, conn, params={
            "start_date": start_date, "end_date": end_date
        })

    log.info(f"Loaded {len(df):,} starters with timing data")

    for col in ["millis_p2", "millis_p3", "millis_p4", "millis_p6", "furlongs", "number_of_runners"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_route"] = df["furlongs"] > 6.5

    # Routes (7f+): S1 = start→half (p2), S2 = half→3/4 or stretch (p4 or p3), S3 = last→finish
    # Sprints (≤6.5f): S1 = start→half (p2), S3 = last_intermediate→finish (p3→p6), S2 = None
    #
    # For routes: use p4 (3/4 mile) as S2 boundary if available, else p3
    # For sprints: S1 = p2, S3 = p6 - p3 (or p6 - p2 if p3 missing)

    def compute_segments(row):
        p2, p3, p4, p6 = row["millis_p2"], row["millis_p3"], row["millis_p4"], row["millis_p6"]
        is_route = row["is_route"]

        if pd.isna(p2) or pd.isna(p6):
            return pd.Series({"s1_ms": None, "s2_ms": None, "s3_ms": None})

        if is_route:
            # S1 = start → half
            s1 = p2
            # S2 = half → last intermediate (prefer p4, fallback p3)
            mid = p4 if pd.notna(p4) else p3
            if pd.notna(mid):
                s2 = mid - p2
                s3 = p6 - mid
            else:
                s2 = None
                s3 = p6 - p2  # fallback: entire second half
        else:
            # Sprints: 2 segments
            s1 = p2
            # S3 = last intermediate to finish
            last_int = p3 if pd.notna(p3) else p2
            s3 = p6 - last_int
            s2 = None

        return pd.Series({"s1_ms": s1, "s2_ms": s2, "s3_ms": s3})

    segments = df.apply(compute_segments, axis=1)
    df = pd.concat([df, segments], axis=1)

    valid = df["s1_ms"].notna() & df["s3_ms"].notna() & (df["s1_ms"] > 0) & (df["s3_ms"] > 0)
    df = df[valid].copy()

    log.info(f"Computed segments for {len(df):,} starters ({df['is_route'].sum():,} routes, {(~df['is_route']).sum():,} sprints)")

    return df[["starter_id", "race_id", "race_date", "furlongs", "surface",
               "number_of_runners", "horse", "s1_ms", "s2_ms", "s3_ms", "is_route"]]
