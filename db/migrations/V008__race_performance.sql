SET search_path = handycapper;

CREATE TABLE IF NOT EXISTS rkm_race_performance (
    starter_id          bigint PRIMARY KEY REFERENCES starters(id),
    race_id             bigint NOT NULL REFERENCES races(id),
    horse_key           varchar(60) NOT NULL,
    predicted_v_avg     decimal(5,2),
    actual_v_avg        decimal(5,2),
    surprise            decimal(5,2),
    surprise_pct        decimal(4,2),
    pace_scenario       varchar(12),
    field_strength      decimal(5,2),
    field_size          smallint,
    is_interesting      boolean DEFAULT false,
    interest_reason     varchar(100),
    computed_at         timestamp NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_perf_race ON rkm_race_performance(race_id);
CREATE INDEX IF NOT EXISTS idx_perf_horse ON rkm_race_performance(horse_key);
CREATE INDEX IF NOT EXISTS idx_perf_surprise ON rkm_race_performance(surprise DESC);
CREATE INDEX IF NOT EXISTS idx_perf_interesting ON rkm_race_performance(is_interesting) WHERE is_interesting = true;
