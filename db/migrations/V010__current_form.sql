SET search_path = handycapper;

CREATE TABLE IF NOT EXISTS rkm_current_form (
    starter_id          bigint PRIMARY KEY REFERENCES starters(id),
    race_id             bigint NOT NULL REFERENCES races(id),
    horse_key           varchar(60) NOT NULL,
    current_v0          decimal(5,2),
    current_decay       decimal(6,4),
    career_v0           decimal(5,2),
    career_decay        decimal(6,4),
    v0_trend            decimal(4,2),
    n_recent_races      smallint,
    days_since_last     smallint,
    computed_at         timestamp NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_form_race ON rkm_current_form(race_id);
CREATE INDEX IF NOT EXISTS idx_form_horse ON rkm_current_form(horse_key);
CREATE INDEX IF NOT EXISTS idx_form_trend ON rkm_current_form(v0_trend DESC);
