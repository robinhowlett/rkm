SET search_path = handycapper;

CREATE TABLE IF NOT EXISTS rkm_velocity_curves (
    horse_key       varchar(60) NOT NULL,
    surface         varchar(12) NOT NULL,
    v0              decimal(5,2) NOT NULL,       -- initial velocity (ft/s)
    decay_rate      decimal(6,4) NOT NULL,       -- velocity loss per 1000ft (ft/s per 1000ft)
    n_observations  integer NOT NULL,
    n_races         integer NOT NULL,
    residual_std    decimal(5,3),
    first_race      date,
    last_race       date,
    computed_at     timestamp NOT NULL DEFAULT now(),
    PRIMARY KEY (horse_key, surface)
);

CREATE INDEX IF NOT EXISTS idx_curves_v0 ON rkm_velocity_curves(v0 DESC);
CREATE INDEX IF NOT EXISTS idx_curves_decay ON rkm_velocity_curves(decay_rate);
