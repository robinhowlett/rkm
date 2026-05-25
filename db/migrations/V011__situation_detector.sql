SET search_path = handycapper;

-- Per-race situation analysis: identifies betting opportunities from ITP framework
CREATE TABLE IF NOT EXISTS rkm_race_situations (
    race_id                 bigint PRIMARY KEY REFERENCES races(id),
    -- Vulnerable favorite detection
    has_vulnerable_fav      boolean NOT NULL DEFAULT false,
    fav_horse_key           varchar(60),
    fav_v0                  decimal(5,2),
    fav_decay_rate          decimal(6,4),
    fav_edge                decimal(5,4),      -- model_prob - odds_prob (negative = overbet)
    fav_finish_position     smallint,
    fav_missed_board        boolean,
    -- Field dynamics
    pace_scenario           varchar(12),
    field_depth             smallint,          -- horses within competitive range
    field_size              smallint,
    -- Usage bias indicators
    usage_concentration     decimal(4,3),      -- how concentrated is the top 2 choices' probability
    separation_available    boolean,           -- are there positive-edge horses at 5/1+?
    n_positive_edge         smallint,          -- how many horses have model > odds
    -- Payoff results (what actually happened)
    exacta_payoff           decimal(8,2),
    trifecta_payoff         decimal(8,2),
    super_payoff            decimal(10,2),
    -- Situation classification
    situation_type          varchar(30),       -- ATTACK_VERTICAL, ATTACK_HORIZONTAL, SPREAD, PASS
    computed_at             timestamp NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_situation_vuln ON rkm_race_situations(has_vulnerable_fav);
CREATE INDEX IF NOT EXISTS idx_situation_type ON rkm_race_situations(situation_type);
CREATE INDEX IF NOT EXISTS idx_situation_tri ON rkm_race_situations(trifecta_payoff DESC) WHERE trifecta_payoff IS NOT NULL;
