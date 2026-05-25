SET search_path = handycapper;

CREATE TABLE IF NOT EXISTS rkm_market_analysis (
    starter_id          bigint PRIMARY KEY REFERENCES starters(id),
    race_id             bigint NOT NULL REFERENCES races(id),
    horse_key           varchar(60),
    model_prob          decimal(5,4),
    odds_prob           decimal(5,4),
    combined_prob       decimal(5,4),
    edge                decimal(5,4),
    won                 boolean,
    computed_at         timestamp NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_market_race ON rkm_market_analysis(race_id);
CREATE INDEX IF NOT EXISTS idx_market_edge ON rkm_market_analysis(edge DESC);
CREATE INDEX IF NOT EXISTS idx_market_won ON rkm_market_analysis(won) WHERE won = true;
