# Race Kinetics Model (RKM)

A horse racing performance analytics system that models each horse's characteristic velocity-distance curve and uses it to detect betting opportunities.

## The Model

Every horse's performance is described by a linear deceleration curve:

```
v(d) = v0 - decay_rate × (d / 1000)
```

- **v0** — initial velocity (ft/s). Captures raw speed and acceleration.
- **decay_rate** — velocity lost per 1000ft of distance. Captures stamina and energy efficiency.

Fitted from individual fractional timing data (feet + milliseconds at each call point) across a horse's career. Separate curves for sprint (≤6.5f) and route (>6.5f) per surface.

## What It Produces

| Output | What It Tells You |
|---|---|
| **Velocity curve** | How fast this horse goes and how quickly it slows down |
| **Track-adjusted v0** | Cross-track comparable speed (normalized via shipping-horse network) |
| **Race surprise** | Did this horse run above or below its curve? (variant-adjusted) |
| **Pace scenario** | CONTESTED / PRESSURED / CONTROLLED (from field v0 distribution) |
| **Market edge** | Where does the model disagree with public odds? (Benter combination, α=1.89) |
| **Current form** | Time-weighted curve — is the horse improving or declining? |
| **Situation type** | ITP classification: ATTACK_VERTICAL, SPREAD, PASS |

## Key Findings

- The model contributes ~2× the weight of public odds in a Benter conditional logit (α=1.89, β=1.0)
- When the model flags a "vulnerable favorite" (high decay + contested pace + negative edge), they miss the board 36.3% of the time (vs ~31% baseline)
- Trifectas pay 4.1× more ($1,158 vs $280) when model-flagged favorites fail
- Elite horses surface correctly: Arrogate (decay 0.34), Zenyatta (0.11), Ghostzapper (v0 62.6)

## Pipeline

```
pdf-importer → PostgreSQL (handycapper schema)
                    ↓
Phase 1: compute_curves.py        → rkm_velocity_curves (514K horses)
Phase 2: compute_adjustments.py   → rkm_track_offsets (104 tracks)
Phase 3: compute_race_performance → rkm_race_performance (7.6M starters)
Phase 4: compute_market.py        → rkm_market_analysis (7.8M starters)
Phase 5: compute_form.py          → rkm_current_form (5.5M snapshots)
Phase 6: compute_situations.py    → rkm_race_situations (182K races)
```

Phases run sequentially. Each reads from prior phases' output tables. Total compute time: ~3-4 hours for full corpus (1997-2016).

## Theoretical Foundations

| Source | Contribution |
|---|---|
| **Wilkins** (Bioenergetics and Racehorse Ratings) | Energy balance equation; velocity declines as anaerobic store depletes. The deceleration model is grounded in physiology. |
| **Benter** (Computer Based Horse Race Handicapping, 1994) | Combine a fundamental model with public odds via conditional logit. The market knows most things — find where it's wrong. |
| **Silverman** (Hierarchical Bayesian Analysis, 2012) | Partial pooling across race profiles handles sparse data. Track × surface × condition grouping. |
| **ITP** (Inside the Pylons, professional horseplayer) | Bad favorites are the most exploitable situation. Separation from the crowd IS the edge. Exotic pools are less efficient than win pools. |

## Setup

Requires Python 3.11+ and access to the `handycapper` PostgreSQL database (populated by [pdf-importer](https://github.com/robinhowlett/pdf-importer)).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configure database (defaults shown)
export RKM_DB_HOST=localhost
export RKM_DB_PORT=5432
export RKM_DB_NAME=handycapper
export RKM_DB_USER=handycapper
export RKM_DB_PASSWORD=handycapper

# Run (in order — each phase writes to the database)
python scripts/compute_curves.py           # → rkm_velocity_curves
python scripts/compute_adjustments.py      # → rkm_track_offsets
python scripts/compute_race_performance.py # → rkm_race_performance
python scripts/compute_market.py           # → rkm_market_analysis
python scripts/compute_form.py             # → rkm_current_form
python scripts/compute_situations.py       # → rkm_race_situations
```

Each script TRUNCATEs its target table and COPYs fresh results. These are full recomputes (~3-4 hours total for 1997-2016 corpus).

## Related Projects

- [chart-parser](https://github.com/robinhowlett/chart-parser) — extracts structured data from Equibase PDF race charts
- [pdf-importer](https://github.com/robinhowlett/pdf-importer) — bulk loads chart-parser output into PostgreSQL
- [wagering-analytics](https://github.com/robinhowlett/wagering-analytics) — Harville/Stern calibration + payoff prediction models
- [race-day-sim](https://github.com/robinhowlett/race-day-sim) — blinded backtesting (reads rkm output tables)
- [redboarders](https://github.com/robinhowlett/redboarders) — Bet Doctor wagering analysis + Redboarders handicapping game
