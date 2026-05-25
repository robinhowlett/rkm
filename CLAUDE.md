# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

RKM (Race Kinetics Model) is a horse racing performance analytics system. It fits velocity-distance deceleration curves to individual horses from historical fractional timing data, then uses those curves to detect betting opportunities via Benter's market combination method and ITP's wagering framework.

The system reads from the `handycapper` PostgreSQL schema (populated by the sibling `pdf-importer` project) and writes derived analytics to `rkm_*` tables in the same schema.

## Setup & Running

```bash
# Install (requires Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Database: Postgres on robinpc via SSH tunnel
ssh -f -N -L 5434:127.0.0.1:5432 robinpc
# Connection: localhost:5434, user/pass: handycapper/handycapper, db: handycapper

# Run scripts (all are standalone, run from project root)
python scripts/compute_curves.py
python scripts/compute_adjustments.py
python scripts/compute_race_performance.py
python scripts/compute_market.py
python scripts/compute_form.py
python scripts/compute_situations.py
```

Scripts must run in order (each phase depends on prior phases). All use CSV + COPY for bulk writes (the SSH tunnel drops under sustained `executemany` load).

## Architecture

### The Model

Every horse's performance is described by a linear velocity-distance curve:

```
v(d) = v0 - decay_rate × (d / 1000)
```

- **v0** = initial velocity (ft/s) — raw speed / acceleration
- **decay_rate** = velocity lost per 1000ft — stamina / energy efficiency

Separate curves for sprint (≤6.5f) vs route (>6.5f) per horse per surface.

### Pipeline (6 phases, sequential)

| Phase | Module | Script | Output Table | What It Does |
|---|---|---|---|---|
| 1 | `curves.py` | `compute_curves.py` | `rkm_velocity_curves` | Fit v0 + decay_rate per horse from `indiv_fractionals` |
| 2 | `adjustments.py` | `compute_adjustments.py` | `rkm_track_offsets` | Normalize across tracks via shipping-horse network |
| 3 | `performance.py` | `compute_race_performance.py` | `rkm_race_performance` | Per-race surprise (actual vs predicted) + pace scenario |
| 4 | `market.py` | `compute_market.py` | `rkm_market_analysis` | Benter logit: combine model probability with odds |
| 5 | `form.py` | `compute_form.py` | `rkm_current_form` | Time-weighted curve (recent form vs career) |
| 6 | `situations.py` | `compute_situations.py` | `rkm_race_situations` | ITP situation classification (vulnerable favorites) |

### Supporting Modules

- `db.py` — connection helpers (`connect()` for dict_row, `connect_raw()` for pandas compatibility)
- `identity.py` — horse name disambiguation (`"Name|BirthYear"` keys, 5-year gap = new horse)
- `segments.py` — extract S1/S2/S3 segment times from `indiv_fractionals`

### Key Design Decisions

- **Linear deceleration** (not exponential): empirically equivalent fit over racing distances, no bound-hitting, 2 params vs 3
- **Sprint/route split**: different physics at different distances — separate curves per zone
- **Variant-adjusted surprise** (Phase 3): subtract race-level mean surprise before flagging individuals — eliminates "fast track day" artifacts
- **Benter combination** (Phase 4): model probability combined with odds via conditional logit (α=1.89, β=1.0, temperature=6500ms)
- **SPLIT_PART join**: `rkm_velocity_curves.horse_key` is `"Name|Year"` while `rkm_market_analysis.horse_key` is just name. Use `SPLIT_PART(vc.horse_key, '|', 1)` with expression index for joins.

## Database

Source tables (from pdf-importer): `races`, `starters`, `indiv_fractionals`, `points_of_call`, `fractionals`, `exotics`, `breeding`

RKM tables: `rkm_velocity_curves`, `rkm_track_offsets`, `rkm_race_performance`, `rkm_market_analysis`, `rkm_current_form`, `rkm_race_situations`, `rkm_race_shape` (retained from v1)

Data range: 1997-2016 (earlier years have coarser timing precision).

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `RKM_DATABASE_URL` | — | Full connection string (overrides individual vars) |
| `RKM_DB_HOST` | `localhost` | Postgres host |
| `RKM_DB_PORT` | `5434` | Postgres port (5434 = SSH tunnel to robinpc) |
| `RKM_DB_NAME` | `handycapper` | Database name |
| `RKM_DB_USER` | `handycapper` | Username |
| `RKM_DB_PASSWORD` | `handycapper` | Password |

## Specs & Context

- `docs/specs/rkm-v3.md` — full technical spec (Bayesian velocity curve model)
- `docs/specs/itp-wagering-framework.md` — ITP's professional wagering principles (the application layer)
- `docs/specs/rkm-elo.md` — deprecated Elo approach (for historical context only)
