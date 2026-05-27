# RKM v3: Bayesian Velocity Curve Model — Specification

## Prior Attempts

### v1: Par-Based Speed Figures

**Approach:** Traditional Carroll methodology — compute daily track variant from the full card, derive a single "figure" per horse per race from final time adjusted by variant. Extended with segment-level figures (S1/S2/S3) to capture energy allocation, and Field Dispersion Score (FDS) for race shape.

**What worked:** Segment decomposition was valuable (speed vs stamina is real). FDS correctly identified compressed vs dispersed fields. The Wilkins bioenergetics framing (aerobic P vs anaerobic E) was the right theoretical lens.

**What failed:** Cross-track normalization. Par times were self-referential — tracks with small fields produced inflated pars. Isolated tracks with few shipping horses had no anchor to the broader network. A horse could look like a 100-figure at a bush track and a 75-figure at Belmont, with no principled way to reconcile.

**Archived:** `docs/specs/archive/rkm.md`

### v2: Elo Ratings

**Approach:** Replace self-referential pars with a connected graph. Ratings earned by beating rated opponents — naturally normalizes across tracks because shipping horses carry their ratings with them. Segment-level Elo (S1, S2, S3 ratings per horse per surface).

**What worked conceptually:** The "horses rate horses" principle is sound. Daily variant emerges naturally (shared residual across a field). The connected graph does solve the cross-track problem in theory.

**What failed:** Ratings exploded. The K-factor tuning was unstable — too high and ratings oscillated wildly; too low and they couldn't track form changes. The Elo framework assumes a stable underlying skill with noise, but horse performance has real trends (maturation, decline, form cycles) that Elo handles poorly. Also, the segment-level decomposition added noise (computing 3 separate ratings from the same race created instability). Ultimately, the Elo abstraction was fighting the data rather than describing it.

**Archived:** `docs/specs/archive/rkm-elo.md`

### What v3 learned from v1 and v2

- The DATA is velocity observations at points in the race. Work with it directly rather than abstracting to figures or ratings.
- Cross-track normalization needs an explicit network solution (shipping horses), not a self-referential one (pars) or an implicit one (Elo propagation).
- Two parameters (speed + decay) describe the physics better than three ratings (S1/S2/S3) because they're on a continuous physical scale rather than arbitrary units.
- Form cycles and trends are real — the model needs a time-weighted current estimate, not just a career aggregate.

---

## Problem Statement

Previous iterations attempted to reduce horse performance to discrete figures (v1) or relative ratings (v2/Elo). Both struggled with cross-track normalization and failed to capture the fundamental nature of the data: **a horse's performance is a continuous velocity curve over distance**, and that curve is shaped by biophysical constraints.

The data supports this directly — we have (feet, millis) pairs at 4-6 points per race for every starter. This is enough to model each horse's characteristic velocity-distance curve, compare curves between horses, and predict how curves will interact in a specific race scenario.

## Theoretical Synthesis

| Source | Contribution |
|---|---|
| **Wilkins** | The velocity curve follows a bioenergetically-constrained shape: initial acceleration (creatine), plateau (anaerobic E), gradual deceleration (aerobic P sustains as E depletes). The P and E parameters define the curve's shape. |
| **Benter** | The best predictions combine a fundamental model with public odds. Don't fight the market — combine with it. Factor selection is empirical, not theoretical. |
| **Silverman** | Bayesian hierarchical models handle the "many profiles, sparse data per cell" problem elegantly through partial pooling. Group by track × distance × surface × condition. |
| **Montagna** | Individual performance trajectories modeled as smooth curves with uncertainty that widens where data is sparse. Career arcs, form cycles, maturation. |

## Core Idea

Every horse has a **characteristic velocity-distance curve** — how fast it travels at each point in the race. This curve is:

1. **Biophysically shaped** (Wilkins: deceleration follows energy depletion)
2. **Individually parameterized** (each horse has its own P and E, estimated from observed data)
3. **Context-modified** (the same horse runs a different curve depending on pace scenario, surface, condition, fitness)
4. **Uncertain** (confidence narrows with more observations, widens with layoffs or new conditions)

## Data Model

### The observation

Each starter in each race produces a set of (distance, time) pairs:

```
(feet_1, millis_1), (feet_2, millis_2), ..., (feet_n, millis_n)
```

From `indiv_fractionals`: typically 4-6 points per race (quarter poles + finish).

From these, we derive instantaneous velocity at each segment:
```
v_segment = (feet_i+1 - feet_i) / (millis_i+1 - millis_i) × 1000  [feet/sec]
```

### The curve

For each horse, across all their races at a given surface, we observe multiple velocity curves. The model estimates:

```
v(d) = velocity at distance d for this horse
     = f(d; P, E, v0, conditions, pace_context) + noise
```

Where:
- `v0` = initial velocity (from gate speed / acceleration)
- `E` = anaerobic energy store (determines how long velocity is maintained)
- `P` = aerobic power (determines the floor velocity / rate of deceleration)
- `conditions` = surface, going, track-specific adjustments
- `pace_context` = how much energy was spent early (from pace scenario)

### The Wilkins-inspired functional form

Simplified energy balance: velocity declines as anaerobic store depletes:

```
v(d) = P + (v0 - P) × exp(-d / E_distance)
```

Where `E_distance` is the characteristic distance over which the anaerobic advantage decays. This produces:
- High velocity early (v ≈ v0 when d is small)
- Exponential decay toward aerobic floor (v → P as d → ∞)
- The RATE of decay is governed by E (large E = slower decay = more stamina-speed)

This is a 3-parameter curve (v0, P, E_distance) that we estimate per horse.

## Hierarchical Structure

### Levels

```
Population level:     mean curve parameters for all TB racehorses
   ↓ partial pooling
Profile level:        mean curve for this (track × surface × condition × class)
   ↓ partial pooling  
Horse level:          this specific horse's curve parameters (v0_i, P_i, E_i)
   ↓ observation noise
Race level:           this horse's actual velocity at each point in this specific race
```

The Bayesian hierarchical structure means:
- A horse with 1 start borrows heavily from the profile-level prior (we don't know much yet)
- A horse with 20 starts has a well-estimated individual curve (data dominates)
- A horse switching surfaces uses their other-surface estimate as an informative prior
- A track with few races borrows from similar tracks (partial pooling across the network)

### Cross-track normalization (solved by design)

Tracks don't have different "pars" — they have different **profile-level priors** that are linked through shared horses. A horse that races at both BEL and FL contributes information to both profiles simultaneously. The partial pooling naturally calibrates:
- If FL horses consistently show lower P values than BEL horses, the FL profile prior reflects this
- A new FL horse inherits the FL prior, but if it ships to BEL and performs well, its individual estimate rises above the FL prior

No explicit track offsets needed. The hierarchical structure IS the normalization.

## Race-Level Model (Prospective)

Given a field of entered horses with estimated curves:

### 1. Pace prediction

From each horse's estimated v0 and early-race behavior, predict:
- Who will establish the pace (highest v0 + front-running history)
- How contested it will be (multiple high-v0 horses = contested)
- The expected "energy cost" of the early pace for each horse

### 2. Curve interaction

In a contested pace, speed horses spend more E early → their curves drop faster in the late stages. This shifts the expected crossover points:

```
                      ← Crossover: closer catches speed horse here
  ft/s
  57 ─ Speed horse (contested): v(d) = 52 + 5×exp(-d/3000)
  56 ─
  55 ─          ╲
  54 ─           ╲        ╱ Closer: v(d) = 54 + 1×exp(-d/5000)
  53 ─            ╲──────●╱
  52 ─             ╲╱─────────────── Both converge to P
       ──┼────┼────┼────┼────┼──
       0    1320  2640  3960  5280 feet
```

### 3. Win probability

Convert the predicted finishing times (from integrating each horse's expected velocity curve) into win probabilities, adjusted for uncertainty:

```
P(horse_i wins) = P(horse_i finishes first)
                = integral over all possible curve realizations (given uncertainty)
```

Monte Carlo simulation: sample from each horse's posterior curve parameters, simulate thousands of races, count wins.

### 4. Market combination (Benter's insight)

```
final_probability = logit_combine(model_probability, odds_implied_probability)
```

The alpha/beta weights tell you how much the model adds beyond what the public already knows.

## Race-Level Model (Retrospective)

After a race:

### 1. Actual vs Expected curve

Overlay the horse's actual velocity points against its predicted curve:

```
CIGAR — BEL Woodward S., 9f, 1995-09-16
ft/sec
58 ┤
57 ┤     ●━━━━━━●━━━━━━━●━━━━━━●━━━━●  ACTUAL (career best)
   │   ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱  ← Outperformed expectation
56 ┤  ╱  ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈  EXPECTED (from prior curve)
   │ ╱         ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
55 ┤╱                                    
   │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  95% credible band
54 ┤░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
   ┼────┼────┼────┼────┼
   Start  ¼    ½    ¾    Fin

   SURPRISE: +1.2 ft/s sustained above expected
   INTERPRETATION: Maintained higher velocity throughout (P higher than estimated?)
   → UPDATE: P estimate increases, confidence tightens
```

### 2. "Interesting" detection

A performance is interesting when:
- Actual curve deviates significantly from expected (outside credible band)
- The deviation has a narrative: pace-related? fitness breakthrough? surface preference?
- Market disagreed: odds implied X% but the curve model implied Y%

## End-User Presentation

### Horse Card (current estimate)

```
CIGAR — Dirt Route Profile (14 observations)
══════════════════════════════════════════════
  VELOCITY CURVE PARAMETERS:
    v0 (initial speed):    57.1 ft/s ± 0.4   [elite acceleration]
    P  (aerobic floor):    54.8 ft/s ± 0.3   [exceptional sustain]
    E  (stamina decay):    4200 ft ± 300      [decays slowly]

  CHARACTERISTIC CURVE:
    57 ┤●━━━━━━●
    56 ┤        ╲━━━━●
    55 ┤              ╲━━━━●━━━━●
    54 ┤                          ╲━━  (P floor)
       ┼────┼────┼────┼────┼────┼
       0   1320  2640  3960  5280  6600

  PROFILE: PRESSER (high v0, very high P — barely decelerates)
  OPTIMAL: 8.5-10f routes, any pace scenario
  VULNERABLE: None identified (elite across all contexts)
  
  FORM: ↑ trending (last 3 above career mean)
  CONFIDENCE: HIGH (14 starts, 0.92)
```

### Race Preview (prospective)

```
Race 9: BEL Woodward S., 9f Dirt
════════════════════════════════════════
PREDICTED PACE: CONTESTED (Cigar + Devil His Due both high v0)
FDS FORECAST: High early dispersion → late compression

VELOCITY CURVE COMPARISON:
ft/sec
58 ┤  ╭── Cigar
57 ┤──●━━━●━━━━●━━━━●━━━●  barely decelerates
56 ┤                         
55 ┤──●━━━●                  Devil His Due
54 ┤        ╲━━●             decelerates faster (lower P)
53 ┤            ╲━━●━━●     
52 ┤
   ┼────┼────┼────┼────┼
   Start  ¼    ½    ¾    Fin

  #  Horse           Win%   Odds  Edge   Profile
  1  Cigar           58%    1/2   +8%    Presser (barely decays)
  2  Devil His Due   18%    5/1   -2%    Speed (decays in contested)
  3  Unaccounted For 12%    8/1   +4%    Closer (benefits from pace)
```

### Post-Race Diagnosis

```
Race 9: BEL Woodward S. — ACTUAL RESULT
════════════════════════════════════════
WINNER: Cigar (1:47.2) — as predicted

CURVE ANALYSIS:
  Cigar:    Ran ABOVE expected by +0.8 ft/s throughout. Career best.
            → P estimate updated: 54.8 → 55.2 (confidence tightens)
  
  Devil His Due: Ran AS EXPECTED for 6f, then collapsed 2 ft/s below curve
            → Contested pace cost him. E depleted early (as predicted by model)
            → No update needed — behaved exactly as curve predicted in this scenario

  Unaccounted For: Started slow (as expected) but closed only 50% of predicted gap
            → UNDERPERFORMED. Possible trip trouble? Check points_of_call wide data.
            → Flag for investigation.

MOST INTERESTING: Unaccounted For's underperformance — the pace setup was perfect 
for his profile but he didn't capitalize. Why?
```

## Data Scope & Quality

### Date range: 1991-2017

~~Drop 1991-1996 (coarser timing precision, 5% at 200ms rounding vs 2% post-2000). 20 years of high-quality data is sufficient.~~

**Updated (research-plan Item 0b, 2026-05-26):** Empirical analysis shows no meaningful 200ms rounding cliff. The "5% at 200ms" is consistent with random chance (1/200 = 0.5% expected; observed 5-6% reflects some values being naturally round, not systematic rounding). All years 1991-2017 have equivalent timing precision (~10ms granularity), consistent indiv_fractional coverage (3.4-3.5 points/starter), and stable points-of-call counts (5.3/starter). Expanding to 1991-2017 adds ~450K races and 27 years of data. 2018-2019 excluded (partial single-track import).

### Observation exclusion criteria

Exclude individual observations (not entire horses) that represent non-performances:

| Criterion | Rationale |
|---|---|
| Total lengths behind > 30 at any call | Catastrophic event, not a performance measurement |
| Finishing time > 2× race field mean | Pulled up, walked home |
| Negative segment velocity (impossible) | Data error |
| Last place, beaten > 20 lengths, odds < 10/1 | Well-regarded horse that collapsed (injury/equipment) |

Milder outliers handled by **robust regression** (Huber loss) — downweighted rather than excluded.

## Cross-Track Normalization

### Track-pair adjustments from shipping horses

Computed empirically (already done in v1 exploration):
- For every pair of tracks with 50+ shared horses (3+ starts at each)
- Average figure difference = the systematic bias between tracks
- Solve for per-track offset that minimizes total pairwise error (network least-squares)

Applied post-curve-fit: adjust each horse's fitted parameters by their primary track's offset.

### Competition tier (hierarchical grouping signal)

Derived from structural race properties (NOT odds):

| Factor | How it's used |
|---|---|
| Grade (Gr1, Gr2, Gr3, Listed, ungraded stakes) | Direct quality indicator |
| Purse percentile within track | Relative importance at that venue |
| Class conditions (maiden, claiming, allowance, stakes) | Entry restrictions determine population |
| Track's shipping connectivity | Well-connected tracks have better-calibrated data |

These define the hierarchical grouping for partial pooling: horses in similar competition tiers share group-level priors.

## Role of Odds

Odds are used **within-race only** (Benter stage-2 combination):

```
final_win_prob = logit_combine(model_predicted_prob, odds_implied_prob)
```

Odds do NOT inform:
- The velocity curve parameters (those come from timing data)
- The hierarchical grouping (that comes from grade/purse/class)
- Cross-track normalization (that comes from shipping data)

Odds tell you: "within this specific race, how does the market rank these horses?" — a powerful but race-local signal.

## Implementation Approach

### Phase 1: Curve fitting (fast, parallel)

For each horse × surface (1997-2016, ~400K horses):
- Collect all (feet, millis) pairs across their career (excluding bad observations)
- Convert to segment velocities: `v = Δfeet / Δmillis × 1000`
- Fit: `v(d) = P + (v0 - P) × exp(-d / E_dist)` via scipy robust curve_fit
- Store: (horse_key, surface, v0, P, E_dist, n_observations, residual_std)
- **Estimated time: 5-10 minutes** (parallel, independent fits)

### Phase 2: Hierarchical pooling (iterative, 2-3 passes)

- Group horses by (track × surface × condition × competition_tier)
- Compute group-level priors: (mean_v0, mean_P, mean_E, variance of each)
- Re-fit each horse with shrinkage toward group mean (weight by n_observations)
- Iterate 2-3× until stable
- **Estimated time: 15-30 minutes**

### Phase 3: Track-pair adjustment

- Compute shipping-horse offsets from Phase 2 fitted curves
- Solve network least-squares for per-track offset
- Apply offset to each horse's parameters
- **Estimated time: 5 minutes** (one-time computation)

### Phase 4: Race context + pace interaction

- For each race: predict pace scenario from field v0 values
- Adjust E_dist for pace cost (contested → faster E depletion for leaders)
- Predict finishing times by integrating adjusted curves
- Compute FDS from predicted vs actual velocity spreads
- **Estimated time: 10-15 minutes** (one pass through all races)

### Phase 5: Market combination (Benter stage 2)

- Convert predicted times → win probabilities (via Monte Carlo or analytical)
- Fit second-stage logit: combine(model_prob, odds_prob)
- Identify divergences as "interesting"
- **Estimated time: 5 minutes** (logistic regression)

### Total estimated compute: ~45-60 minutes (one-shot, no multi-pass)

## Tech Stack

- **scipy.optimize** — curve fitting (Phase 1)
- **numpy/pandas** — data manipulation and hierarchical aggregation (Phase 2-3)
- **Existing Postgres data** — indiv_fractionals source, results stored back
- **statsmodels** — logistic regression for Benter combination (Phase 5)
- **Plotly** (later) — interactive curve visualizations for end user

## What v1/v2 Provided That Carries Forward

| Component | Status |
|---|---|
| `indiv_fractionals` data extraction | Reuse — same source, narrowed to 1997-2016 |
| FDS / race shape (`rkm_race_shape`) | Reuse — still valid field-level metric |
| Horse identity resolution | Reuse from v2 (`identity.py`) |
| Shipping-horse track offsets | Reuse — already computed, applies directly |
| Segment time extraction | Refactor — now produces (feet, velocity) pairs rather than S1/S2/S3 |
| Par tables | Replaced — profile-level priors replace fixed pars |
| Elo ratings | Replaced entirely |
| Daily variant | Replaced — emerges as group-level residual in hierarchical model |
| v1 figures (gen0) | Deprecated — velocity curves replace single-number figures |

## What Was Actually Built

The implementation simplified the spec's Bayesian hierarchical approach into a pragmatic pipeline that produces equivalent outputs with less complexity:

| Spec intent | Implementation | Rationale |
|---|---|---|
| 3-parameter exponential: `v(d) = P + (v0-P) × exp(-d/E_dist)` | 2-parameter linear: `v(d) = v0 - decay_rate × (d/1000)` | Over racing distances (1000-7000 ft), the exponential and linear fits are empirically indistinguishable. Linear has no bound-hitting, converges reliably, and 2 params > 3 for sparse data. |
| Bayesian partial pooling across tracks | Shipping-horse network least-squares offsets (Phase 2) | Achieves the same cross-track normalization with simpler computation. 104 tracks calibrated via shared horses. |
| Monte Carlo win probability | Softmax on predicted finishing times (temperature=6500ms) | Analytically tractable. Calibrated via Benter conditional logit (α=1.89) against actual win frequencies. |
| Per-race curve adjustment for pace cost | Pace classification (CONTESTED/PRESSURED/CONTROLLED) without modifying curve parameters | Pace shapes qualitative assessment; the curve itself is treated as stable. Race-level effects appear in `rkm_race_performance.surprise`. |
| Hierarchical grouping by track × surface × condition × class | Separate curves per horse × surface × distance_zone (sprint/route) | Simpler grouping. Class effects captured implicitly through the shipping-horse network. |

The theoretical foundations (Wilkins bioenergetics, Benter market combination, Silverman hierarchical intuition) remain the intellectual basis. The implementation chose computational simplicity over theoretical elegance where the data couldn't distinguish between them.

See [rkm/CLAUDE.md](https://github.com/robinhowlett/rkm) for the actual pipeline as built.

## Open Questions (from original spec, partially resolved)

| # | Question | Resolution |
|---|---|---|
| 1 | 3-parameter curve sufficient? | Resolved: 2 parameters (linear) are sufficient. Residuals don't show systematic curvature that 3 params would capture. |
| 2 | Sprints vs routes: same model? | Resolved: same functional form, separate fits per zone. Sprint curves have higher v0 and higher decay. |
| 3 | Distance switches? | Partially resolved: separate sprint/route curves. Cross-zone prediction remains a research question (see `race-day-sim/docs/research-plan.md` item 11). |
| 4 | Robust fit method? | Resolved: Huber regression with sklearn. Handles outliers without exclusion. |
| 5 | Minimum observations? | Resolved: 12+ velocity points (from 3+ races with 4+ calls each). |
| 6 | Visualization? | Deferred to the output layer. See `race-day-sim/docs/rating-calibration-plan.md`. |
