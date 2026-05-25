"""Multi-outcome Kelly staking for pari-mutuel horse racing.

Given model probabilities (from RKM) and market-implied probabilities (from odds),
compute optimal bet fractions per horse using the pari-mutuel Kelly criterion.

Key concepts:
- In pari-mutuel, outcomes are mutually exclusive (only one horse wins)
- The "edge ratio" r_i = model_prob_i / implied_prob_i tells you which horses are overlays
- Full Kelly maximizes log-growth but has brutal variance
- Fractional Kelly (1/4 to 1/2) is practical for real betting

Reference: octonion/betting (Christopher D. Long)
"""

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class StakingResult:
    """Result of Kelly staking computation for one race."""
    horses_to_bet: list[int]       # indices of horses worth betting
    fractions: list[float]         # fraction of bankroll per horse (same order as input)
    total_fraction: float          # total bankroll fraction deployed
    expected_growth: float         # expected log-growth rate at this allocation
    edge_ratios: list[float]      # model_prob / implied_prob per horse


def compute_kelly_fractions(model_probs: list[float], odds_probs: list[float],
                            takeout: float = 0.20,
                            kelly_fraction: float = 0.25) -> StakingResult:
    """Compute optimal Kelly bet fractions for a pari-mutuel race.

    Args:
        model_probs: model's estimated win probability per horse (sum ≈ 1)
        odds_probs: market-implied probabilities per horse (sum ≈ 1 after normalization)
        takeout: pool takeout rate (e.g., 0.20 for trifecta)
        kelly_fraction: multiplier for fractional Kelly (0.25 = quarter Kelly)

    Returns:
        StakingResult with bet fractions and supporting metrics
    """
    p = np.array(model_probs, dtype=float)
    ip = np.array(odds_probs, dtype=float)

    # Ensure valid inputs
    n = len(p)
    if n < 2 or np.sum(p) < 0.5:
        return StakingResult([], [0.0] * n, 0.0, 0.0, [0.0] * n)

    # Normalize probabilities
    p = p / p.sum()
    ip = ip / ip.sum()

    # Edge ratio: how much better does the model think each horse is vs the market?
    edge_ratios = np.where(ip > 1e-10, p / ip, 0.0)

    # Sort by edge ratio (best overlays first)
    sorted_idx = np.argsort(-edge_ratios)

    # Build the active set: horses where model_prob > implied_prob (positive edge)
    # Then compute optimal fractions using the multi-outcome Kelly formula
    active = [int(i) for i in sorted_idx if edge_ratios[i] > 1.0]

    if not active:
        return StakingResult([], [0.0] * n, 0.0, 0.0, edge_ratios.tolist())

    # Compute Kelly fractions for active horses
    # For pari-mutuel with mutually exclusive outcomes:
    # f_i = p_i - (1-p_i) / (o_i - 1) where o_i = 1/ip_i (decimal odds after takeout)
    # Simplified: f_i = p_i - ip_i * (1 - sum_active_p) / (1 - sum_active_ip)
    # But the simpler individual Kelly works well for pari-mutuel:
    # f_i = (p_i * o_i - 1) / (o_i - 1) where o_i = (1-takeout) / ip_i
    full_kelly = np.zeros(n)
    for i in active:
        # Effective decimal odds after takeout
        effective_odds = (1.0 - takeout) / ip[i] if ip[i] > 1e-10 else 1.0
        if effective_odds > 1.0:
            # Standard Kelly: f = (b*p - q) / b where b = odds-1, p = win prob, q = 1-p
            b = effective_odds - 1.0
            f_i = (b * p[i] - (1.0 - p[i])) / b
            full_kelly[i] = max(0.0, f_i)
        # If effective_odds <= 1 after takeout, no bet (negative expected value)

    total_full_kelly = full_kelly.sum()

    # Apply fractional Kelly
    fractions = full_kelly * kelly_fraction

    # At fractional Kelly, some marginal horses may go negative
    # (their edge doesn't justify betting at reduced fraction)
    # Remove them from the active set
    final_active = [i for i in active if fractions[i] > 1e-6]
    for i in range(n):
        if i not in final_active:
            fractions[i] = 0.0

    total_fraction = fractions.sum()

    # Expected log-growth at this allocation
    growth = 0.0
    for i in range(n):
        return_if_wins = 1.0 - total_fraction + fractions[i] / ip[i] * (1 - takeout) if ip[i] > 1e-10 else 1.0
        if return_if_wins > 0:
            growth += p[i] * np.log(return_if_wins)
        else:
            growth += p[i] * np.log(1e-10)  # catastrophic loss

    return StakingResult(
        horses_to_bet=final_active,
        fractions=fractions.tolist(),
        total_fraction=float(total_fraction),
        expected_growth=float(growth),
        edge_ratios=edge_ratios.tolist(),
    )


def compute_bet_amounts(staking: StakingResult, bankroll: float,
                        min_bet: float = 2.0) -> dict[int, float]:
    """Convert Kelly fractions to actual dollar amounts.

    Args:
        staking: result from compute_kelly_fractions
        bankroll: total available bankroll
        min_bet: minimum bet size (below this, don't bother)

    Returns:
        dict of {horse_index: dollar_amount} for horses worth betting
    """
    bets = {}
    for i in staking.horses_to_bet:
        amount = staking.fractions[i] * bankroll
        if amount >= min_bet:
            bets[i] = round(amount, 2)
    return bets


def summarize_race_staking(model_probs: list[float], odds: list[float],
                           horse_names: list[str] = None,
                           bankroll: float = 1000.0,
                           takeout: float = 0.20,
                           kelly_fraction: float = 0.25) -> str:
    """Produce a human-readable staking summary for a race.

    Args:
        model_probs: model win probabilities
        odds: decimal odds per horse (e.g., 5.0 means 4/1)
        horse_names: optional names for display
        bankroll: total bankroll
        takeout: pool takeout
        kelly_fraction: Kelly multiplier
    """
    # Convert odds to implied probabilities
    odds_probs = [1.0 / o if o > 0 else 0.01 for o in odds]
    total_ip = sum(odds_probs)
    odds_probs = [p / total_ip for p in odds_probs]

    result = compute_kelly_fractions(model_probs, odds_probs, takeout, kelly_fraction)
    bets = compute_bet_amounts(result, bankroll)

    if not horse_names:
        horse_names = [f"Horse {i+1}" for i in range(len(model_probs))]

    lines = [f"STAKING ANALYSIS (Kelly={kelly_fraction:.0%}, Bankroll=${bankroll:.0f}, Takeout={takeout:.0%})"]
    lines.append("=" * 70)
    lines.append(f"{'Horse':<20} {'Model%':>7} {'Odds%':>7} {'Edge':>6} {'Fraction':>9} {'Bet':>8}")
    lines.append("-" * 70)

    for i in range(len(model_probs)):
        name = horse_names[i][:20]
        mp = model_probs[i] * 100
        op = odds_probs[i] * 100
        edge = result.edge_ratios[i]
        frac = result.fractions[i] * 100
        bet = bets.get(i, 0)

        marker = " ★" if i in result.horses_to_bet else ""
        lines.append(f"{name:<20} {mp:>6.1f}% {op:>6.1f}% {edge:>5.2f}x {frac:>7.2f}% ${bet:>7.2f}{marker}")

    lines.append("-" * 70)
    lines.append(f"Total deployed: {result.total_fraction*100:.2f}% of bankroll (${result.total_fraction*bankroll:.2f})")
    lines.append(f"Expected log-growth: {result.expected_growth:.6f}")
    lines.append(f"Horses to bet: {len(result.horses_to_bet)} of {len(model_probs)}")

    return "\n".join(lines)
