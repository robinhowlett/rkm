"""Situation detector — identifies races with ITP-style betting opportunities.

Classifies each race by:
1. Is the favorite vulnerable? (high decay + contested pace + negative model edge)
2. Is there depth? (multiple competitive horses)
3. Where does usage concentrate? (crowd bias from odds structure)
4. Where's the separation? (positive-edge horses at value odds)
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

# A favorite is "vulnerable" if:
# - Their decay rate is above field median (they fade relative to the field)
# - The pace scenario is CONTESTED or PRESSURED (multiple speed types)
# - The model gives them negative edge (we think they're overbet)
VULNERABLE_DECAY_THRESHOLD = 0.5  # must be 0.5+ above field median decay
VULNERABLE_EDGE_THRESHOLD = -0.02  # model dislikes by at least 2%


def analyze_race_situation(race_data: dict) -> dict | None:
    """Analyze a single race for ITP-style betting situations.

    Args:
        race_data: {
            race_id, furlongs, field_size,
            starters: [{horse_key, odds, choice, finish_position, v0, decay_rate, model_prob, odds_prob, edge}]
            pace_scenario: str,
            exacta_payoff, trifecta_payoff, super_payoff: float or None
        }
    """
    starters = race_data["starters"]
    if len(starters) < 5:
        return None

    # Identify the favorite (choice=1)
    favorite = next((s for s in starters if s["choice"] == 1), None)
    if favorite is None or favorite.get("v0") is None:
        return None

    # Field metrics
    decay_rates = [s["decay_rate"] for s in starters if s.get("decay_rate") is not None]
    if not decay_rates:
        return None

    field_median_decay = np.median(decay_rates)
    v0s = [s["v0"] for s in starters if s.get("v0") is not None]

    # Competitive depth: horses within 2 ft/s of the best adjusted speed at race midpoint
    race_mid_ft = race_data["furlongs"] * 660 / 2
    speeds_at_mid = [s["v0"] - s["decay_rate"] * (race_mid_ft / 1000) for s in starters
                     if s.get("v0") is not None and s.get("decay_rate") is not None]
    if not speeds_at_mid:
        return None

    best_speed = max(speeds_at_mid)
    field_depth = sum(1 for sp in speeds_at_mid if sp >= best_speed - 2.0)

    # Vulnerable favorite check
    fav_decay_above_median = favorite["decay_rate"] - field_median_decay
    fav_edge = favorite.get("edge", 0) or 0
    pace = race_data.get("pace_scenario", "UNKNOWN")

    has_vulnerable_fav = (
        fav_decay_above_median >= VULNERABLE_DECAY_THRESHOLD
        and pace in ("CONTESTED", "PRESSURED")
        and fav_edge < VULNERABLE_EDGE_THRESHOLD
    )

    # Usage concentration: top 2 choices' combined odds-implied probability
    sorted_by_choice = sorted(starters, key=lambda s: s.get("choice", 99))
    top2_odds_prob = sum(s.get("odds_prob", 0) for s in sorted_by_choice[:2])

    # Separation: positive-edge horses at 5/1+
    positive_edge_at_price = [s for s in starters
                              if s.get("edge", 0) > 0.01 and s.get("odds", 0) >= 5.0]
    n_positive_edge = len([s for s in starters if s.get("edge", 0) > 0.01])

    # Did the favorite miss the board?
    fav_finish = favorite.get("finish_position")
    fav_missed_board = fav_finish is not None and fav_finish > 3

    # Situation classification (ITP framework)
    if has_vulnerable_fav and field_depth >= 5:
        situation_type = "ATTACK_VERTICAL"  # Bad fav + depth = basket of bets in tris/supers
    elif has_vulnerable_fav and field_depth >= 3:
        situation_type = "ATTACK_NARROW"  # Bad fav but limited depth = key a contender
    elif n_positive_edge >= 3 and not has_vulnerable_fav:
        situation_type = "SPREAD"  # Multiple value plays, no clear bad fav
    else:
        situation_type = "PASS"  # No clear situation

    return {
        "race_id": race_data["race_id"],
        "has_vulnerable_fav": has_vulnerable_fav,
        "fav_horse_key": favorite.get("horse_key"),
        "fav_v0": favorite.get("v0"),
        "fav_decay_rate": favorite.get("decay_rate"),
        "fav_edge": round(fav_edge, 4) if fav_edge else None,
        "fav_finish_position": fav_finish,
        "fav_missed_board": fav_missed_board,
        "pace_scenario": pace,
        "field_depth": field_depth,
        "field_size": len(starters),
        "usage_concentration": round(top2_odds_prob, 3),
        "separation_available": len(positive_edge_at_price) > 0,
        "n_positive_edge": n_positive_edge,
        "exacta_payoff": race_data.get("exacta_payoff"),
        "trifecta_payoff": race_data.get("trifecta_payoff"),
        "super_payoff": race_data.get("super_payoff"),
        "situation_type": situation_type,
    }
