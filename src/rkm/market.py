"""Benter market combination — model probability vs odds probability."""

import logging

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger(__name__)

SOFTMAX_TEMPERATURE = 6500.0  # ms — calibrated: predicted probabilities match actual win frequencies


def compute_model_probabilities(starters: list[dict], race_distance_ft: float) -> list[float]:
    """Compute model win probabilities from velocity curves via predicted finishing times.

    Each starter dict needs: adj_v0, decay_rate
    Returns list of probabilities (same order as input, sums to 1).
    """
    predicted_times = []
    for s in starters:
        avg_v = s["adj_v0"] - s["decay_rate"] * (race_distance_ft / 2000.0)
        if avg_v <= 0:
            avg_v = 30.0  # floor to avoid division by zero
        pred_time_ms = race_distance_ft / avg_v * 1000.0
        predicted_times.append(pred_time_ms)

    # Softmax on negative times (faster = lower time = higher probability)
    times_arr = np.array(predicted_times)
    margins = times_arr.min() - times_arr  # negative for slower horses
    exp_margins = np.exp(margins / SOFTMAX_TEMPERATURE)
    probs = exp_margins / exp_margins.sum()
    return probs.tolist()


def compute_odds_probabilities(odds_list: list[float]) -> list[float]:
    """Convert odds to normalized implied probabilities."""
    raw_probs = [1.0 / (o + 1.0) if o > 0 else 0.01 for o in odds_list]
    total = sum(raw_probs)
    return [p / total for p in raw_probs]


def fit_benter_logit(model_probs_by_race: list[list[float]],
                     odds_probs_by_race: list[list[float]],
                     winners_by_race: list[int]) -> tuple[float, float]:
    """Fit Benter's conditional logit: P(i) = exp(α*log(f_i) + β*log(x_i)) / Σ

    Args:
        model_probs_by_race: list of races, each a list of model probs per starter
        odds_probs_by_race: list of races, each a list of odds probs per starter
        winners_by_race: index of winner in each race

    Returns: (alpha, beta) — weights for model and odds
    """
    def neg_log_likelihood(params):
        alpha, beta = params
        total_ll = 0.0
        for model_probs, odds_probs, winner_idx in zip(
            model_probs_by_race, odds_probs_by_race, winners_by_race
        ):
            n = len(model_probs)
            if winner_idx < 0 or winner_idx >= n:
                continue

            log_f = np.log(np.clip(model_probs, 1e-10, 1.0))
            log_x = np.log(np.clip(odds_probs, 1e-10, 1.0))

            utilities = alpha * log_f + beta * log_x
            # Log-sum-exp for numerical stability
            max_u = utilities.max()
            log_sum_exp = max_u + np.log(np.sum(np.exp(utilities - max_u)))
            total_ll += utilities[winner_idx] - log_sum_exp

        return -total_ll  # minimize negative log-likelihood

    result = minimize(neg_log_likelihood, x0=[1.0, 1.0], method="Nelder-Mead",
                      options={"maxiter": 1000, "xatol": 0.01, "fatol": 0.01})

    alpha, beta = result.x
    log.info(f"Benter logit fit: α={alpha:.4f} (model), β={beta:.4f} (odds)")
    log.info(f"  α/β ratio: {alpha/beta:.3f} (>0 means model adds value)")
    return float(alpha), float(beta)


def compute_combined_probabilities(model_probs: list[float], odds_probs: list[float],
                                   alpha: float, beta: float) -> list[float]:
    """Compute combined probabilities using fitted Benter logit."""
    log_f = np.log(np.clip(model_probs, 1e-10, 1.0))
    log_x = np.log(np.clip(odds_probs, 1e-10, 1.0))
    utilities = alpha * log_f + beta * log_x
    exp_u = np.exp(utilities - utilities.max())
    probs = exp_u / exp_u.sum()
    return probs.tolist()
