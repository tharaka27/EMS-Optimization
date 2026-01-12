import math
from typing import Dict, Iterable
from .types import Call


def survival_cardiac_arrest(response_time_sec: float) -> float:
    """
    Compute survival probability for cardiac arrest cases.

    The survival model follows a logistic function of response time:
        sc = 1 / (1 + exp(-0.26 + 0.139 * Tr))

    where Tr is the response time in minutes.

    Parameters
    ----------
    response_time_sec : float
        Emergency response time in seconds.

    Returns
    -------
    float
        Probability of survival for a cardiac arrest patient.
    """
    Tr = response_time_sec / 60.0  # Convert response time to minutes
    return 1.0 / (1.0 + math.exp(-0.26 + 0.139 * Tr))


def survival_catA(response_time_sec: float) -> float:
    """
    Compute survival probability for Category A calls.

    This is a threshold-based survival model:
    - Survival probability is 1.0 if response time <= 8 minutes
    - Survival probability is 0.0 otherwise

    Parameters
    ----------
    response_time_sec : float
        Emergency response time in seconds.

    Returns
    -------
    float
        Survival probability for a Category A patient.
    """
    return 1.0 if response_time_sec <= 8 * 60.0 else 0.0


def compute_eta_s(
    calls: Iterable[Call],
    response_times: Dict[int, float]
) -> Dict[str, float]:
    """
    Compute the weighted system survival KPI (eta_s).

    The metric aggregates survival outcomes for different call categories:
    - Cardiac arrest calls receive double weight
    - Category A calls receive single weight

    eta_s is defined as:
        eta_s = (2 * sum(sc) + sum(sa)) / (2 * gamma + delta)

    where:
    - gamma = number of cardiac calls
    - delta = number of Category A calls
    - sc = cardiac survival probabilities
    - sa = Category A survival probabilities

    Parameters
    ----------
    calls : Iterable[Call]
        Collection of emergency calls.
    response_times : Dict[int, float]
        Mapping from call_id to response time (seconds).

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - "eta_s": overall weighted survival score
        - "cardiac_calls": number of cardiac arrest calls
        - "catA_calls": number of Category A calls
    """
    gamma = 0      # Number of cardiac arrest calls
    delta = 0      # Number of Category A calls
    sum_sc = 0.0   # Sum of cardiac survival probabilities
    sum_sa = 0.0   # Sum of Category A survival probabilities

    for c in calls:
        # Skip calls with no recorded response time
        if c.call_id not in response_times:
            continue

        rt = response_times[c.call_id]

        # Cardiac arrest calls
        if c.category == "cardiac":
            gamma += 1
            sum_sc += survival_cardiac_arrest(rt)

        # Category A calls
        elif c.category == "catA":
            delta += 1
            sum_sa += survival_catA(rt)

    # Weighted numerator and denominator
    num = 2.0 * sum_sc + sum_sa
    den = 2.0 * gamma + delta if (2.0 * gamma + delta) > 0 else 1.0

    eta_s = num / den

    return {
        "eta_s": eta_s,
        "cardiac_calls": float(gamma),
        "catA_calls": float(delta),
    }
