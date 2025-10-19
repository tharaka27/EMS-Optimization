import math
from typing import Dict, Iterable
from .types import Call

def survival_cardiac_arrest(response_time_sec: float) -> float:
    # sc = 1 / (1 + exp(-0.26 + 0.139*Tr)), Tr in minutes
    Tr = response_time_sec / 60.0
    return 1.0 / (1.0 + math.exp(-0.26 + 0.139 * Tr))

def survival_catA(response_time_sec: float) -> float:
    # sa = 1 if Tr <= 8 min else 0
    return 1.0 if response_time_sec <= 8 * 60.0 else 0.0

def compute_eta_s(calls: Iterable[Call], response_times: Dict[int, float]) -> Dict[str, float]:
    gamma = 0      # # cardiac
    delta = 0      # # catA
    sum_sc = 0.0
    sum_sa = 0.0
    for c in calls:
        if c.call_id not in response_times:
            continue
        rt = response_times[c.call_id]
        if c.category == "cardiac":
            gamma += 1
            sum_sc += survival_cardiac_arrest(rt)
        elif c.category == "catA":
            delta += 1
            sum_sa += survival_catA(rt)
    num = 2.0 * sum_sc + sum_sa
    den = 2.0 * gamma + delta if (2.0 * gamma + delta) > 0 else 1.0
    eta_s = num / den
    return {"eta_s": eta_s, "cardiac_calls": float(gamma), "catA_calls": float(delta)}