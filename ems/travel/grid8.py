from __future__ import annotations
from typing import Tuple, List
import math

from .base import ITravelModel

class Grid8Travel(ITravelModel):
    def __init__(self, n_rows: int, n_cols: int, cell_km: float = 2.0, default_speed_kmph: float = 45.0):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.cell_km = cell_km
        self.default_speed_kmph = default_speed_kmph
        self.minute_of_week_speeds = None  # Optional[List[float]] length 10080

    def _speed_kmph(self, t0_sec: float, kind: str) -> float:
        if self.minute_of_week_speeds:
            minute = int((t0_sec // 60) % (7 * 24 * 60))
            return max(5.0, self.minute_of_week_speeds[minute])
        return self.default_speed_kmph

    def distance(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
        (r1, c1), (r2, c2) = a, b
        dr, dc = abs(r1 - r2), abs(c1 - c2)
        steps_diag = min(dr, dc)
        steps_straight = abs(dr - dc)
        return steps_diag * (self.cell_km * math.sqrt(2)) + steps_straight * self.cell_km

    def travel_time(self, a: Tuple[int,int], b: Tuple[int,int], t0_sec: float, kind: str) -> float:
        d_km = self.distance(a, b)
        v = max(1e-6, self._speed_kmph(t0_sec, kind))
        return (d_km / v) * 3600.0

    def route(self, a: Tuple[int,int], b: Tuple[int,int]) -> List[Tuple[int,int]]:
        r, c = a; rt, ct = b
        path = [(r, c)]
        while (r, c) != (rt, ct):
            dr = 0 if r == rt else (1 if rt > r else -1)
            dc = 0 if c == ct else (1 if ct > c else -1)
            r += dr; c += dc
            path.append((r, c))
        return path