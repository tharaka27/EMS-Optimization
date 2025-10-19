from __future__ import annotations
from typing import Tuple, List

from .base import ITravelModel

class ManhattanTravel(ITravelModel):
    def __init__(self, cell_km: float = 2.0, speed_kmph: float = 45.0):
        self.cell_km = cell_km
        self.speed_kmph = speed_kmph

    def distance(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
        (r1, c1), (r2, c2) = a, b
        return (abs(r1 - r2) + abs(c1 - c2)) * self.cell_km

    def travel_time(self, a: Tuple[int,int], b: Tuple[int,int], t0_sec: float, kind: str) -> float:
        return (self.distance(a, b) / max(1e-6, self.speed_kmph)) * 3600.0

    def route(self, a: Tuple[int,int], b: Tuple[int,int]) -> List[Tuple[int,int]]:
        # simple 4-neighbor route
        (r, c), (rt, ct) = a, b
        path = [(r, c)]
        while r != rt: r += 1 if rt > r else -1; path.append((r, c))
        while c != ct: c += 1 if ct > c else -1; path.append((r, c))
        return path