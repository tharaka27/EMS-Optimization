from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class StationPlan:
    stations: Dict[int, Tuple[int,int]]  # sid -> (r,c)

@dataclass
class AllocationPlan:
    counts: Dict[int, Dict[str,int]]     # sid -> {"A": nA, "R": nR}

@dataclass
class FleetPlan:
    total: int
    ratio: tuple[int,int] | None