from __future__ import annotations
from typing import Protocol, Tuple
from ..plans.schema import StationPlan, AllocationPlan

class IOptimizer(Protocol):
    def run(self) -> Tuple[list[float], float]: ...
    def decode_plan(self, chrom: list[float]) -> Tuple[StationPlan, AllocationPlan]: ...