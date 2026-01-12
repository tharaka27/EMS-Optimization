from __future__ import annotations
from typing import Protocol, List, Dict
from ..simulation.types import Call, Vehicle
from ..travel.base import ITravelModel

class IDispatchPolicy(Protocol):
    def assign(self, now: float, call: Call, vehicles: Dict[int, Vehicle], travel: ITravelModel) -> List[int]:
        ...