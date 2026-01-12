from __future__ import annotations
from typing import Dict, List
from .dispatch_base import IDispatchPolicy
from ..simulation.types import Call, Vehicle
from ..travel.base import ITravelModel

class NearestETA(IDispatchPolicy):
    """Greedy nearest-ETA by required vehicle types. Allows partial assignment."""
    def assign(self, now: float, call: Call, vehicles: Dict[int, Vehicle], travel: ITravelModel) -> List[int]:
        needA = call.need_ambulances
        needR = call.need_rrc
        idle = [v for v in vehicles.values() if not v.busy]
        idle.sort(key=lambda v: travel.travel_time(v.loc, call.loc_scene, now, v.kind))
        assigned: List[int] = []
        for v in idle:
            if v.kind == "A" and needA > 0:
                assigned.append(v.vehicle_id); needA -= 1
            elif v.kind == "R" and needR > 0:
                assigned.append(v.vehicle_id); needR -= 1
            if needA == 0 and needR == 0:
                break
        return assigned