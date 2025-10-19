from __future__ import annotations
from typing import Dict
from .types import Vehicle

def build_vehicles_from_allocation(allocation: Dict[int, Dict[str, int]]) -> Dict[int, Vehicle]:
    """
    allocation: {station_id: {"A": nA, "R": nR}}
    returns: {vehicle_id: Vehicle}
    """
    vid = 0
    vehicles: Dict[int, Vehicle] = {}
    for sid, counts in allocation.items():
        for _ in range(counts.get("A", 0)):
            vehicles[vid] = Vehicle(vehicle_id=vid, kind="A", home_station_id=sid)
            vid += 1
        for _ in range(counts.get("R", 0)):
            vehicles[vid] = Vehicle(vehicle_id=vid, kind="R", home_station_id=sid)
            vid += 1
    return vehicles