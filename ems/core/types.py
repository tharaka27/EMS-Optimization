from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Any

@dataclass
class Call:
    call_id: int
    t_call: float
    dispatch_delay: float
    scene_time: float
    hospital_time: float
    handover_time: float
    need_ambulances: int
    need_rrc: int
    category: str                      # "cardiac", "catA", "catC"
    loc_scene: Tuple[int, int]
    loc_hospital: Tuple[int, int]

@dataclass
class Station:
    station_id: int
    grid_rc: Tuple[int, int]

@dataclass
class Vehicle:
    vehicle_id: int
    kind: str                           # "A" or "R"
    home_station_id: int
    busy: bool = False
    route: List[Tuple[int, int]] = field(default_factory=list)
    loc: Tuple[int, int] | None = None
    assigned_call: Optional[int] = None

class Event:
    __slots__ = ("time", "etype", "payload")
    def __init__(self, time: float, etype: str, payload: Any):
        self.time = time
        self.etype = etype
        self.payload = payload
    def __lt__(self, other):  # heapq ordering
        return self.time < other.time