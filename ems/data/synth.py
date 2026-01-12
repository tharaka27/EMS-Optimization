from __future__ import annotations
from typing import List, Tuple
from ..simulation.types import Call

def mock_calls() -> List[Call]:
    t0 = 1_700_000_000
    calls: List[Call] = []
    grid_points = [(10,10),(11,10),(10,11),(12,12),(8,9)]
    hospitals = [(5,5)] * 5
    cats = ["cardiac","catA","catA","catA","catC"]
    for i in range(5):
        calls.append(Call(
            call_id=i,
            t_call=t0 + i*180,
            dispatch_delay=30.0,
            scene_time=600.0,
            hospital_time=900.0,
            handover_time=300.0,
            need_ambulances=1,
            need_rrc=0,
            category=cats[i],
            loc_scene=grid_points[i],
            loc_hospital=hospitals[i],
        ))
    return calls