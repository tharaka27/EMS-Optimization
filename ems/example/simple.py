from __future__ import annotations
from typing import Dict
from ..core.types import Station
from ..core.builder import build_vehicles_from_allocation
from ..travel.grid8 import Grid8Travel
from ..policies.dispatch_nearest import NearestETA
from ..optim.ga import GAOptimizer, GAConfig
from ..data.synth import mock_calls
from ..core.sim import Simulation

import sys

def main():
    # stations (4 total; 2 movable)
    stations: Dict[int, Station] = {
        0: Station(0, (10, 10)),
        1: Station(1, (9, 12)),
        2: Station(2, (12, 9)),
        3: Station(3, (7, 7)),
    }

    vehicles_allocation = {
        0: {"A": 1 , "R": 0},
        1: {"A": 3 , "R": 0},
        2: {"A": 1 , "R": 0},
        3: {"A": 1 , "R": 0}
    }

    movable = [1, 2]

    travel = Grid8Travel(n_rows=26, n_cols=22, cell_km=2.0, default_speed_kmph=45.0)
    dispatch = NearestETA()

    calls = mock_calls()
    t_start = calls[0].t_call
    t_end = t_start + 3600  # 1h
    total_vehicles = 6

    stations_decoded = stations #{sid: Station(sid, rc) for sid, rc in stations.items()}
    vehicles = build_vehicles_from_allocation(vehicles_allocation)

    for v in vehicles.values():
        v.loc = stations_decoded[v.home_station_id].grid_rc
    sim = Simulation(calls, stations_decoded, vehicles, travel, dispatch)
    kpi = sim.run(t_start=t_start, t_end=t_end, warmup_buffer=90*60.0)
    print("\nSimulation KPIs before optimization:", kpi)

    ga = GAOptimizer(
        calls=calls,
        stations_fixed=stations,
        movable_station_ids=movable,
        N_grid=(26,22),
        total_vehicles=total_vehicles,
        fleet_ratio_fixed=None,           # optimize A vs R
        t_window=(t_start, t_end),
        travel=travel,
        dispatch=dispatch,
        ga_cfg=GAConfig(pop_size=16, generations=40, rx=0.85, rm=0.04, elite_k=1),
        seed=123
    )

    best_ch, best_fit = ga.run()
    
    stn_plan, alloc_plan = ga.decode_plan(best_ch)
    print("\nDecoded movable station positions:")
    for sid in sorted(stn_plan.stations):
        print(f"  Station {sid}: rc={stn_plan.stations[sid]}")

    print("\nDecoded allocation (per station):")
    for sid in sorted(alloc_plan.counts):
        counts = alloc_plan.counts[sid]
        print(f"  Station {sid}: A={counts['A']}, R={counts['R']}")

    # Optional evaluation
    stations_decoded = {sid: Station(sid, rc) for sid, rc in stn_plan.stations.items()}
    vehicles = build_vehicles_from_allocation(alloc_plan.counts)
    
    for v in vehicles.values():
        v.loc = stations_decoded[v.home_station_id].grid_rc
    sim = Simulation(calls, stations_decoded, vehicles, travel, dispatch)
    kpi = sim.run(t_start=t_start, t_end=t_end, warmup_buffer=90*60.0)
    print("\nSimulation KPIs:", kpi)

if __name__ == "__main__":
    main()