from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random, copy

from ..core.types import Station
from ..core.sim import Simulation
from ..core.builder import build_vehicles_from_allocation
from ..travel.base import ITravelModel
from ..policies.dispatch_base import IDispatchPolicy
from ..plans.schema import StationPlan, AllocationPlan

@dataclass
class GAConfig:
    pop_size: int = 25
    rx: float = 0.85
    rm: float = 0.04
    elite_k: int = 1
    generations: int = 180

class GAOptimizer:
    """
    Chromosome = [ movable station (r) | (c) ... | fleet-mix (optional) | per-vehicle station index genes ]
    For simplicity here: we optimize station positions, fleet mix, and explicit per-vehicle placements.
    """
    def __init__(
        self,
        calls,
        stations_fixed: Dict[int, Station],
        movable_station_ids: List[int],
        N_grid: Tuple[int,int],
        total_vehicles: int,
        fleet_ratio_fixed: Tuple[int,int] | None,
        t_window: Tuple[float,float],
        travel: ITravelModel,
        dispatch: IDispatchPolicy,
        ga_cfg: GAConfig = GAConfig(),
        seed: int = 42,
    ):
        random.seed(seed)
        self.calls = calls
        self.stations_fixed = stations_fixed
        self.movable_station_ids = movable_station_ids[:]  # length n
        self.n_rows, self.n_cols = N_grid
        self.m = total_vehicles
        self.fleet_ratio_fixed = fleet_ratio_fixed
        self.t_window = t_window
        self.travel = travel
        self.dispatch = dispatch
        self.cfg = ga_cfg

        self.n = len(self.movable_station_ids)
        self.chrom_len = 2*self.n + (0 if fleet_ratio_fixed else 1) + self.m

    # -------- decode / encode --------
    def decode_plan(self, chrom: List[float]) -> Tuple[StationPlan, AllocationPlan]:
        stations = {sid: st.grid_rc for sid, st in self.stations_fixed.items()}
        idx = 0
        # movable station positions
        for sid in self.movable_station_ids:
            r = int(round(chrom[idx] * (self.n_rows - 1))); idx += 1
            c = int(round(chrom[idx] * (self.n_cols - 1))); idx += 1
            stations[sid] = (r, c)

        # fleet mix
        if self.fleet_ratio_fixed is None:
            g = chrom[idx]; idx += 1
            nA = int(round(g * self.m)); nA = max(0, min(self.m, nA))
            nR = self.m - nA
        else:
            nA, nR = self.fleet_ratio_fixed

        # per-vehicle station indices
        station_ids = list(stations.keys())
        N = len(station_ids)
        alloc = {sid: {"A": 0, "R": 0} for sid in station_ids}
        for _ in range(nA):
            s_idx = int(round(chrom[idx] * (N - 1))); idx += 1
            alloc[station_ids[s_idx]]["A"] += 1
        for _ in range(nR):
            s_idx = int(round(chrom[idx] * (N - 1))); idx += 1
            alloc[station_ids[s_idx]]["R"] += 1

        return StationPlan(stations=stations), AllocationPlan(counts=alloc)

    # -------- fitness --------
    def _fitness(self, chrom: List[float]) -> float:
        stn_plan, alloc_plan = self.decode_plan(chrom)
        stations_decoded = {sid: Station(sid, rc) for sid, rc in stn_plan.stations.items()}
        vehicles = build_vehicles_from_allocation(alloc_plan.counts)
        # initialize vehicle locs at home stations
        for v in vehicles.values():
            v.loc = stations_decoded[v.home_station_id].grid_rc
        sim = Simulation(
            calls=self.calls,
            stations=stations_decoded,
            vehicles=vehicles,
            travel=self.travel,
            dispatch=self.dispatch,
        )
        t0, t1 = self.t_window
        kpi = sim.run(t_start=t0, t_end=t1, warmup_buffer=90*60.0)
        return kpi["eta_s"]

    # -------- GA loop --------
    def _init_population(self) -> List[List[float]]:
        P = []
        half = self.cfg.pop_size // 2
        for _ in range(half):
            P.append([random.random() for _ in range(self.chrom_len)])
        P += [[1.0 - g for g in ch] for ch in P[:half]]
        while len(P) < self.cfg.pop_size:
            P.append([random.random() for _ in range(self.chrom_len)])
        return P[:self.cfg.pop_size]

    def _tournament(self, P: List[List[float]], F: List[float], k: int = 3) -> List[float]:
        best = None; bestf = -1e18
        for _ in range(k):
            i = random.randrange(len(P))
            if F[i] > bestf:
                bestf = F[i]; best = P[i]
        return best[:]

    def _crossover(self, p1: List[float], p2: List[float]) -> tuple[List[float], List[float]]:
        c1, c2 = p1[:], p2[:]
        for i in range(len(p1)):          # uniform swap
            if random.random() < 0.5: c1[i], c2[i] = c2[i], c1[i]
        idxs = random.sample(range(len(p1)), k=max(1, len(p1)//2))
        beta = random.uniform(0.0, 1.1)   # blend
        for i in idxs:
            g1, g2 = c1[i], c2[i]
            c1[i] = g1 + beta*(g2-g1)
            c2[i] = g2 + beta*(g1-g2)
        c1 = [min(1.0, max(0.0, x)) for x in c1]
        c2 = [min(1.0, max(0.0, x)) for x in c2]
        return c1, c2

    def _mutate(self, P: List[List[float]]):
        n_genes = len(P) * self.chrom_len
        n_mut = int(round(self.cfg.rm * n_genes))
        for _ in range(n_mut):
            i = random.randrange(len(P))
            j = random.randrange(self.chrom_len)
            P[i][j] = random.random()

    def run(self) -> tuple[List[float], float]:
        P = self._init_population()
        F = [self._fitness(ch) for ch in P]
        best_idx = max(range(len(P)), key=lambda i: F[i])
        best_ch, best_fit = P[best_idx][:], F[best_idx]

        for _ in range(self.cfg.generations):
            ranked = sorted(zip(P, F), key=lambda x: x[1], reverse=True)
            elites = [copy.deepcopy(ch) for ch, _ in ranked[:self.cfg.elite_k]]
            Pnext: List[List[float]] = elites[:]

            while len(Pnext) < self.cfg.pop_size:
                if random.random() < self.cfg.rx:
                    p1 = self._tournament(P, F, k=3)
                    p2 = self._tournament(P, F, k=3)
                    c1, c2 = self._crossover(p1, p2)
                    Pnext.append(c1)
                    if len(Pnext) < self.cfg.pop_size:
                        Pnext.append(c2)
                else:
                    Pnext.append(self._tournament(P, F, k=3))

            self._mutate(Pnext[self.cfg.elite_k:])
            P, F = Pnext, [self._fitness(ch) for ch in Pnext]
            i = max(range(len(P)), key=lambda k: F[k])
            if F[i] > best_fit:
                best_fit, best_ch = F[i], P[i][:]

        return best_ch, best_fit