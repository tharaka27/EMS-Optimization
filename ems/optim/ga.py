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
    """
    Configuration parameters for the genetic algorithm.

    Attributes
    ----------
    pop_size : int
        Population size.
    rx : float
        Probability of crossover.
    rm : float
        Mutation rate (fraction of genes mutated per generation).
    elite_k : int
        Number of elite individuals preserved each generation.
    generations : int
        Number of GA iterations.
    """
    pop_size: int = 25
    rx: float = 0.85
    rm: float = 0.04
    elite_k: int = 1
    generations: int = 180


class GAOptimizer:
    """
    Genetic Algorithm optimizer for EMS system design.

    Chromosome structure:
    - 2 genes per movable station (row, column)
    - Optional fleet-mix gene (fraction of type-A vehicles)
    - One gene per vehicle indicating its assigned station

    All genes are normalized floats in [0, 1].
    """

    def __init__(
        self,
        calls,
        stations_fixed: Dict[int, Station],
        movable_station_ids: List[int],
        N_grid: Tuple[int, int],
        total_vehicles: int,
        fleet_ratio_fixed: Tuple[int, int] | None,
        t_window: Tuple[float, float],
        travel: ITravelModel,
        dispatch: IDispatchPolicy,
        ga_cfg: GAConfig = GAConfig(),
        seed: int = 42,
    ):
        """
        Initialize the GA optimizer and problem definition.

        Parameters
        ----------
        calls
            Emergency call dataset.
        stations_fixed
            Mapping of fixed station IDs to Station objects.
        movable_station_ids
            Station IDs whose locations may be optimized.
        N_grid
            Grid dimensions (rows, columns).
        total_vehicles
            Total number of vehicles in the system.
        fleet_ratio_fixed
            Optional fixed fleet composition (nA, nR).
        t_window
            Simulation time window (start, end).
        travel
            Travel model.
        dispatch
            Dispatch policy.
        ga_cfg
            Genetic algorithm configuration.
        seed
            Random seed for reproducibility.
        """
        random.seed(seed)

        self.calls = calls
        self.stations_fixed = stations_fixed
        self.movable_station_ids = movable_station_ids[:]  # number of movable stations
        self.n_rows, self.n_cols = N_grid
        self.m = total_vehicles
        self.fleet_ratio_fixed = fleet_ratio_fixed
        self.t_window = t_window
        self.travel = travel
        self.dispatch = dispatch
        self.cfg = ga_cfg

        # Number of movable stations
        self.n = len(self.movable_station_ids)

        # Chromosome length calculation
        self.chrom_len = (
            2 * self.n +
            (0 if fleet_ratio_fixed else 1) +
            self.m
        )

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode_plan(self, chrom: List[float]) -> Tuple[StationPlan, AllocationPlan]:
        """
        Decode a chromosome into a station placement and vehicle allocation plan.

        Parameters
        ----------
        chrom
            Chromosome vector.

        Returns
        -------
        StationPlan
            Concrete station locations.
        AllocationPlan
            Vehicle counts per station and type.
        """
        # Start with fixed stations
        stations = {sid: st.grid_rc for sid, st in self.stations_fixed.items()}
        idx = 0

        # Decode movable station positions
        for sid in self.movable_station_ids:
            r = int(round(chrom[idx] * (self.n_rows - 1))); idx += 1
            c = int(round(chrom[idx] * (self.n_cols - 1))); idx += 1
            stations[sid] = (r, c)

        # Decode fleet mix
        if self.fleet_ratio_fixed is None:
            g = chrom[idx]; idx += 1
            nA = int(round(g * self.m))
            nA = max(0, min(self.m, nA))
            nR = self.m - nA
        else:
            nA, nR = self.fleet_ratio_fixed

        # Decode per-vehicle station assignment
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

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def _fitness(self, chrom: List[float]) -> float:
        """
        Evaluate the fitness of a chromosome via simulation.

        Parameters
        ----------
        chrom
            Chromosome to evaluate.

        Returns
        -------
        float
            Fitness value (eta_s).
        """
        stn_plan, alloc_plan = self.decode_plan(chrom)

        # Instantiate station objects
        stations_decoded = {
            sid: Station(sid, rc)
            for sid, rc in stn_plan.stations.items()
        }

        # Build vehicles from allocation plan
        vehicles = build_vehicles_from_allocation(alloc_plan.counts)

        # Initialize vehicle locations at home stations
        for v in vehicles.values():
            v.loc = stations_decoded[v.home_station_id].grid_rc

        # Run simulation
        sim = Simulation(
            calls=self.calls,
            stations=stations_decoded,
            vehicles=vehicles,
            travel=self.travel,
            dispatch=self.dispatch,
        )

        t0, t1 = self.t_window
        kpi = sim.run(t_start=t0, t_end=t1, warmup_buffer=90 * 60.0)

        return kpi["eta_s"]

    # ------------------------------------------------------------------
    # Genetic algorithm operators
    # ------------------------------------------------------------------

    def _init_population(self) -> List[List[float]]:
        """
        Initialize the GA population using random and antithetic sampling.

        Returns
        -------
        List[List[float]]
            Initial population.
        """
        P = []
        half = self.cfg.pop_size // 2

        for _ in range(half):
            P.append([random.random() for _ in range(self.chrom_len)])

        # Antithetic sampling
        P += [[1.0 - g for g in ch] for ch in P[:half]]

        while len(P) < self.cfg.pop_size:
            P.append([random.random() for _ in range(self.chrom_len)])

        return P[:self.cfg.pop_size]

    def _tournament(self, P: List[List[float]], F: List[float], k: int = 3) -> List[float]:
        """
        Tournament selection operator.

        Parameters
        ----------
        P
            Population.
        F
            Fitness values.
        k
            Tournament size.

        Returns
        -------
        List[float]
            Selected chromosome.
        """
        best = None
        bestf = -1e18

        for _ in range(k):
            i = random.randrange(len(P))
            if F[i] > bestf:
                bestf = F[i]
                best = P[i]

        return best[:]

    def _crossover(
        self,
        p1: List[float],
        p2: List[float]
    ) -> tuple[List[float], List[float]]:
        """
        Perform uniform + blend crossover.

        Parameters
        ----------
        p1, p2
            Parent chromosomes.

        Returns
        -------
        tuple[List[float], List[float]]
            Two offspring chromosomes.
        """
        c1, c2 = p1[:], p2[:]

        # Uniform swap
        for i in range(len(p1)):
            if random.random() < 0.5:
                c1[i], c2[i] = c2[i], c1[i]

        # Blend crossover
        idxs = random.sample(range(len(p1)), k=max(1, len(p1) // 2))
        beta = random.uniform(0.0, 1.1)

        for i in idxs:
            g1, g2 = c1[i], c2[i]
            c1[i] = g1 + beta * (g2 - g1)
            c2[i] = g2 + beta * (g1 - g2)

        # Clamp genes to [0, 1]
        c1 = [min(1.0, max(0.0, x)) for x in c1]
        c2 = [min(1.0, max(0.0, x)) for x in c2]

        return c1, c2

    def _mutate(self, P: List[List[float]]):
        """
        Apply random reset mutation to the population.

        Parameters
        ----------
        P
            Population (modified in place).
        """
        n_genes = len(P) * self.chrom_len
        n_mut = int(round(self.cfg.rm * n_genes))

        for _ in range(n_mut):
            i = random.randrange(len(P))
            j = random.randrange(self.chrom_len)
            P[i][j] = random.random()

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def run(self) -> tuple[List[float], float]:
        """
        Run the genetic algorithm optimization.

        Returns
        -------
        tuple[List[float], float]
            Best chromosome found and its fitness value.
        """
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
