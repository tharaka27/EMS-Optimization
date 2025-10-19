from __future__ import annotations
from typing import Dict, List, Tuple
import heapq

from .types import Call, Station, Vehicle, Event
from .kpis import compute_eta_s

# Interfaces are imported via Protocols
from ..policies.dispatch_base import IDispatchPolicy
from ..travel.base import ITravelModel

class Simulation:
    """
    Event-driven EMS simulation core.
    Depends only on two interfaces: ITravelModel and IDispatchPolicy.
    """
    def __init__(
        self,
        calls: List[Call],
        stations: Dict[int, Station],
        vehicles: Dict[int, Vehicle],
        travel: ITravelModel,
        dispatch: IDispatchPolicy,
        loc_update_period: float = 120.0,  # seconds
    ):
        self.calls = sorted(calls, key=lambda c: c.t_call)
        self.stations = stations
        self.vehicles = vehicles
        self.travel = travel
        self.dispatch = dispatch
        self.loc_update_period = loc_update_period

        self.event_q: List[Event] = []
        self.wait_q: List[int] = []
        self.now: float = 0.0

        self.response_times: Dict[int, float] = {}
        self.call_assignments: Dict[int, List[int]] = {}

        for v in self.vehicles.values():
            if v.loc is None:
                v.loc = self.stations[v.home_station_id].grid_rc

    def run(self, t_start: float, t_end: float, warmup_buffer: float = 90 * 60.0) -> Dict[str, float]:
        warm_lo = t_start - warmup_buffer
        for c in self.calls:
            if warm_lo <= c.t_call < t_start:
                heapq.heappush(self.event_q, Event(c.t_call, "CALL_ARRIVE", c))
        self._loop(until=t_start, record_stats=False)

        for c in self.calls:
            if t_start <= c.t_call < t_end:
                heapq.heappush(self.event_q, Event(c.t_call, "CALL_ARRIVE", c))
        self._loop(until=t_end, record_stats=True)

        return compute_eta_s(self.calls, self.response_times)

    # ---------------- internal loop ----------------

    def _loop(self, until: float, record_stats: bool):
        while self.event_q and self.event_q[0].time < until:
            ev = heapq.heappop(self.event_q)
            self.now = ev.time
            et = ev.etype

            if et == "CALL_ARRIVE":
                self._on_call_arrive(ev.payload, record_stats)
            elif et == "SCENE_DEPART":
                self._on_scene_depart(ev.payload)
            elif et == "JOB_COMPLETE":
                self._on_job_complete(ev.payload)
            elif et == "LOC_UPDATE":
                self._on_loc_update(ev.payload)

        self.now = until

    # ---------------- handlers ----------------

    def _on_call_arrive(self, call: Call, record_stats: bool):
        assigned = self.dispatch.assign(self.now, call, self.vehicles, self.travel)
        if assigned:
            self._flag_busy(assigned, call.call_id)
            self.call_assignments[call.call_id] = assigned[:]

            # first response time
            first_eta = min(
                self.travel.travel_time(self.vehicles[vid].loc, call.loc_scene, self.now, self.vehicles[vid].kind)
                for vid in assigned
            )
            t_arrive_first = self.now + call.dispatch_delay + first_eta
            if record_stats:
                self.response_times[call.call_id] = t_arrive_first - call.t_call

            # move all assigned to scene
            for vid in assigned:
                self._begin_travel_to(vid, call.loc_scene, self.now + call.dispatch_delay)

            # depart scene after first arrival + scene_time
            scene_depart_time = t_arrive_first + call.scene_time
            heapq.heappush(self.event_q, Event(scene_depart_time, "SCENE_DEPART", call))
        else:
            self.wait_q.append(call.call_id)

    def _on_scene_depart(self, call: Call):
        assigned = self.call_assignments.get(call.call_id, [])
        if not assigned:
            return

        # choose transporting ambulance if needed
        transport_vid = None
        if call.hospital_time > 0.0 and any(self.vehicles[v].kind == "A" for v in assigned):
            for v in assigned:
                if self.vehicles[v].kind == "A":
                    transport_vid = v
                    break

        if transport_vid is not None:
            self._begin_travel_to(transport_vid, call.loc_hospital, self.now)
            t_arrive_hosp = self.now + self.travel.travel_time(
                self.vehicles[transport_vid].loc, call.loc_hospital, self.now, "A"
            )
            job_complete_time = t_arrive_hosp + call.hospital_time + call.handover_time
            heapq.heappush(self.event_q, Event(job_complete_time, "JOB_COMPLETE", (call.call_id, transport_vid)))

        for vid in assigned:
            if vid != transport_vid:
                self._vehicle_become_available(vid)
                self._begin_travel_to(vid, self.stations[self.vehicles[vid].home_station_id].grid_rc, self.now)

        self._check_queue()

    def _on_job_complete(self, payload: Tuple[int, int]):
        _, transport_vid = payload
        self._vehicle_become_available(transport_vid)
        self._begin_travel_to(transport_vid, self.stations[self.vehicles[transport_vid].home_station_id].grid_rc, self.now)
        self._check_queue()

    def _on_loc_update(self, vid: int):
        v = self.vehicles[vid]
        if v.route:
            v.route.pop(0)
            if v.route:
                v.loc = v.route[0]
        if v.route:
            heapq.heappush(self.event_q, Event(self.now + self.loc_update_period, "LOC_UPDATE", vid))

    # ---------------- helpers ----------------

    def _begin_travel_to(self, vid: int, dest: tuple[int, int], t_depart: float):
        v = self.vehicles[vid]
        v.route = self.travel.route(v.loc, dest)
        heapq.heappush(self.event_q, Event(max(self.now, t_depart) + self.loc_update_period, "LOC_UPDATE", vid))

    def _flag_busy(self, vids: List[int], call_id: int):
        for vid in vids:
            v = self.vehicles[vid]; v.busy = True; v.assigned_call = call_id

    def _vehicle_become_available(self, vid: int):
        v = self.vehicles[vid]; v.busy = False; v.assigned_call = None

    def _check_queue(self):
        if not self.wait_q:
            return
        id_to_call = {c.call_id: c for c in self.calls}
        still_waiting = []
        for call_id in self.wait_q:
            c = id_to_call[call_id]
            assigned = self.dispatch.assign(self.now, c, self.vehicles, self.travel)
            if assigned:
                self._flag_busy(assigned, c.call_id)
                self.call_assignments[c.call_id] = assigned[:]
                first_eta = min(
                    self.travel.travel_time(self.vehicles[vid].loc, c.loc_scene, self.now, self.vehicles[vid].kind)
                    for vid in assigned
                )
                t_arrive_first = self.now + c.dispatch_delay + first_eta
                self.response_times[c.call_id] = t_arrive_first - c.t_call
                scene_depart_time = t_arrive_first + c.scene_time
                heapq.heappush(self.event_q, Event(scene_depart_time, "SCENE_DEPART", c))
                for vid in assigned:
                    self._begin_travel_to(vid, c.loc_scene, self.now + c.dispatch_delay)
            else:
                still_waiting.append(call_id)
        self.wait_q = still_waiting