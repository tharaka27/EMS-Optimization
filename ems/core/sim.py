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
    Event-driven EMS simulation engine.

    This class simulates the lifecycle of emergency calls, vehicle dispatch,
    travel, on-scene operations, transport to hospitals, and vehicle recovery.

    The simulation core is fully decoupled from:
    - Dispatch logic (IDispatchPolicy)
    - Travel-time and routing logic (ITravelModel)
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
        """
        Initialize the simulation state.

        Parameters
        ----------
        calls
            List of emergency calls.
        stations
            Mapping from station ID to Station objects.
        vehicles
            Mapping from vehicle ID to Vehicle objects.
        travel
            Travel-time and routing model.
        dispatch
            Dispatching policy used to assign vehicles to calls.
        loc_update_period
            Time interval (seconds) between vehicle location updates.
        """
        # Sort calls chronologically
        self.calls = sorted(calls, key=lambda c: c.t_call)

        self.stations = stations
        self.vehicles = vehicles
        self.travel = travel
        self.dispatch = dispatch
        self.loc_update_period = loc_update_period

        # Event priority queue (min-heap)
        self.event_q: List[Event] = []

        # Queue of calls waiting for vehicle assignment
        self.wait_q: List[int] = []

        # Current simulation time
        self.now: float = 0.0

        # Response time tracking per call
        self.response_times: Dict[int, float] = {}

        # Mapping from call_id to assigned vehicle IDs
        self.call_assignments: Dict[int, List[int]] = {}

        # Initialize vehicle locations at home stations if unset
        for v in self.vehicles.values():
            if v.loc is None:
                v.loc = self.stations[v.home_station_id].grid_rc

    def run(
        self,
        t_start: float,
        t_end: float,
        warmup_buffer: float = 90 * 60.0
    ) -> Dict[str, float]:
        """
        Run the simulation for a specified time window.

        The simulation includes a warm-up period to allow the system
        to reach a steady state before KPIs are recorded.

        Parameters
        ----------
        t_start
            Start time of KPI recording.
        t_end
            End time of simulation.
        warmup_buffer
            Duration of warm-up period before t_start (seconds).

        Returns
        -------
        Dict[str, float]
            KPI dictionary containing eta_s and call counts.
        """
        # Warm-up phase: process calls before t_start without recording stats
        warm_lo = t_start - warmup_buffer
        for c in self.calls:
            if warm_lo <= c.t_call < t_start:
                heapq.heappush(self.event_q, Event(c.t_call, "CALL_ARRIVE", c))
        self._loop(until=t_start, record_stats=False)

        # Main simulation phase: record statistics
        for c in self.calls:
            if t_start <= c.t_call < t_end:
                heapq.heappush(self.event_q, Event(c.t_call, "CALL_ARRIVE", c))
        self._loop(until=t_end, record_stats=True)

        # Compute and return system-level survival KPI
        return compute_eta_s(self.calls, self.response_times)

    # ------------------------------------------------------------------
    # Internal event-processing loop
    # ------------------------------------------------------------------

    def _loop(self, until: float, record_stats: bool):
        """
        Process events until a specified simulation time.

        Parameters
        ----------
        until
            Time to advance the simulation to.
        record_stats
            Whether to record response time statistics.
        """
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

        # Advance simulation clock
        self.now = until

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_call_arrive(self, call: Call, record_stats: bool):
        """
        Handle arrival of a new emergency call.

        Attempts to dispatch vehicles immediately; otherwise,
        places the call in a waiting queue.

        Parameters
        ----------
        call
            Incoming emergency call.
        record_stats
            Whether response time statistics should be recorded.
        """
        assigned = self.dispatch.assign(self.now, call, self.vehicles, self.travel)

        if assigned:
            # Mark vehicles as busy
            self._flag_busy(assigned, call.call_id)
            self.call_assignments[call.call_id] = assigned[:]

            # Compute first-arrival response time
            first_eta = min(
                self.travel.travel_time(
                    self.vehicles[vid].loc,
                    call.loc_scene,
                    self.now,
                    self.vehicles[vid].kind
                )
                for vid in assigned
            )
            t_arrive_first = self.now + call.dispatch_delay + first_eta

            if record_stats:
                self.response_times[call.call_id] = t_arrive_first - call.t_call

            # Send all assigned vehicles to the scene
            for vid in assigned:
                self._begin_travel_to(vid, call.loc_scene, self.now + call.dispatch_delay)

            # Schedule departure from scene
            scene_depart_time = t_arrive_first + call.scene_time
            heapq.heappush(self.event_q, Event(scene_depart_time, "SCENE_DEPART", call))
        else:
            # No vehicles available
            self.wait_q.append(call.call_id)

    def _on_scene_depart(self, call: Call):
        """
        Handle departure of vehicles from the scene.

        Determines whether transport to hospital is required and
        schedules job completion events.

        Parameters
        ----------
        call
            Emergency call whose scene service has completed.
        """
        assigned = self.call_assignments.get(call.call_id, [])
        if not assigned:
            return

        # Select a transporting ambulance if required
        transport_vid = None
        if call.hospital_time > 0.0 and any(self.vehicles[v].kind == "A" for v in assigned):
            for v in assigned:
                if self.vehicles[v].kind == "A":
                    transport_vid = v
                    break

        # Handle transport to hospital
        if transport_vid is not None:
            self._begin_travel_to(transport_vid, call.loc_hospital, self.now)
            t_arrive_hosp = self.now + self.travel.travel_time(
                self.vehicles[transport_vid].loc,
                call.loc_hospital,
                self.now,
                "A"
            )
            job_complete_time = t_arrive_hosp + call.hospital_time + call.handover_time
            heapq.heappush(
                self.event_q,
                Event(job_complete_time, "JOB_COMPLETE", (call.call_id, transport_vid))
            )

        # Release non-transporting vehicles
        for vid in assigned:
            if vid != transport_vid:
                self._vehicle_become_available(vid)
                self._begin_travel_to(
                    vid,
                    self.stations[self.vehicles[vid].home_station_id].grid_rc,
                    self.now
                )

        self._check_queue()

    def _on_job_complete(self, payload: Tuple[int, int]):
        """
        Handle completion of hospital transport and handover.

        Parameters
        ----------
        payload
            Tuple of (call_id, vehicle_id).
        """
        _, transport_vid = payload
        self._vehicle_become_available(transport_vid)
        self._begin_travel_to(
            transport_vid,
            self.stations[self.vehicles[transport_vid].home_station_id].grid_rc,
            self.now
        )
        self._check_queue()

    def _on_loc_update(self, vid: int):
        """
        Update vehicle location along its route.

        Parameters
        ----------
        vid
            Vehicle ID.
        """
        v = self.vehicles[vid]
        if v.route:
            v.route.pop(0)
            if v.route:
                v.loc = v.route[0]

        if v.route:
            heapq.heappush(
                self.event_q,
                Event(self.now + self.loc_update_period, "LOC_UPDATE", vid)
            )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _begin_travel_to(self, vid: int, dest: tuple[int, int], t_depart: float):
        """
        Initialize vehicle travel to a destination.

        Parameters
        ----------
        vid
            Vehicle ID.
        dest
            Destination grid location.
        t_depart
            Departure time.
        """
        v = self.vehicles[vid]
        v.route = self.travel.route(v.loc, dest)
        heapq.heappush(
            self.event_q,
            Event(max(self.now, t_depart) + self.loc_update_period, "LOC_UPDATE", vid)
        )

    def _flag_busy(self, vids: List[int], call_id: int):
        """
        Mark vehicles as busy and assign them to a call.

        Parameters
        ----------
        vids
            List of vehicle IDs.
        call_id
            Call ID.
        """
        for vid in vids:
            v = self.vehicles[vid]
            v.busy = True
            v.assigned_call = call_id

    def _vehicle_become_available(self, vid: int):
        """
        Mark a vehicle as available for dispatch.

        Parameters
        ----------
        vid
            Vehicle ID.
        """
        v = self.vehicles[vid]
        v.busy = False
        v.assigned_call = None

    def _check_queue(self):
        """
        Attempt to dispatch vehicles to queued calls.

        Calls that still cannot be served remain in the waiting queue.
        """
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
                    self.travel.travel_time(
                        self.vehicles[vid].loc,
                        c.loc_scene,
                        self.now,
                        self.vehicles[vid].kind
                    )
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
