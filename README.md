# EMS Framework

A modular Python framework for **Emergency Medical Services (EMS)** simulation and **resource allocation optimization**.  
It combines an **event-driven ambulance simulator** with flexible components for travel modeling, dispatch strategies, and optimization algorithms.

## See the preliminary work
[Event Driven Emergency Medical Services](https://github.com/tharaka27/Event-Driven-Emergency-Medical-Services)

##  Overview

This framework allows you to:

- Simulate ambulance operations on a **grid-based map**.
- Model **response times**, **patient categories**, and **vehicle movements**.
- Evaluate system performance via **heterogeneous survival efficiency (ηₛ)**.
- Optimize station locations and fleet allocation using pluggable algorithms like **Genetic Algorithms (GA)**.

The architecture separates simulation, policies, travel maps, and optimization —  
so you can easily swap algorithms or maps without touching the core simulation.


## Project Structure

```txt
ems/
├── core/              # Simulation engine and basic types
│   ├── types.py       # Call, Station, Vehicle, Event dataclasses
│   ├── kpis.py        # Survival models and ηₛ computation
│   ├── builder.py     # Build vehicles from allocation plans
│   └── sim.py         # Main event-driven Simulation class
│
├── policies/          # Dispatch logic modules
│   ├── dispatch_base.py   # IDispatchPolicy interface
│   └── dispatch_nearest.py# Greedy nearest-ETA dispatch
│
├── travel/            # Map and travel-time models
│   ├── base.py            # ITravelModel interface
│   ├── grid8.py           # 8-neighbor grid model (default)
│   └── grid_manhattan.py  # Manhattan grid (simpler baseline)
│
├── plans/             # Resource allocation schemas
│   └── schema.py      # StationPlan, AllocationPlan, FleetPlan
│
├── optim/             # Optimization algorithms
│   ├── base.py        # IOptimizer interface
│   └── ga.py          # Genetic Algorithm implementation
│
├── data/              # Call and station data loaders
│   └── synth.py       # Synthetic mock call generator
│
└── run/               # Experiment scripts / entry points
└── run_ga.py      # Example GA optimization experiment
```

## Quick Start

### 1. Clone or copy the project structure.

### 2. Run the demo Genetic Algorithm example:
```bash
python -m ems.run.run_ga