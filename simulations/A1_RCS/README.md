# Track A1 — Radar Signature (RCS) — Simulation Starter

This is a **comparative (toy)** RCS simulator for micro-UAV geometries. It generates:
- Heatmaps: RCS vs. frequency (2–10 GHz) and aspect angle (−90°…+90°)
- Line plots at 0° and 45° (compare shapes)
- Snapshot CSVs at 5 GHz, 0° aspect for quick tables

> ⚠️ Outputs show **relative trends**, not certified absolute RCS.

## Requirements
- Python 3.9+
- `pip install numpy pandas matplotlib`

## How to run
```bash
python rcs_sim.py
