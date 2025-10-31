# Track A2 — Thermal Signature (Transient Heat) Simulation

Lumped thermal RC model for battery, motors, ESC, and shell. Compares:
- baseline
- insulated (reduced convection/emissivity)
- heat_spreader (increased conduction to shell)
- insulated_plus_spreader

## Run
```bash
pip install numpy pandas matplotlib pyyaml
python thermal_sim.py
