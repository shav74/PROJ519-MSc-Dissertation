# Track A3 — Acoustic Signature (Simulation)

This sim produces:
- **LAeq vs distance** curves (10–100 m) for each configuration
- **Mel-spectrograms** (optional) at a reference distance
- **2D SPL footprints** on the ground plane
- A **comparison table** (`outputs/summary_tables/comparison.csv`) with LAeq@10/20/50/100 m, RPM at hover, BPF, added mass, and benefit-per-gram

## Install
```bash
pip install numpy pandas matplotlib pyyaml
pip install librosa scipy   # only if you want spectrograms
