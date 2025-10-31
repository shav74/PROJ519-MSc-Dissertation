# Acoustic and Visual Stealth in Drones: Detection and Performance Trade-offs

End-to-end code and assets for a dissertation evaluating how visual camouflage and acoustic-quieting measures affect AI-based detection (YOLO for vision, CRNN for audio) and flight performance (noise levels, spectra, endurance).

---

## What this repository contains

- Data preparation for three sources: Anti-UAV-VIS, Drone-vs-Bird, and Own recordings (vision + audio)
- Training and evaluation: YOLO (vision), CRNN (audio)
- Inspection tools: hit/miss split, crop saving, failure analysis
- Plotting: detection curves, confusion metrics, LAeq and spectra, endurance deltas
- Reproducible experiment configurations via YAML


## Requirements and installation

Create an environment and install dependencies.

```bash
# Option A: conda
conda create -n uavstealth python=3.10 -y
conda activate uavstealth
pip install -r env/requirements.txt

# Option B: venv
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r env/requirements.txt
Suggested env/requirements.txt:

shell
Copy code
torch>=2.2
torchvision>=0.17
torchaudio>=2.2
ultralytics>=8.2
numpy
pandas
scikit-learn
matplotlib
librosa
tqdm
pyyaml
rich
opencv-python
soundfile
Datasets and expected layout
Keep data/raw read-only. All derived files go under data/interim and data/processed.
```
## Acknowledgements
Thanks to the Anti-UAV-VIS and Drone-vs-Bird dataset authors, the Ultralytics YOLO team, and the open-source CRNN community.

## Maintainer
J. A. S. Silva — issues and suggestions are welcome.