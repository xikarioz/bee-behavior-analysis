# Bee Antennae Behavioral Analysis 🐝

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![DeepLabCut](https://img.shields.io/badge/DeepLabCut-2.x-orange)](https://github.com/DeepLabCut/DeepLabCut)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20%2F%20In%20progress-yellow)]()
[![Species](https://img.shields.io/badge/Species-Apis%20mellifera-gold)]()

A complete machine-learning and signal-analysis pipeline for studying honey bee (*Apis mellifera*) antennae behavior from video recordings.

Keypoints are extracted at high resolution using **DeepLabCut**, then analyzed with dynamical systems methods, spectral analysis, latent embeddings, and algebraic topology to identify and characterize behavioral states.

---

## Table of contents

1. [Pipeline overview](#pipeline-overview)
2. [Repository structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Modules](#modules)
6. [Dependencies](#dependencies)
7. [Citing this work](#citing-this-work)
8. [License](#license)

---

## Pipeline overview

```
Raw video → [DeepLabCut inference] → Keypoints (HDF5/CSV)
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              │                             │                             │
        Signal analysis               Latent space               Event detection
        VMD · HHT · CWT              CEBRA · TDA                Twitch detector
              │                             │                             │
              └─────────────────────────────┴─────────────────────────────┘
                                            │
                                   Publication figures
                              (paper_maestro: 15 · cebra_paper: 10)
```

### Sample outputs

<!-- Once you export figures, place PNGs in assets/ and uncomment these lines -->
<!-- ![Polar rose activity plot](assets/rose_actividad.png) -->
<!-- ![VMD intrinsic modes](assets/vmd_lento.png) -->
<!-- ![CEBRA latent embedding](assets/cebra_embedding.png) -->
<!-- ![Twitch detection](assets/twitch_detection.png) -->
<!-- ![CWT vs HHT comparison](assets/cwt_vs_hht.png) -->

> **Add figures here.** Export any PNG from `paper_maestro.py` or `cebra_paper.py`, place it in `assets/`, and uncomment the lines above.

---

## Repository structure

```
bee-behavior-analysis/
├── assets/                      # README figures — place exported PNGs here
├── extraccion_dlc/              # Pose estimation with DeepLabCut
│   ├── abejas_linux.py          #   Sequential inference → HDF5/CSV
│   ├── abejas_paralelo.py       #   Batch processing (GPU)
│   ├── abejas_pi_v2.py          #   Real-time inference (Raspberry Pi + Arduino)
│   └── analizar_completo.py     #   Full extraction pipeline with export
├── analisis_y_figuras/          # Quantitative analysis and publication figures
│   ├── paper_maestro.py         #   Polar plots, VMD, HHT, CWT (15 figures)
│   ├── cebra_paper.py           #   CEBRA embeddings, TDA, Kuramoto, Wasserstein
│   ├── twitch_analysis.py       #   Ultrafast event detection
│   └── vmd_rapido.py            #   High-frequency VMD (fs = 2 Hz, 0.5 s window)
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Installation

### Prerequisites

- Python ≥ 3.9
- CUDA ≥ 11.3 (recommended for GPU inference)
- [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) installed in a separate conda environment (see their official docs)

### Setup

```bash
git clone https://github.com/xikarioz/bee-behavior-analysis.git
cd bee-behavior-analysis

python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

For PyTorch, follow the [official installation guide](https://pytorch.org/get-started/locally/) and select the build that matches your CUDA version. Then install DeepLabCut separately following [their documentation](https://github.com/DeepLabCut/DeepLabCut).

---

## Usage

Each script has a `CONFIG` block near the top. Edit the paths there before running — no command-line arguments are needed.

### 1. Extract keypoints

Open `extraccion_dlc/abejas_linux.py` and set:

```python
# ── Edit these before running ─────────────────────────────────
CONFIG_PATH = "/path/to/your/dlc_config.yaml"
VIDEO_PATH  = "/path/to/your/video.mp4"
OUTPUT_DIR  = "/path/to/output/"
```

Then run the appropriate script for your setup:

```bash
# Standard sequential inference (single video, Linux/macOS)
python extraccion_dlc/abejas_linux.py

# Batch processing across multiple videos (GPU recommended)
python extraccion_dlc/abejas_paralelo.py

# Real-time inference on Raspberry Pi with Arduino stimulus sync
python extraccion_dlc/abejas_pi_v2.py
```

### 2. Generate publication figures

Open `analisis_y_figuras/paper_maestro.py` and `cebra_paper.py` and set:

```python
# ── Edit these before running ─────────────────────────────────
CSV_PATH   = "/path/to/your/poses_completo.csv"
OUTPUT_DIR = "/path/to/output/figures/"
```

Then run:

```bash
# 15 figures: polar rose plots, VMD modes, HHT spectra, CWT vs HHT comparison
python analisis_y_figuras/paper_maestro.py

# CEBRA embeddings, TDA, Kuramoto synchronization, Wasserstein distances
python analisis_y_figuras/cebra_paper.py

# Ultrafast event (twitch) detection and behavioral state analysis
python analisis_y_figuras/twitch_analysis.py

# High-frequency VMD over short windows
python analisis_y_figuras/vmd_rapido.py
```

---

## Modules

### `extraccion_dlc/`

| Script | Description |
|---|---|
| `abejas_linux.py` | Sequential DeepLabCut inference. Exports keypoint coordinates in HDF5 and CSV with per-frame likelihood metadata. |
| `abejas_paralelo.py` | GPU-optimized batch inference engine. Processes multiple videos concurrently to reduce total analysis time. |
| `abejas_pi_v2.py` | Real-time inference on Raspberry Pi using `dlclive`. Includes hardware control via Arduino for synchronized stimulus delivery. |
| `analizar_completo.py` | Integrated pipeline: video preprocessing → inference → low-confidence filtering → export. |

### `analisis_y_figuras/`

| Script | Description |
|---|---|
| `paper_maestro.py` | Generates 15 publication figures: polar rose plots of activity, Variational Mode Decomposition (VMD) intrinsic modes, Hilbert-Huang Transform (HHT) instantaneous frequency spectra, and CWT Morlet spectrograms. |
| `cebra_paper.py` | Trains and evaluates CEBRA models for latent representations of antennal movement. Includes persistent homology (Ripser), phase synchronization (Kuramoto model), and Wasserstein distances between behavioral state distributions. |
| `twitch_analysis.py` | Automatic detection of ultrafast events (*twitches*) via adaptive thresholding. Evaluates dependency of event frequency and amplitude on behavioral state. |
| `vmd_rapido.py` | VMD over short windows (0.5 s) at high sampling rate to resolve fast rhythms in antennal dynamics. |

---

## Dependencies

| Category | Packages |
|---|---|
| Deep learning / Pose estimation | `torch`, `dlclive`, `deeplabcut` |
| Signal analysis | `scipy`, `vmdpy`, `astropy` |
| Tabular data | `numpy`, `pandas`, `polars`, `h5py` |
| Latent representation | `cebra` |
| Algebraic topology (TDA) | `ripser`, `persim` |
| Visualization | `matplotlib`, `seaborn` |
| Utilities | `tqdm`, `joblib` |

---

## Citing this work

If you use this pipeline in your research, please cite it as:

```bibtex
@misc{bee-behavior-analysis,
  author    = {Fitte, Franco},
  title     = {Bee Antennae Behavioral Analysis: A DeepLabCut + CEBRA Pipeline},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/xikarioz/bee-behavior-analysis}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
