# Robust One-Shot Instance Segmentation Under Viewpoint, Illumination, Scale and Occlusion Changes

MSc-level computer vision research project. This repository currently contains
**Phase 0 only**: a reproducible Google Colab research environment.

> **Status: Phase 0 - environment setup, verified.**
> No datasets, no DINOv2 / SAM / Matcher / PerSAM code, no experiments yet.

---

## Quick start (Google Colab)

1. Set a GPU runtime: `Runtime > Change runtime type > Hardware accelerator > GPU`.
2. Open `notebooks/01_colab_setup.ipynb` in Colab
   (`File > Open notebook > GitHub`, or upload the file).
3. Edit **Section 1** only: `GITHUB_USER`, `GITHUB_REPO`, `DRIVE_FOLDER`.
4. `Runtime > Run all`.

The notebook mounts Drive, clones this repository, installs the Phase 0
dependencies, probes the GPU, creates the persistent directory tree, seeds every
RNG, demonstrates the feature-cache contract and finishes with a single
`PASS / WARN / FAIL` verdict (`PHASE0_OK`).

Full walkthrough: **[docs/colab_setup.md](docs/colab_setup.md)**.

---

## Repository layout

```
.
├── configs/
│   └── paths.yaml               # single source of truth for every path
├── docs/
│   └── colab_setup.md           # complete Colab guide + troubleshooting
├── notebooks/
│   └── 01_colab_setup.ipynb     # the Phase 0 notebook (10 sections)
├── scripts/
│   ├── environment_check.py     # GPU / CUDA / PyTorch / RAM / disk diagnostics
│   ├── reproducibility.py       # seeding, determinism, experiment metadata
│   ├── cache_utils.py           # DINOv2 feature-cache design + dummy demo
│   └── project_setup.py         # paths.yaml resolution, Drive, dirs, artefacts
├── models/                      # Phase 2: backbones (DINOv2, SAM, SAM2)
├── matching/                    # Phase 2: Matcher / PerSAM integration
├── perturbations/               # Phase 2: viewpoint, illumination, scale, occlusion
├── evaluation/                  # Phase 2: instance-segmentation metrics
├── experiments/                 # Phase 2: experiment drivers
└── visualization/               # Phase 2: qualitative figures
```

Code and configuration live in git. Everything large lives in Google Drive:

```
MyDrive/one-shot-segmentation/
├── data/{raw,processed,cache,splits}
├── checkpoints/
├── results/{baseline,robustness,proposed,ablation,final}
└── logs/
```

| Artefact | Location |
| --- | --- |
| Datasets | `MyDrive/one-shot-segmentation/data/raw/<dataset>/` |
| Model checkpoints | `MyDrive/one-shot-segmentation/checkpoints/` |
| Cached DINOv2 features | `MyDrive/one-shot-segmentation/data/cache/dinov2/<model_tag>/<split>/` |
| Experiment results | `MyDrive/one-shot-segmentation/results/<group>/` |
| Environment reports + metadata | `MyDrive/one-shot-segmentation/results/environment/` |
| Code | git clone at `/content/one-shot-segmentation` (ephemeral) |

---

## Path configuration

`configs/paths.yaml` uses `${TOKEN}` placeholders resolved by
`project_setup.resolve_paths()`. The notebook Section 1 supplies
`DRIVE_MOUNT`, `DRIVE_FOLDER` and `DRIVE_ROOT`; individual paths can be
overridden with environment variables such as `OSS_DATA_ROOT`.

```python
from scripts import project_setup as psetup

paths = psetup.resolve_paths(token_overrides={"DRIVE_FOLDER": "one-shot-segmentation"})
paths.cache_root          # .../data/cache
paths.results_proposed    # .../results/proposed
paths.checkpoint_path("dinov2_base")
```

---

## Using the modules directly

```bash
python scripts/environment_check.py                     # full GPU/CUDA report
python scripts/project_setup.py --init --report         # create and inspect the tree
python scripts/reproducibility.py --seed 1234           # seeding + metadata
python scripts/cache_utils.py --root /tmp/cache_demo    # feature-cache contract demo
```

Each module is importable from notebooks as well:

```python
from scripts import environment_check, reproducibility, cache_utils, project_setup
```

---

## Phase 0 guarantees

* GPU detected dynamically (name, total/free VRAM, compute capability) - no
  hardware model is hard-coded.
* CUDA availability, CUDA build/runtime version, cuDNN, driver, PyTorch version.
* A real GPU inference test (tensor → CUDA → matmul → value check → timing).
* Memory sufficiency verdict with actionable warnings; exit code `1` on `FAIL`.
* Idempotent Drive mount, clone, package installation, directory creation and
  cache lookups - the notebook is safe to re-run.
* Deterministic seeding (Python, NumPy, PyTorch, CUDA) plus an experiment
  metadata record (timestamp, git commit, config, seed, GPU, versions).
* A feature-cache contract keyed by image identity *and* model configuration,
  demonstrated with dummy tensors.

---

## Planned phases

| Phase | Scope |
| --- | --- |
| 0 | Colab environment, storage, paths, seeding, cache design (**current**) |
| 1 | Dataset download/verification, deterministic splits, DINOv2 feature extraction |
| 2 | Baselines: Matcher, SAM/SAM2, PerSAM; perturbation suite; robustness evaluation |
| 3 | Proposed method, ablations, final results, demo UI |

---

## Requirements

* Google Colab with a GPU runtime (minimum 8 GB VRAM, 15 GB recommended).
* A GitHub repository containing this code, or a local upload.
* Google Drive with at least 20 GB free.
* Python 3.10+ (satisfied by Colab).
