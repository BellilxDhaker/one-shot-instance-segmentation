# Colab setup guide (Phase 0)

Environment reference for
**Robust One-Shot Instance Segmentation Under Viewpoint, Illumination, Scale and
Occlusion Changes**.

Phase 0 builds and verifies a reproducible Google Colab environment. It does
**not** download datasets, does not load DINOv2 / SAM / Matcher / PerSAM, and
does not implement the research method.

---

## 1. What Phase 0 delivers

| Section | Notebook cell | What it does |
| --- | --- | --- |
| 1 | `01_colab_setup.ipynb` #2 | The only cell you edit: repo coordinates, Drive folder, install switches, seed |
| 2 | #4 | Mounts Google Drive once, reports free space, defines the persistent root |
| 3 | #6 | Clones the repository to `/content/one-shot-segmentation`, adds it to `sys.path` |
| 4 | #8, #9 | Installs the Phase 0 packages, then prints the resolved versions |
| 5 | #11, #12 | GPU name / memory / CUDA / PyTorch versions, CUDA availability, real matmul test, sufficiency warnings |
| 6 | #14, #15 | Creates the persistent directory tree, lists declared datasets and checkpoints |
| 7 | #17, #18 | Seeds Python / NumPy / PyTorch / CUDA, enables deterministic kernels, writes experiment metadata |
| 8 | #20, #21 | Defines the feature-cache naming contract and proves it with dummy tensors |
| 9 | #23, #24 | One verdict table (`PASS` / `WARN` / `FAIL`) and a JSON record on Drive |
| 10 | #25 | Next steps and the phase boundary |

Supporting modules (used by the notebook, and runnable standalone):

| File | Role |
| --- | --- |
| `scripts/environment_check.py` | GPU / CUDA / PyTorch / RAM / disk diagnostics, warnings, JSON report |
| `scripts/reproducibility.py` | Seeding, determinism, git + version capture, experiment metadata |
| `scripts/cache_utils.py` | Feature-cache keys, `get_or_load()`, atomic `.pt` / `.npz` storage, dummy demo |
| `scripts/project_setup.py` | `paths.yaml` resolution, Drive mount, directory creation, dataset / checkpoint presence checks |
| `configs/paths.yaml` | Single source of truth for every path |

---

## 2. How to open the notebook in Colab

1. Push this repository to GitHub (see §4 if it is not there yet).
2. Open Google Colab: <https://colab.research.google.com/>.
3. `File > Open notebook > GitHub`, choose the repository, then
   `notebooks/01_colab_setup.ipynb`.
   Alternatively `File > Open notebook > Upload` and select the file locally.
4. **Before running anything:** `Runtime > Change runtime type >
   Hardware accelerator > GPU`, then confirm.
5. `Runtime > Run all`.

Recommended Colab settings (optional): `Edit > Settings > Runtime` →
*Leave runtime running for 60 min* while downloading or extracting data, and
*Connected runtime* only if you use a Colab-hosted runtime.

---

## 3. How Google Drive is connected

Section 2 calls `google.colab.drive.mount("/content/drive", force_remount=False)`.

* First run: an authorisation window opens, you pick the account, and a
  confirmation code is pasted back into the notebook. Nothing else is needed on
  later runs - the mount is detected and skipped.
* The persistent root becomes
  `/content/drive/MyDrive/one-shot-segmentation` (`DRIVE_FOLDER` in Section 1).
* If mounting fails (not authorised, non-Colab runtime, mounted elsewhere), the
  notebook falls back to a local folder and prints a warning that the results
  will be lost when the runtime is recycled.
* If Drive is already mounted somewhere else, keep `DRIVE_MOUNT` pointing at it
  or set `DRIVE_FOLDER` to the folder you already use.

**Rule enforced throughout the project:** code and configs live in the git
repository; datasets, checkpoints, cached features and results live on Drive.
`/content` is ephemeral and is only used for the clone and for pip caches.

---

## 4. How the repository is cloned

In Section 1 set:

```python
GITHUB_USER = "your-github-username"
GITHUB_REPO = "one-shot-segmentation"
GITHUB_BRANCH = "main"
```

Section 3 then runs
`git clone --branch main --depth 1 <url> /content/one-shot-segmentation`.

* Private repository: add the token to Colab secrets
  (`Colab > Secrets > Add new secret`, name `GITHUB_TOKEN`) and set
  `USE_GITHUB_TOKEN = True`. The token is used only for the clone and the
  remote URL is rewritten back to the clean URL immediately afterwards.
* No GitHub access at all: upload the repository yourself (Drive, Colab file
  upload, a zip) and set `USE_LOCAL_REPO = True`.
* Already cloned: the clone is skipped. Set `UPDATE_REPO = True` for
  `git pull --ff-only`.
* The repository root is added to `sys.path`, so `from scripts import ...` and
  the future `models` / `matching` / `perturbations` / `evaluation` /
  `experiments` / `visualization` packages are importable from any cell.

To create the GitHub repository the notebook expects:

```bash
git init
git add .
git commit -m "Phase 0: Colab environment setup"
git branch -M main
git remote add origin https://github.com/<user>/one-shot-segmentation.git
git push -u origin main
```

---

## 5. How to run the environment check

From the notebook: `Runtime > Run all`, then read Section 9.
`PHASE0_OK = True` means the environment is ready.

From a shell (local machine, or inside Colab):

```bash
python scripts/environment_check.py                     # human-readable report
python scripts/environment_check.py --json report.json  # + machine-readable
python scripts/project_setup.py --init --report         # create/inspect the tree
python scripts/reproducibility.py --seed 1234           # seeding + metadata
python scripts/cache_utils.py --root /tmp/cache_demo    # cache contract demo
```

`environment_check.py` exits with status `1` when the verdict is `FAIL`, so it
can be used as a CI gate.

What it reports:

* GPU name, total and free VRAM, compute capability, GPU count (queried at
  runtime - **no GPU model is hard-coded**, Colab hardware varies).
* CUDA availability, CUDA build and runtime versions, cuDNN, NVIDIA driver.
* PyTorch version and the `torch.version.cuda` it was built against.
* A functional GPU test: `randn -> cuda`, two matmuls, value sanity check,
  ms/iteration, peak VRAM.
* Resource sufficiency: minimum 8 GB VRAM (15 GB recommended for later phases),
  minimum 12 GB system RAM, minimum 20 GB free on Drive.
* Package versions, disk usage, and a consolidated `PASS` / `WARN` / `FAIL` list
  plus actionable warnings ("switch to a GPU runtime", "restart the runtime to
  free VRAM", "free up Drive space", "less VRAM than later phases need").

---

## 6. Where everything is stored

```
/content/one-shot-segmentation/          <- git clone (code + configs, ephemeral)

Google Drive
└── MyDrive/one-shot-segmentation/
    ├── data/
    │   ├── raw/<dataset>/               <- downloaded datasets
    │   ├── processed/                   <- preprocessed images / annotations
    │   ├── cache/
    │   │   ├── dinov2/<model_tag>/<split>/   <- cached DINOv2 features (.pt)
    │   │   └── demo/                    <- Section 8 dummy-feature demo
    │   └── splits/<dataset>.txt         <- deterministic split definitions
    ├── checkpoints/                     <- DINOv2 / SAM / SAM2 weights
    ├── results/
    │   ├── baseline/ robustness/ proposed/ ablation/ final/
    │   └── environment/                 <- env_report.json, phase0_verification.json, metadata
    └── logs/
```

| Question | Answer |
| --- | --- |
| Datasets | `.../data/raw/<dataset>/`, processed versions in `.../data/processed/` |
| Model checkpoints | `.../checkpoints/` (e.g. `dinov2_vitb14_pretrain.pth`) |
| Cached features | `.../data/cache/<model_family>/<model_tag>/<split>/<image_id>__<config_hash>.pt` |
| Experiment results | `.../results/<group>/`, metadata in `.../results/environment/` |

Every path is defined once in `configs/paths.yaml` and resolved by
`project_setup.resolve_paths()`. `DRIVE_FOLDER`, `DRIVE_MOUNT` and
`DRIVE_ROOT` come from the Section 1 configuration cell; any individual path can
also be overridden with an environment variable (`OSS_DATA_ROOT=...`).

---

## 7. Avoiding repeated downloads

`scripts/project_setup.py` exposes the presence checks that later phases call
before fetching anything:

```python
from scripts import project_setup as psetup

paths = psetup.resolve_paths(token_overrides={"DRIVE_FOLDER": "one-shot-segmentation"})

if not psetup.dataset_exists("coco2017_val", paths):
    ...  # download once, then it is on Drive permanently

if not psetup.checkpoint_exists("dinov2_base", paths):
    ...  # download once
```

For model downloads, `torch.hub` / `transformers` caches are redirected to Drive
by Section 6 (`HF_HOME`, `TORCH_HOME`), so a weight file is fetched once even
across runtime restarts.

---

## 8. Feature cache contract

```
image ──► DINOv2 ──► feature tensor ──► data/cache/dinov2/<model_tag>/<split>/<image_id>__<config_hash>.pt
                                                       └── later experiments only load it
```

* **Key inputs:** image identity (`image_id`, `split`, `transform_tag` for a
  perturbed variant), model family / tag / config, preprocessing (resize, crop,
  normalisation).
* **Consequence:** changing the backbone, the input resolution or the
  perturbation produces a *different* filename, so a stale tensor can never be
  reused silently.
* **Single entry point:** `cache_utils.get_or_load(root, identity, extract_fn, ...)`.
  The extractor runs on a miss only. Writes are atomic (`tmp` + `os.replace`),
  so an interrupted run cannot leave a corrupt entry that later reads as valid.
* **Each entry stores** the tensor plus metadata: cache version, identity, model
  config, preprocessing, dtype, shape, a content hash and the creation time.
* **Phase 0 status:** the contract is defined and demonstrated with dummy
  tensors (Section 8). No DINOv2 model is loaded and no real features exist yet.

---

## 9. Reproducibility

`scripts/reproducibility.py`:

* `set_seed(seed, deterministic=True)` seeds Python `random`, NumPy, PyTorch
  and all CUDA devices; sets `cudnn.deterministic=True`,
  `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True,
  warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
* `seeded_generator(seed)` and `seed_worker(worker_id)` give DataLoaders their
  own reproducible streams.
* `start_experiment(...)` returns an `ExperimentMetadata` record containing:
  UTC timestamp, git commit / branch / remote / dirty flag, resolved config,
  seed, determinism flag, GPU summary and software versions.
* `meta.save(directory)` / `log_experiment_result(meta, metrics, directory)`
  write JSON next to the results, so every number in `results/` can be traced to
  the commit and seed that produced it.

---

## 10. Re-running safely

The notebook is idempotent: re-running it skips an existing clone, packages
that already satisfy the version requirements, directories that already exist,
and cached feature entries. Nothing is deleted.

Restart procedure for a fresh session:
`Runtime > Restart runtime`, then `Runtime > Run all`.

---

## 11. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `No CUDA GPU is available` | Runtime is CPU-only: `Runtime > Change runtime type > GPU`. |
| `Only X of Y GB of VRAM is free` | Another session holds memory: restart the runtime, then re-run. |
| `Less than 20 GB free on Drive` | Free space in `MyDrive`; results and feature caches grow fast. |
| `Drive mount failed` | Re-run Section 2 and authorise the account; if Drive is mounted elsewhere, set `DRIVE_MOUNT`. |
| `Clone failed` | Wrong repo name/visibility, or a private repo without a token: set `USE_LOCAL_REPO = True`. |
| `ModuleNotFoundError: scripts` | The clone step failed or the repo root was not added to `sys.path`; re-run Section 3. |
| Packages changed after install | Restart the runtime so the new imports are picked up, then re-run from Section 1. |
| `PHASE0_OK = False` | Only the GPU-related checks can be fatal on a GPU runtime; read the warnings in Sections 5 and 9. |

---

## 12. Phase boundary

**Not implemented in Phase 0:** dataset downloading, DINOv2 feature
extraction, SAM / SAM2, Matcher, PerSAM, perturbation generation, robustness
evaluation, ablation studies, the proposed method, and the demo UI.

Phase 1 starts with dataset download / verification and deterministic splits,
then feature extraction through `cache_utils.get_or_load`, recording every run
with `reproducibility.start_experiment`. Matcher dependencies are added at that
point, not before.
