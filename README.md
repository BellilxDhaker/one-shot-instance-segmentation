# One-Shot Instance Segmentation - Phase 1

Training-free instance segmentation from a single reference image and mask. Two
frozen pretrained models, one linear pipeline, no fine-tuning, no external
repository:

| stage | model | checkpoint | output |
| --- | --- | --- | --- |
| dense features | DINOv2 ViT-B/14 | `facebook/dinov2-base` | patch tokens for both images |
| prompted mask | Segment Anything ViT-B | `facebook/sam-vit-base` | `predicted_mask` |

```python
from pipeline.one_shot_segmentation import segment_object

result = segment_object(reference_image, reference_mask, target_image)
result.predicted_mask, result.candidate_region, result.similarity_map
result.confidence, result.runtime
```

> **Status: Phase 1 implemented and verified** end to end on the synthetic demo
> pair with the real checkpoints: IoU 0.996, Dice 0.998, precision 1.000,
> recall 0.996 on CPU in 15 s. Phase 2 (robustness under viewpoint, illumination,
> scale and occlusion; baselines; the proposed method) is out of scope here and
> has not been started.

Design, model rationale and limitations: **[docs/architecture.md](docs/architecture.md)**.

---

## Quick start (Google Colab)

1. `Runtime > Change runtime type > Hardware accelerator > GPU`.
2. Open `notebooks/02_one_shot_pipeline.ipynb` in Colab.
3. `Runtime > Run all`.

Nothing needs to be cloned, unzipped or uploaded: the notebook embeds the project
modules and writes them to a scratch directory on `sys.path`. It runs on the
synthetic demo pair by default, so a clean runtime produces a result in about a
minute. Datasets are never downloaded unless you set `DOWNLOAD_DATASETS = True`.

The notebook follows the same order as the code: Environment, Datasets, Load
models, Load reference, Load target, Extract features, Correspondence,
Localization, Segmentation, Full pipeline, Visualize, Evaluate, Save result.

---

## Repository layout

```
.
├── configs/paths.yaml               # paths, model ids, stage defaults
├── docs/architecture.md             # data flow, model choice, limitations
├── models/
│   ├── feature_encoder.py           # step 1  DINOv2 dense features
│   └── segmenter.py                 # step 6  SAM with box + point prompts
├── matching/
│   ├── reference_representation.py  # step 2  masked reference tokens
│   ├── correspondence.py            # steps 3-4  target tokens, similarity
│   └── localization.py              # step 5  candidates, boxes, confidence
├── pipeline/one_shot_segmentation.py# step 7  segment_object, OneShotResult
├── evaluation/
│   ├── metrics.py                   # step 9  IoU, Dice, precision, recall
│   └── cases.py                     # JSON-manifest cases for a dataset
├── visualization/results.py         # step 8  result panels, patch grid
├── tools/
│   ├── embed_sources.py             # generates + verifies the notebook
│   ├── download_datasets.py         # COCO-20i, FSS-1000, LVIS, PACO, Pascal-Part
│   ├── make_demo_pair.py            # deterministic synthetic example
│   ├── smoke_test.py                # end-to-end test, stubs or real models
│   ├── run_notebook.py              # headless run of the notebook cells
│   ├── test_download_datasets.py    # offline test of the dataset layout
│   └── test_notebook_bundle.py      # notebook freshness + stale-session guard
└── notebooks/02_one_shot_pipeline.ipynb
```

`data/`, `datasets/`, `checkpoints/`, `results/` and `logs/` hold generated or
downloaded artefacts and are git-ignored.

---

## Using the modules directly

```bash
pip install "transformers>=4.40" accelerate safetensors

python tools/smoke_test.py                    # end-to-end, test doubles, no downloads
python tools/smoke_test.py --real --device cuda  # end-to-end, real checkpoints
python tools/test_download_datasets.py        # dataset layout logic, offline
python tools/test_notebook_bundle.py         # notebook freshness + stale-session guard
python tools/run_notebook.py                  # run the notebook in order, headless
python tools/embed_sources.py                 # regenerate the notebook
python tools/embed_sources.py --check         # fail if the notebook is stale

python tools/download_datasets.py --list
python tools/download_datasets.py --only coco20i pascal --val-only
python tools/download_datasets.py --verify
```

`tools/smoke_test.py` runs the real pipeline shape against a colour-and-gradient
patch descriptor and a box-filling segmenter, so the plumbing, the mask pooling,
the correspondence maths, the metrics and the plots are all verified on a laptop
with no network and no checkpoints. The stubs are labelled as stubs; only
`--real` exercises the pretrained models.

---

## Datasets

```python
from tools.download_datasets import download_all, print_layout_table

download_all("datasets", val_only=True, stage="images")
print_layout_table("datasets")
```

| dataset | images | few-shot masks |
| --- | --- | --- |
| COCO-20i | `val2014` + official annotations (HTTP) | Google Drive |
| FSS-1000 | Google Drive | Google Drive |
| LVIS-92i | COCO 2017 (HTTP, shared with PACO) | Google Drive |
| PACO-Part | COCO 2017 (HTTP, shared) | Google Drive |
| Pascal-Part | VOC 2010 + parts (HTTP) | Google Drive |

The mask archives are on Google Drive, so they are **skipped** until you add their
file ids to `tools/drive_ids.json` (or pass `--drive-ids key=value`, or set
`GDRIVE_IDS`). The layout is created either way and `--verify` reports exactly
what is still missing. `evaluation/cases.py` reads a `data/cases.json` manifest of
paths, so no dataset-specific split format is assumed:

```json
{"root": "datasets",
 "cases": [{"name": "coco-20i-val-1",
            "reference_image": "COCO2014/val2014/COCO_val2014_000000391895.jpg",
            "reference_mask": "COCO2014/annotations/val2014/000000391895.png",
            "target_image": "COCO2014/val2014/COCO_val2014_000000000139.jpg",
            "target_mask": "COCO2014/annotations/val2014/000000000139.png"}]}
```

---

## The notebook is generated, and stays honest

`notebooks/02_one_shot_pipeline.ipynb` embeds the eleven project modules and prints
a fingerprint of them. Regenerate it after any source change:

```
python tools/embed_sources.py
python tools/embed_sources.py --check    # --check exits 1 when stale
```

Each embedded module also carries a `NOTEBOOK_SOURCE_REVISION` marker, and the
cell right after the source cell compares the modules in memory against it. This
catches the one confusing failure mode of a generated notebook: re-running the
source cell rewrites the files but does not reload what Python already imported,
so a session keeps executing the previous code and a traceback quotes line
numbers from two different versions of the same file. The check names the
mismatched modules and tells you to restart the runtime; `python
tools/test_notebook_bundle.py` asserts that behaviour offline.

---

## Requirements

* Python 3.10+, `torch`, `transformers>=4.40`, NumPy, SciPy, Matplotlib, Pillow.
  Colab ships all of these; only `transformers` is installed by the notebook.
* A GPU runtime for comfortable speed (DINOv2-base plus SAM ViT-B fit in 8 GB).
  The pipeline also runs on CPU - the verified demo run took 15 s.
* ~750 MB of checkpoint downloads on first run.
* A few GB per dataset; all five together are ~40 GB of COCO images, so use
  `--val-only` while iterating.
