# Architecture - one-shot instance segmentation (Phases 1 and 2)

Two frozen pretrained models, no training, no fine-tuning, no external repository.
The reference mask is the only object-specific input.

## 1. Data flow

```
reference image + reference mask          target image
              |                                 |
              v                                 v
   FeatureEncoder.extract_batch  ->  FeatureMap(reference), FeatureMap(target)
              |                                 |
              v                                 v
   ReferenceRepresentation  (masked tokens)     TargetRepresentation (all tokens)
              |                                 |
              +-------------> cosine_similarity (max + prototype) <----+
                                          |
                                          v
                              similarity_map  (H, W) in [0, 1]
                                          |
                                          v
                              localize -> Candidate list (box, region, score)
                                          |
                                          v
                              Segmenter.segment_region -> predicted_mask (H, W)
```

## 2. Stages and modules

| Step | Module | Responsibility | Output |
| --- | --- | --- | --- |
| 1 | `models/feature_encoder.py` | `facebook/dinov2-base`, last block, frozen | `FeatureMap` of L2-normalised patch tokens `(C, h, w)` |
| 2 | `matching/reference_representation.py` | pool the reference mask onto the patch grid, keep covered patches | `ReferenceRepresentation` |
| 3 | `matching/correspondence.py` | `build_target_representation` | `TargetRepresentation` |
| 4 | `matching/correspondence.py` | `cosine_similarity` between target patches and reference patches | `CorrespondenceResult` with `similarity_map` |
| 5 | `matching/localization.py` | threshold, clean, label, rank connected components | `LocalizationResult` with `Candidate` boxes, regions and scores |
| 6 | `models/segmenter.py` | `facebook/sam-vit-base`, frozen, prompted with box + centre point | `SegmentationResult` with `predicted_mask` |
| 7 | `pipeline/one_shot_segmentation.py` | runs 1-6, times each stage | `OneShotResult` |
| 8 | `visualization/results.py` | result panels, patch grid, similarity heatmap | matplotlib figures |
| 9 | `evaluation/metrics.py` | IoU, Dice, precision, recall | `MaskMetrics`, `EvaluationReport` |
| 9 | `evaluation/cases.py` | loads cases from a JSON manifest and runs them | `EvaluationReport` |

## 3. Model choices and why

### `facebook/dinov2-base` for features

* **Spatial quality.** DINOv2 is trained without a text-image alignment loss, so
  its patch tokens stay spatially consistent. That is the property this pipeline
  depends on: Step 4 is a per-patch comparison, not a global image embedding.
* **Resolution.** Patch size 14, and the image is resized (never cropped) so both
  sides are multiples of 14. At `max_size: 518` a 512x384 image yields a 37x28
  grid of 1036 tokens - fine enough to localise an object, coarse enough to stay
  cheap.
* **Licence and access.** Apache-2.0, served by the Hugging Face hub, loaded with
  `transformers` alone. Nothing is cloned, and no other project's code is reused.
* **Cost.** 86M parameters, one forward pass per image, `eval()` plus
  `requires_grad_(False)`.
* **Layer.** `layer: -1`, the last transformer block, which is the standard
  choice for DINOv2 dense prediction. Earlier blocks are reachable through the
  same parameter, but nothing in Phase 1 tunes them.

### `facebook/sam-vit-base` for segmentation

* **Promptable.** SAM takes exactly the weak supervision a one-shot setting can
  afford - one box and one point - and returns a pixel mask. No reference mask is
  ever passed to SAM, so localisation and segmentation stay independent.
* **No adaptation.** Promptable inference needs no fine-tuning, no LoRA and no
  per-object optimisation, so the method remains a preprocessing pipeline plus
  two frozen forward passes.
* **Cost.** 91M parameters, the smallest SAM variant; three candidate masks are
  returned and the highest predicted-IoU one is kept.
* **Alternative considered.** Grounded SAM or a SAM 2 tracker would add a
  detector or a memory bank this phase does not need, and an external repository
  dependency of its own.

### Rejected alternatives

* **CLIP / SigLIP image embeddings.** Strong semantics, but the global pooled
  vector is exactly the spatial information this pipeline must not throw away.
* **A trained few-shot model (PerSAM, FSS-1000 style).** It needs a training
  stage and a per-dataset checkpoint, which contradicts the no-training scope.
* **Segment Anything as the matcher.** SAM gives masks, not features for
  comparison, so it cannot produce the correspondence map that localisation needs.

## 4. Design decisions worth knowing

* **The image is resized, never cropped.** Patch geometry is therefore known
  exactly, which is what lets the reference mask be pooled onto the target's own
  grid without guessing. `FeatureMap.input_size` and `grid_size` record it.
* **Spatial information is kept twice.** `ReferenceRepresentation` stores the
  selected tokens with their `(row, col)` positions, and the full `coverage` map
  as well. Collapsing the object into one vector would destroy the layout that
  Step 5 needs.
* **Two aggregations, then average.** `max` similarity is sharp and good for
  repeated texture; the coverage-weighted prototype is robust and describes the
  object as a whole. `aggregate: both` uses their mean, so agreement between the
  two acts as a cheap confidence signal.
* **Robust normalisation.** Patch scores are min-max normalised between the 1st
  and 99th percentiles, so a few extreme values cannot flatten the rest of the map.
* **Localisation is a separate stage.** It answers *where* and *how strongly*;
  segmentation answers *which pixels*. Both are inspectable alone in the notebook,
  and the boxes can be scored without loading SAM.
* **A very small reference object cannot reach `min_coverage`.** In that case the
  most covered patches are used and a warning is raised, rather than silently
  producing an empty reference.
* **`confidence` is the mean similarity inside the selected region.** It is a
  ranking signal across runs, not a calibrated probability. It is reported next
  to SAM's own predicted IoU, which is a different quantity and is kept separate.
* **Devices are reconciled, not assumed.** A reference built by
  `build_reference_representation` is on the feature device, but one restored with
  `ReferenceRepresentation.load` is on the CPU, because that is where NumPy arrays
  live. Comparing it with GPU target features used to fail inside a matmul with a
  bare "expected all tensors to be on the same device". `cosine_similarity` now
  calls `align_reference` to move the whole representation onto the target's device
  and warns when it does, and `weighted_mean_token` moves the coverage weights onto
  the tokens' device itself. `ReferenceRepresentation.to(device)` is available for
  callers who would rather do it themselves.

## 5. Public interface

```python
from pipeline.one_shot_segmentation import segment_object

result = segment_object(reference_image, reference_mask, target_image)
result.predicted_mask      # (H, W) bool, target resolution
result.candidate_region    # (H, W) bool or None
result.similarity_map      # (H, W) float in [0, 1]
result.confidence          # float in [0, 1]
result.runtime             # per-stage seconds, total, devices, model ids
result.save("results", stem="one_shot_result")
```

Every stage is injectable, which is how `tools/smoke_test.py` exercises the whole
pipeline on a laptop CPU with test doubles instead of checkpoints.

## 6. Reproduction

`notebooks/02_one_shot_pipeline.ipynb` (Phase 1) and
`notebooks/03_dataset_evaluation.ipynb` (Phase 2) are generated by
`tools/embed_sources.py`, which embeds the project modules into each notebook
and prints a fingerprint of them. The notebooks therefore run in Colab without
cloning anything, and neither can ever drift from the source:

```
python tools/embed_sources.py           # regenerate both
python tools/embed_sources.py --check   # fail if either is stale
python tools/test_notebook_bundle.py    # freshness + stale-session reload, offline
python tools/test_dataset_integrity.py  # decoders, pairing, splits, fault injection
python tools/test_benchmark_suite.py    # aggregation, provenance, figures, runner
python tools/run_notebook.py --notebook notebooks/03_dataset_evaluation.ipynb
```

`configs/paths.yaml` records the paths, model ids, stage defaults and benchmark
selection rules. Nothing reads it at import time - it is documentation that
matches the code, and the notebooks set the same values explicitly.

## 7. Datasets

`tools/download_datasets.py` fetches COCO-20i, FSS-1000, LVIS-92i, PACO-Part and
Pascal-Part into the reference layout. Public archives come over HTTP; the
few-shot mask archives are on Google Drive and are skipped until their file ids
are added to `tools/drive_ids.json`. `evaluation/cases.py` then reads a JSON
manifest of paths, so no dataset-specific split format is assumed.

The Phase 2 benchmark does not use those five sets: it builds its own pairs from
**COCO 2017 val** (`instances_val2017.json`: 5,000 images, 36,781 instance
annotations, 80 categories), because it needs per-instance masks with measured
areas rather than a category split, and it fetches only the annotation member of
the 241 MB zip plus the few hundred images the pairs actually reference. The
survey of what else was considered - OVIS and DAVIS for same-object video,
PASCAL3D+ for dense viewpoint labels, YouTube-VIS for tracking - is in
`docs/dataset_comparison.md`, with the reason each was put second.

## 8. Phase 2: the reproducible benchmark

Phase 1 was verified on one synthetic pair, which proves the plumbing and
nothing else. Phase 2 measures the same frozen pipeline on data that varies the
way photographs vary, and it is built so that a reader can reproduce every
number without trusting the person who produced them.

```bash
python tools/build_benchmark.py --root data --pairs 100 --seed 0   # pairs + splits
python scripts/check_dataset.py --root data                        # gate
python tools/run_benchmark.py --split all --output results/baseline --seed 0
```

Four components, none of which touches the Phase 1 algorithm:

* **`tools/build_benchmark.py`** selects 100 reference/target pairs from COCO
  2017 val annotations, rasterises their masks (polygons through Pillow, RLE
  through a decoder written against the COCO column-major packing), and writes
  the samples and the `train`/`validation`/`test` manifests. Selection is
  deterministic for a seed, round-robin over categories, never uses the same
  image twice, and keeps masks whose area is between 0.5% and 60% of the frame.
  Images are fetched individually and hard-linked into `data/samples/`, so a
  pair costs one copy of each JPEG rather than an unzipped 241 MB archive.
* **`evaluation/dataset_integrity.py`** is a gate, not a report: `check_dataset`
  returns findings tagged `critical`, `warning` or `info`, and both the runner
  and the notebook refuse to start when anything critical is present. It checks
  file presence, that images open and match their masks' dimensions, that masks
  are neither empty nor full, that pair metadata agrees with the pixels, that
  no sample id or image pair is duplicated, and that the splits cover every
  sample without leaking across them.
* **`tools/run_benchmark.py`** scores one split through the existing
  `segment_object` interface and writes four things: `results.csv` with a fixed
  column order (metrics, timings, then the condition columns), `summary.json`
  with overall and grouped statistics, `provenance.json` with the seed, device,
  package versions, model ids and dataset identity, and one JSON record per
  pair under `samples/`. A record is written the moment its pair finishes, so a
  run that dies halfway resumes exactly where it stopped and never re-runs a
  cached pair unless `--force` is given.
* **`visualization/benchmark.py`** draws one seven-panel figure per pair
  (reference image, reference mask, target image, similarity map, candidate
  region, predicted mask, truth/prediction overlay) and the three run-level
  figures (IoU distribution, IoU by category, seconds per stage).

Two honesty rules are enforced in the data rather than left to a caption. The
`identity_level` of every COCO pair is `category`: the reference and the target
are two different objects of the same class, because COCO has no instance video
and faking same-object views would manufacture the robustness this phase is
supposed to measure. And only `scale_condition` is measured (from the mask area
ratio); `viewpoint_condition`, `illumination_condition` and
`occlusion_condition` are recorded as `unknown` rather than guessed from the
class name. The pairing rules and the rejected alternatives are written down in
`docs/pairing_protocol.md`, the dataset survey with citations in
`docs/dataset_comparison.md`.

`notebooks/03_dataset_evaluation.ipynb` is generated by the same
`tools/embed_sources.py` as notebook 02, embeds the fourteen modules the
benchmark needs, mounts Drive, runs the gate, and resumes like the script does.
`OSB_LIMIT=1` turns a full run into a one-pair smoke test.

## 9. Limitations to state in any report

* A single box-and-point prompt can settle on a neighbouring instance when two
  objects look alike; the candidate list exists to make that visible.
* `confidence` and the localisation threshold were chosen from the synthetic demo
  pair, not tuned on an evaluation set, and no threshold sweep was run.
* One DINOv2 layer and one backbone size were used; neither was ablated.
* Results on a single synthetic example are a smoke test, not evidence of
  generalisation.
* The Phase 2 baseline is **class-level transfer**, not same-object
  re-identification: a poor score on `person` means the frozen pipeline does not
  carry across two instances of a class, which is a different claim than
  viewpoint robustness. The `identity_level` column exists so this cannot be
  read as the latter.
* 100 pairs over 80 categories leaves most categories with one or two pairs, so
  per-category means are thin; `summary.json` stores a `count` beside every
  group, and the pooled overall and scale-condition groups are the numbers to
  quote.
* No artificial viewpoint, illumination, scale, occlusion, noise or blur
  transformations were applied in this phase, so nothing here measures
  sensitivity to those factors. That is Phase 3's job.
* The benchmark ran on CPU (~25-30 s per pair); no GPU timing is claimed, and
  `provenance.json` records `cuda_available: false` for exactly this run.
