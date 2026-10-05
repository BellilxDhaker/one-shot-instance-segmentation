# Architecture - one-shot instance segmentation (Phase 1)

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

`notebooks/02_one_shot_pipeline.ipynb` is generated by `tools/embed_sources.py`,
which embeds the project modules into the notebook and prints a fingerprint of
them. The notebook therefore runs in Colab without cloning anything, and it can
never drift from the source:

```
python tools/embed_sources.py           # regenerate
python tools/embed_sources.py --check   # fail if the notebook is stale
```

`configs/paths.yaml` records the paths, model ids and stage defaults. Nothing
reads it at import time - it is documentation that matches the code, and the
notebook sets the same values explicitly.

## 7. Datasets

`tools/download_datasets.py` fetches COCO-20i, FSS-1000, LVIS-92i, PACO-Part and
Pascal-Part into the reference layout. Public archives come over HTTP; the
few-shot mask archives are on Google Drive and are skipped until their file ids
are added to `tools/drive_ids.json`. `evaluation/cases.py` then reads a JSON
manifest of paths, so no dataset-specific split format is assumed.

## 8. Limitations to state in any report

* A single box-and-point prompt can settle on a neighbouring instance when two
  objects look alike; the candidate list exists to make that visible.
* `confidence` and the localisation threshold were chosen from the synthetic demo
  pair, not tuned on an evaluation set, and no threshold sweep was run.
* One DINOv2 layer and one backbone size were used; neither was ablated.
* Results on a single synthetic example are a smoke test, not evidence of
  generalisation.
