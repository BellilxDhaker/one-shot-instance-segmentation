# Pairing protocol

This document defines what a *pair* in the Phase 2 benchmark is, how it was
chosen, what each condition column is allowed to say, and what was deliberately
not done. It is the contract that `results/baseline/results.csv` was produced
under; a result is only meaningful read against it.

## What a pair is

One pair = one reference (image + mask) and one target (image + mask), where the
two masks are the *same category* of object. The pipeline sees only the
reference; the target mask is used exclusively for scoring.

```
samples/sample_0001/
  reference.jpg          a photograph containing the object
  reference_mask.png     that object, as annotated by the source corpus
  target.jpg             a different photograph, same category
  target_mask.png        the annotated object in the target photograph
  metadata.json          identity, conditions, measurements, provenance
```

Images are hard-linked from `data/raw/coco2017-val/`, so the same photograph
never occupies disk twice while every sample still stands alone.

## Identity level — the central caveat

**COCO does not record which physical object is which across images.** There is
therefore no honest way to pair *the same object* twice. Every sample declares:

```json
"identity_level": "category",
"object_id": "coco2017-112568-993776"
```

`identity_level` is one of:

| value | meaning | available now |
|---|---|---|
| `category` | reference and target are different instances of one category | **this build** (COCO val2017) |
| `instance` | the same tracked physical object appears in both frames | not yet — OVIS, DAVIS, YouTube-VOS |

`object_id` is a stable identifier for the *pair* when `identity_level` is
`category`. For a future `instance` build it will be the tracked instance
identifier. Nothing in the result table may be described as same-object
matching unless `identity_level == "instance"`.

## Selection rules

All rules are deterministic given `instances_val2017.json` and `--seed`, so the
pair set is reproducible without storing the images:

1. **Exclude** `iscrowd` annotations; they are regions, not objects.
2. **Exclude** instances whose mask area is outside `[0.005, 0.60]` of the frame.
   The lower bound removes masks too small to score meaningfully; the upper bound
   removes the near-full-frame masks where any prediction would score well.
3. **Require** a category to have usable instances in **two different images**.
4. **Balance categories round-robin**, so `person` cannot dominate: with 100
   pairs across 80 categories, every COCO category appears at least once and
   none exceeds a handful of pairs.
5. **Never reuse a photograph inside a pair**: `reference.image_id !=
   target.image_id`, always.
6. **Split 60 / 20 / 20** with a seeded shuffle (`seed + 1`) into
   `train` / `validation` / `test`.

Measured distribution of the 100 built pairs (seed 0):

| property | value |
|---|---|
| pairs | 100 |
| categories | 80 |
| `scale_condition` | 35 `similar`, 34 `smaller_in_target`, 31 `larger_in_target` |
| identity level | 100 × `category` |
| splits | 60 train / 20 validation / 20 test |

## Condition columns: what may be written

The protocol forbids inventing labels. Each column is either **measured**,
**sourced**, or `"unknown"` — never guessed.

| column | rule | this build |
|---|---|---|
| `viewpoint_condition` | `"unknown"` unless the source annotates pose | `unknown` (COCO has none; PASCAL3D+ would supply azimuth/elevation) |
| `illumination_condition` | `"unknown"` unless the source annotates lighting | `unknown` |
| `scale_condition` | **measured** from mask area ratio, thresholds below | measured |
| `occlusion_condition` | `"unknown"` unless the source annotates occlusion | `unknown` (OVIS would supply none/slight/severe) |

`scale_condition` thresholds, applied to
`area(target_mask) / area(reference_mask)`:

| ratio | value |
|---|---|
| `< 0.5` | `smaller_in_target` — the object occupies less of the target frame |
| `0.5 … 2.0` | `similar` |
| `> 2.0` | `larger_in_target` |

The raw ratio is stored in `metadata.json` under
`measured.area_ratio_target_over_reference`, so any other binning can be
recomputed. Scale is a *frame-relative* quantity: it says how much of the
photograph the object covers, not how big the object is in metres.

`unknown` is a real value, not a missing one. Aggregating over a column of
`unknown` tells the reader the column contributes nothing; silently averaging
guessed values would not.

## Prohibitions in this phase

Implemented on purpose as things the benchmark does **not** do yet, so that any
change in score is attributable to the pairing rather than to the pipeline or to
preprocessing:

- **No artificial transformations.** No synthetic viewpoint rotation, scale
  change, colour/illumination shift, blur, noise, cropping, or cut-and-paste
  occlusion. Every variation observed is one that actually occurred when the
  photograph was taken.
- **No changes to the segmentation algorithm.** The frozen DINOv2 encoder, the
  masked reference representation, correspondence, localisation and the frozen
  SAM segmenter are byte-identical to the Phase 1 implementation. The benchmark
  is a measurement instrument, not a model improvement.
- **No tuning on the test split.** The 20 test pairs are scored once; anything
  inspected and adjusted against goes into `train` or `validation`.
- **No post-hoc filtering of failures.** A 0 IoU pair stays in the CSV.

## Mask rasterisation and how it was checked

COCO stores instance masks as polygons or as run-length encoded (RLE) masks.
Both are rasterised here without pulling in an external format dependency. Two
independent checks confirm the pixels agree with the source:

- **Polygon path:** across all 200 masks of the 100 built pairs, the tight
  bounding box of the rendered mask deviates from the annotation's own `bbox` by
  at most **1.98 px**.
- **RLE path:** all 446 RLE annotations in COCO val2017 decode with a worst-case
  bbox deviation of **1.00 px** and zero unusable masks.

Pixel counts can differ from the annotation's `area` field by ~1.8% median and
up to ~16% on the smallest objects, because Pillow and pycocotools round polygon
edges differently; that disagreement is on the object boundary only and is why
the bbox check is the primary test. Unit coverage lives in
`tools/test_dataset_integrity.py`.

## Integrity gate

`python scripts/check_dataset.py --root data` runs before any benchmark run and
exits non-zero on a critical finding, refusing to produce numbers from damaged
data: missing files, unreadable images, empty masks, mask/image size mismatches,
unparseable metadata, duplicate sample ids, or a split manifest whose paths do
not resolve. Warnings (metadata disagreeing with pixels, a photograph reused
across samples, the same pair appearing in two splits) pass by default and fail
under `--strict`.

## Reproducing this benchmark

```bash
python tools/build_benchmark.py --pairs 100 --seed 0     # deterministic pair set
python scripts/check_dataset.py --root data              # integrity gate
python tools/run_benchmark.py --split all --output results/baseline --seed 0
```

Rebuilding with the same seed and the same `instances_val2017.json` yields the
same sample ids, the same pair assignment and the same splits. Provenance for
the saved results is in `results/baseline/provenance.json`.
