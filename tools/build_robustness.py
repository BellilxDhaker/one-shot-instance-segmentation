"""Build the Phase 3 robustness dataset: controlled variants of every sample.

Phase 2 measured the pipeline on real re-photographs; Phase 3 needs ladders that
change exactly one thing at a time - illumination, scale, viewpoint, occlusion,
and a few pairs of them - with severities that are numbers, not adjectives. This
tool turns each benchmark sample into one directory per variant:

    python tools/build_robustness.py                     # all samples, all families
    python tools/build_robustness.py --limit 2 --force   # smoke rebuild

Outputs, relative to `--root` (default `data`):

    processed/control/<sample_id>/target.jpg            identity re-encode (level 0)
    processed/illumination/<sample_id>__<variant>/...   19 photometric variants
    processed/scale/<sample_id>__<variant>/...          6 crop-zoom variants
    processed/occlusion/<sample_id>__<variant>/...      5 texture occlusions
    processed/viewpoint/<sample_id>__<variant>/...      3 rotations + mirror
    processed/combined/<sample_id>__<variant>/...       8 two-failure combos
    processed/<family>/manifest.json                    loadable by `cases_from_manifest`

(`data/processed/selection.json` - which samples the runner uses, and why - is
written by `tools/run_robustness.py`, because it needs the baseline results.)

Each variant directory holds `target.jpg` (JPEG quality 95, 4:4:4 - the same
codec settings as the control, so codec noise is identical everywhere),
`target_mask.png` (the ground truth for what the transformed image shows),
`metadata.json` (`original_sample_id`, `transformation`, `severity`,
`random_seed`, plus everything the transform actually did: achieved scale
factors, achieved occlusion shares, rotation angles, padding), and for
occlusions the `occluder_mask.png` that visible-region scoring needs.

Reference images are *never* copied: manifests point back at
`../../samples/<id>/reference.jpg`, so one reference serves 43 variants.

Rebuilding is deterministic - a variant's pixels depend only on `--seed`, the
sample id and the variant name - and resumable: a directory with both
`target.jpg` and `metadata.json` is skipped unless `--force` is given. Deleting
a directory is how you ask for one variant to be redone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmentation import registry  # noqa: E402
from augmentation.common import (  # noqa: E402
    FAMILIES,
    family_directory,
    load_mask,
    load_rgb,
    manifest_path,
    read_sample_metadata,
    record_id,
    relative,
    sample_paths,
    save_image,
    save_mask,
    variant_id,
    write_manifest,
    write_metadata,
)

SELECTION_FILE = "selection.json"
SELECTION_RATIONALE = (
    "samples ranked by baseline IoU in results/baseline/samples, then picked at "
    "evenly spaced ranks so retention is measured across the whole performance "
    "range instead of only on the easy end"
)


def list_sample_ids(root: str) -> List[str]:
    """Every built benchmark sample, in id order."""
    samples_root = os.path.join(root, "samples")
    if not os.path.isdir(samples_root):
        raise FileNotFoundError(f"no samples under {samples_root}; run tools/build_benchmark.py first")
    return sorted(entry for entry in os.listdir(samples_root)
                  if os.path.isdir(os.path.join(samples_root, entry)))


def _complete(directory: str) -> bool:
    return (os.path.exists(os.path.join(directory, "target.jpg"))
            and os.path.exists(os.path.join(directory, "metadata.json")))


def _case_entry(*, name: str, reference_paths: Dict[str, str], target_paths: Dict[str, str],
                sample_id: str, variant, seed: int, sample_meta: Dict[str, Any],
                info: Dict[str, Any], manifest_dir: str) -> Dict[str, Any]:
    """One manifest row: paths relative to the manifest plus flat metadata."""
    entry: Dict[str, Any] = {
        "name": name,
        "reference_image": relative(reference_paths["reference_image"], manifest_dir),
        "reference_mask": relative(reference_paths["reference_mask"], manifest_dir),
        "target_image": relative(target_paths["target_image"], manifest_dir),
        "target_mask": relative(target_paths["target_mask"], manifest_dir),
        "original_sample_id": sample_id,
        "family": variant.family,
        "condition": variant.condition,
        "severity": int(variant.severity),
        "variant": variant.variant,
        "transformation": variant.transformation,
        "random_seed": int(seed),
        "category": sample_meta.get("category"),
        "object_id": sample_meta.get("object_id"),
        "dataset": sample_meta.get("dataset", "coco2017-val"),
    }
    for key in ("achieved_factor", "achieved_fraction", "target_fraction",
                "occluder_type", "visible_object_fraction", "yaw_deg"):
        if key in info:
            entry[key] = info[key]
    return entry


def build_sample(root: str, sample_id: str, seed: int, families: Sequence[str],
                 force: bool, dry_run: bool) -> Tuple[Dict[str, int], Dict[str, List[Dict[str, Any]]]]:
    """Build every variant of one sample; returns counts and manifest cases."""
    paths = sample_paths(root, sample_id)
    for key in ("reference_image", "reference_mask", "target_image", "target_mask"):
        if not os.path.exists(paths[key]):
            raise FileNotFoundError(f"{sample_id} is missing {paths[key]}")
    sample_meta = read_sample_metadata(root, sample_id)
    image = np.asarray(load_rgb(paths["target_image"]))
    mask = load_mask(paths["target_mask"])
    index = registry.sample_index(sample_id)

    counts: Dict[str, int] = {family: 0 for family in families}
    cases: Dict[str, List[Dict[str, Any]]] = {family: [] for family in families}
    for variant in registry.variants_for_sample(sample_id):
        if variant.family not in cases:
            continue
        directory = os.path.join(family_directory(root, variant.family),
                                 variant_id(sample_id, variant))
        name = record_id(sample_id, variant)
        variant_seed = registry.variant_seed(seed, sample_id, variant)
        reusable = _complete(directory)

        if reusable and not force:
            # A finished variant still belongs in the manifest, so a resumed
            # build must be able to describe it without redoing it.
            with open(os.path.join(directory, "metadata.json"), "r", encoding="utf-8") as handle:
                info = json.load(handle).get("parameters") or {}
            counts[variant.family] += 1
        elif dry_run:
            print(f"  would build {name}")
            counts[variant.family] += 1
            continue
        else:
            rng = np.random.default_rng(variant_seed)
            transformed, transformed_mask, info = registry.apply_variant(
                variant.family, image, mask, variant, rng)
            artifacts = info.pop("artifacts", {})
            os.makedirs(directory, exist_ok=True)
            save_image(os.path.join(directory, "target.jpg"), transformed)
            save_mask(os.path.join(directory, "target_mask.png"), transformed_mask)
            for artifact_name, artifact in artifacts.items():
                save_mask(os.path.join(directory, f"{artifact_name}.png"), artifact)
            write_metadata(
                directory,
                original_sample_id=sample_id,
                transformation=variant.transformation,
                severity=variant.severity,
                random_seed=variant_seed,
                variant=variant,
                parameters=info if info else None,
                extra={"category": sample_meta.get("category"),
                       "object_id": sample_meta.get("object_id"),
                       "dataset": sample_meta.get("dataset", "coco2017-val")},
            )
            counts[variant.family] += 1

        target_paths = {
            "target_image": os.path.join(directory, "target.jpg"),
            "target_mask": os.path.join(directory, "target_mask.png"),
        }
        cases[variant.family].append(_case_entry(
            name=name,
            reference_paths=paths,
            target_paths=target_paths,
            sample_id=sample_id,
            variant=variant,
            seed=variant_seed,
            sample_meta=sample_meta,
            info=info,
            manifest_dir=family_directory(root, variant.family),
        ))
    return counts, cases


def build(root: str, seed: int, families: Sequence[str], limit: Optional[int],
          sample_ids: Optional[Sequence[str]], force: bool, dry_run: bool) -> Dict[str, Any]:
    """Build manifests and variants for the requested samples."""
    started = time.time()
    ids = list(sample_ids) if sample_ids else list_sample_ids(root)
    if limit is not None:
        ids = ids[:limit]
    if not ids:
        raise SystemExit("no samples selected")

    manifest_cases: Dict[str, List[Dict[str, Any]]] = {family: [] for family in FAMILIES}
    totals = {family: 0 for family in FAMILIES}
    for sample_id in ids:
        counts, cases = build_sample(root, sample_id, seed, families, force, dry_run)
        for family in families:
            totals[family] += counts.get(family, 0)
            manifest_cases[family].extend(cases.get(family, []))

    notes = {
        "control": "level 0: the sample re-encoded with the family-wide JPEG "
                   "settings, so severity 0 measures codec cost alone",
        "illumination": "brightness/contrast/temperature/gamma ladders; mask unchanged",
        "scale": "object-centric crop-zoom; achieved factor recorded per variant",
        "occlusion": "texture-filled occluders hiding 10-50% of the object; "
                     "ground truth unchanged, occluder_mask.png stored for "
                     "visible-region scoring",
        "viewpoint": "synthetic rotation homography (pivot re-centred) plus mirror; "
                     "planar approximation, yaw sign and angles recorded",
        "combined": "two representative failures composed in the documented order "
                    "viewpoint -> scale -> illumination -> occlusion",
    }
    written: Dict[str, Any] = {}
    if not dry_run:
        for family in families:
            if manifest_cases[family]:
                path = write_manifest(
                    family_directory(root, family), family, manifest_cases[family],
                    seed, notes[family])
                written[family] = path
        print(f"processed {len(ids)} samples, {sum(totals.values())} variants "
              f"in {time.time() - started:.1f}s")
        for family in families:
            print(f"  {family:12s} {totals[family]:5d} variants "
                  f"-> {manifest_path(root, family)}")
    return {"samples": len(ids), "variants": totals, "manifests": written}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data", help="dataset root (default: data)")
    parser.add_argument("--seed", type=int, default=7, help="base seed for variant RNGs")
    parser.add_argument("--families", nargs="+", default=list(FAMILIES),
                        choices=list(FAMILIES), help="families to build")
    parser.add_argument("--limit", type=int, default=None, help="only the first N samples")
    parser.add_argument("--samples", nargs="+", default=None,
                        help="explicit sample ids (default: every sample on disk)")
    parser.add_argument("--force", action="store_true", help="rebuild existing variants")
    parser.add_argument("--dry-run", action="store_true", help="list work, write nothing")
    arguments = parser.parse_args(argv)
    build(root=arguments.root, seed=arguments.seed, families=arguments.families,
          limit=arguments.limit, sample_ids=arguments.samples,
          force=arguments.force, dry_run=arguments.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
