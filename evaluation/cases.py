"""Evaluation cases: the bridge between a dataset on disk and the pipeline.

A few-shot case is a reference image plus its mask, a target image, and
optionally a target mask to score against. This module stays deliberately
format-agnostic: it does not know how COCO-20i, FSS-1000, LVIS-92i, PACO-Part or
Pascal-Part encode their splits, because those come from external archives whose
internal layout is not ours to assume. Instead, a case is described by paths, and
`cases_from_manifest` reads that description from JSON:

    {
      "root": "datasets",
      "cases": [
        {"name": "coco-20i-val-0001",
         "reference_image": "COCO2014/val2014/COCO_val2014_000000391895.jpg",
         "reference_mask":  "COCO2014/annotations/val2014/000000391895.png",
         "target_image":    "COCO2014/val2014/COCO_val2014_000000000139.jpg",
         "target_mask":     "COCO2014/annotations/val2014/000000000139.png"}
      ]
    }

Multiple instances in one mask file are supported through `instance_index`, which
keeps only the requested label, so a converted COCO-20i mask file and a
single-object Pascal-Part mask both work unchanged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from evaluation.metrics import EvaluationReport, evaluate_results
from matching.reference_representation import MaskInput
from models.feature_encoder import ImageInput, to_pil_image


@dataclass
class Case:
    """One few-shot example.

    Attributes:
        name: identifier used in the report table.
        reference_image: image containing the query object.
        reference_mask: binary mask of that object.
        target_image: image to segment.
        target_mask: optional ground truth for scoring.
        meta: anything else worth keeping (category, dataset, split).
    """

    name: str
    reference_image: ImageInput
    reference_mask: MaskInput
    target_image: ImageInput
    target_mask: Optional[MaskInput] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def scorable(self) -> bool:
        return self.target_mask is not None

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "scorable": self.scorable, **self.meta}


def load_mask(path: str, instance_index: Optional[int] = None) -> np.ndarray:
    """Read a mask image; `instance_index` keeps only that instance's label.

    White foreground over black background is assumed. A multi-colour palette
    file is handled by treating the top-left pixel as the background and every
    distinct remaining colour as one instance.
    """
    from PIL import Image

    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        background = array[0, 0].astype(np.int16)
        mask = np.abs(array.astype(np.int16) - background).sum(axis=-1) > 24
        if instance_index is not None:
            colours = np.unique(array[mask]) if mask.any() else np.zeros((0, array.shape[-1]), array.dtype)
            if instance_index >= len(colours):
                raise IndexError(
                    f"{path} holds {len(colours)} instance labels, cannot take index {instance_index}"
                )
            mask = (array == colours[instance_index]).all(axis=-1)
    else:
        if instance_index is not None:
            raise ValueError(f"{path} is a single-instance grayscale mask; drop instance_index")
        mask = array > 127
    return mask.astype(bool)


def case_from_paths(
    root: str,
    name: str,
    reference_image: str,
    reference_mask: str,
    target_image: str,
    target_mask: Optional[str] = None,
    instance_index: Optional[int] = None,
    target_instance_index: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Case:
    """Build a `Case` from paths relative to `root`."""
    def resolve(relative: Optional[str]) -> Optional[str]:
        if relative is None:
            return None
        path = relative if os.path.isabs(relative) else os.path.join(root, relative)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path

    reference_path = resolve(reference_image)
    target_path = resolve(target_image)
    assert reference_path is not None and target_path is not None
    return Case(
        name=name,
        reference_image=to_pil_image(reference_path),
        reference_mask=load_mask(resolve(reference_mask) or "", instance_index),
        target_image=to_pil_image(target_path),
        target_mask=None if target_mask is None else load_mask(resolve(target_mask) or "", target_instance_index),
        meta=dict(meta or {}),
    )


def cases_from_manifest(path: str, root: Optional[str] = None) -> List[Case]:
    """Read a JSON manifest describing several cases."""
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    base = root or manifest.get("root") or os.path.dirname(os.path.abspath(path))
    cases: List[Case] = []
    for entry in manifest.get("cases", []):
        meta = {key: value for key, value in entry.items() if key not in {
            "name", "reference_image", "reference_mask", "target_image", "target_mask",
            "instance_index", "target_instance_index"}}
        cases.append(
            case_from_paths(
                base,
                entry["name"],
                entry["reference_image"],
                entry["reference_mask"],
                entry["target_image"],
                entry.get("target_mask"),
                entry.get("instance_index"),
                entry.get("target_instance_index"),
                meta,
            )
        )
    return cases


def run_cases(pipeline: Any, cases: Sequence[Case], verbose: bool = True) -> EvaluationReport:
    """Run every case through the pipeline and score the ones with a ground truth."""
    from evaluation.metrics import evaluate_mask

    rows: List[Tuple[str, Optional[np.ndarray], Optional[np.ndarray], Any]] = []
    for case in cases:
        result = pipeline.segment_object(case.reference_image, case.reference_mask, case.target_image)
        rows.append((case.name, result.predicted_mask, case.target_mask, result))
        if verbose:
            suffix = "" if not case.scorable else f" IoU {evaluate_mask(result.predicted_mask, case.target_mask).iou:.3f}"
            print(f"{case.name:32s} conf {result.confidence:.3f} "
                  f"{result.runtime['total']:.2f}s{suffix}")
    scorable = [row for row in rows if row[2] is not None]
    if not scorable:
        return EvaluationReport(
            rows=[{"example": name, "confidence": float(result.confidence),
                   "seconds": float(result.runtime["total"])}
                  for name, _, _, result in rows],
            metric_names=("confidence",),
        )
    return evaluate_results(scorable)
