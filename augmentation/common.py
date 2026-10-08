"""Shared plumbing for the Phase 3 robustness transformations.

Every robustness experiment has the same shape: take one benchmark sample, change
only its target image in a controlled, severity-graded way, keep the reference
untouched, and keep the ground truth correct for what the transformed image
actually shows. This module holds the pieces all four families share:

* `Variant` - one named transform at one severity, with its parameters.
* `derive_seed` - a stable per-variant seed, so a rebuild reproduces the same
  pixels on any machine and in any order.
* image/mask IO with fixed JPEG settings, so codec noise is identical for every
  variant (and the level-0 control measures what that codec noise alone costs).
* `write_metadata` / `write_manifest` - the two files that make a transformed
  sample interpretable months later.

Nothing here touches the segmentation pipeline; these are dataset-side utilities
only.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

FAMILIES: Tuple[str, ...] = (
    "control",
    "illumination",
    "scale",
    "occlusion",
    "viewpoint",
    "combined",
)

# The control is a re-encode, not a byte copy: every variant on disk went through
# this JPEG setting, so level 0 goes through it too. That way "severity 0" costs
# exactly the codec, and nothing else.
JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0  # 4:4:4, the least destructive 8-bit JPEG option

METADATA_NAME = "metadata.json"
MANIFEST_NAME = "manifest.json"
GENERATOR = "tools/build_robustness.py"
TRANSFORM_VERSION = 1


@dataclass(frozen=True)
class Variant:
    """One controlled transformation at one severity.

    Attributes:
        family: which experiment family this belongs to (see `FAMILIES`).
        condition: the label used for grouping and for the `condition` CSV
            column, e.g. `brightness_up` or `smaller`.
        severity: ordinal level inside the family. 0 is always "unchanged".
        variant: unique name inside the family, e.g. `brightness_up_2`. It is
            also the directory name, so it must stay filesystem-safe.
        transformation: the primitive actually applied (`brightness`,
            `crop_zoom`, `occlude`, ...), kept separate from `condition` so one
            primitive can appear under several conditions.
        parameters: everything needed to reproduce the numbers inside the
            transform; written verbatim into the variant's metadata.
    """

    family: str
    condition: str
    severity: int
    variant: str
    transformation: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "condition": self.condition,
            "severity": int(self.severity),
            "variant": self.variant,
            "transformation": self.transformation,
            "parameters": dict(self.parameters),
        }


def derive_seed(base_seed: int, *parts: Any) -> int:
    """A stable 32-bit seed for one variant.

    `random.Random(hash(...))` is not enough on its own: Python's `hash` is
    salted per process, so a rebuild would silently produce different pixels.
    A SHA-256 of the parts is stable across machines, runs and Python versions.
    """
    payload = ":".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def load_rgb(path: str):
    """Read an image as an RGB `PIL.Image`."""
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def load_mask(path: str) -> np.ndarray:
    """Read a mask PNG as a boolean array; anything above 127 is foreground."""
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    return array > 127


def save_image(path: str, image) -> str:
    """Write an RGB image with the family-wide JPEG settings."""
    from PIL import Image

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    frame = image if not isinstance(image, np.ndarray) else Image.fromarray(
        np.clip(np.asarray(image), 0, 255).astype(np.uint8), mode="RGB"
    )
    frame.convert("RGB").save(path, format="JPEG", quality=JPEG_QUALITY,
                              subsampling=JPEG_SUBSAMPLING)
    return path


def save_mask(path: str, mask: np.ndarray) -> str:
    """Write a boolean mask as an 8-bit PNG (255 foreground, 0 background)."""
    from PIL import Image

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8),
                    mode="L").save(path, format="PNG")
    return path


def variant_id(sample_id: str, variant: Variant) -> str:
    """Directory name of one transformed sample: `<sample>__<variant>`."""
    if variant.family == "control":
        return sample_id
    return f"{sample_id}__{variant.variant}"


def record_id(sample_id: str, variant: Variant) -> str:
    """Globally unique id of one experiment row, used as the result record name."""
    if variant.family == "control":
        return f"control__{sample_id}"
    return f"{variant.family}__{sample_id}__{variant.variant}"


def sample_directory(root: str, sample_id: str) -> str:
    return os.path.join(root, "samples", sample_id)


def sample_paths(root: str, sample_id: str) -> Dict[str, str]:
    """The four files of one benchmark sample, by role."""
    directory = sample_directory(root, sample_id)
    return {
        "reference_image": os.path.join(directory, "reference.jpg"),
        "reference_mask": os.path.join(directory, "reference_mask.png"),
        "target_image": os.path.join(directory, "target.jpg"),
        "target_mask": os.path.join(directory, "target_mask.png"),
        "metadata": os.path.join(directory, "metadata.json"),
    }


def read_sample_metadata(root: str, sample_id: str) -> Dict[str, Any]:
    with open(sample_paths(root, sample_id)["metadata"], "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_metadata(directory: str, *, original_sample_id: str, transformation: str,
                   severity: int, random_seed: int, variant: Variant,
                   parameters: Optional[Dict[str, Any]] = None,
                   extra: Optional[Dict[str, Any]] = None) -> str:
    """Write the per-variant metadata file.

    The first four fields are the contract asked for by the Phase 3 protocol;
    everything else is context that makes the numbers reusable without
    re-reading this source file.
    """
    os.makedirs(directory, exist_ok=True)
    payload: Dict[str, Any] = {
        "original_sample_id": original_sample_id,
        "transformation": transformation,
        "severity": int(severity),
        "random_seed": int(random_seed),
        "generator": GENERATOR,
        "transform_version": TRANSFORM_VERSION,
        **variant.describe(),
        "parameters": dict(parameters if parameters is not None else variant.parameters),
    }
    if extra:
        payload.update(extra)
    path = os.path.join(directory, METADATA_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def relative(path: str, base: str) -> str:
    """POSIX-style path of `path` relative to `base`, for manifest portability."""
    return os.path.relpath(path, base).replace(os.sep, "/")


def write_manifest(directory: str, family: str, cases: Sequence[Dict[str, Any]],
                   seed: int, notes: str) -> str:
    """Write one family manifest in the schema `evaluation.cases` already reads.

    Paths are relative to the manifest's own directory, so
    `cases_from_manifest("data/processed/illumination/manifest.json")` works from
    any working directory - the same contract the Phase 2 split manifests use.
    """
    os.makedirs(directory, exist_ok=True)
    payload = {
        "dataset": "coco2017-val",
        "family": family,
        "seed": int(seed),
        "generator": GENERATOR,
        "transform_version": TRANSFORM_VERSION,
        "protocol": notes,
        "cases": list(cases),
    }
    path = os.path.join(directory, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def manifest_path(root: str, family: str) -> str:
    return os.path.join(root, "processed", family, MANIFEST_NAME)


def family_directory(root: str, family: str) -> str:
    return os.path.join(root, "processed", family)


def mask_area(mask: np.ndarray) -> int:
    return int(np.asarray(mask, dtype=bool).sum())


def bbox_of(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """`(x0, y0, x1, y1)` inclusive-exclusive bounding box of a boolean mask."""
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def seed_for(base_seed: int, sample_id: str, family: str, variant_name: str) -> int:
    return derive_seed(base_seed, sample_id, family, variant_name)
