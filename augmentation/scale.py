"""Scale robustness: change the object's relative size inside the target frame.

Resizing the whole target image would not answer the question - the encoder
rescales its input anyway, so a uniformly resized photograph is nearly free.
What matters is the object's size *relative to its surroundings*, so this family
uses an object-centric crop-and-resize:

    zoom  s = sqrt(requested_area_factor)
    crop a window of size (W/s, H/s) centred on the object bounding box
    pad the window back out to the full frame when it reaches outside the image
    resize the window to (W, H) - image bicubic, mask nearest neighbour

Because the window is centred on the object, the object keeps roughly the same
position while its share of the frame changes by the requested factor. The mask
receives the identical geometric operation, so the ground truth stays correct
for what the transformed image shows.

Two things this deliberately does **not** claim:

* it is not a camera dolly. Background outside the window is discarded when
  zooming in, and edge/reflect-padded when zooming out. Both are synthetic
  artifacts, and `parameters.padded` says which case occurred.
* the factor may be clamped. A large object has no room to grow without being
  cut off, and cutting the object would change the ground truth. The zoom is
  capped by the bounding box, and the *achieved* area ratio is measured from the
  transformed mask and stored in `parameters.achieved_factor`, so no result ever
  reports a factor the pixels did not deliver.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from augmentation.common import Variant, bbox_of

FAMILY = "scale"

# (condition, severity, requested area factor relative to the original frame)
SMALLER = (("mild", 1, 0.85), ("moderate", 2, 0.70), ("strong", 3, 0.50))
LARGER = (("mild", 1, 1.18), ("moderate", 2, 1.43), ("strong", 3, 2.00))
MARGIN_PIXELS = 4  # breathing room around the bounding box when clamping the zoom


def build_variants() -> List[Variant]:
    variants: List[Variant] = []
    for direction, ladder in (("smaller", SMALLER), ("larger", LARGER)):
        for label, severity, factor in ladder:
            variants.append(Variant(
                family=FAMILY,
                condition=direction,
                severity=severity,
                variant=f"{direction}_{severity}",
                transformation="crop_zoom",
                parameters={"requested_factor": factor, "level": label},
            ))
    return variants


def _pad_array(array: np.ndarray, left: int, top: int, right: int, bottom: int,
               mode: str) -> np.ndarray:
    """Pad `array` on the four sides; falls back to edge replication when the
    requested reflection is wider than the array itself."""
    if max(left, right) <= 0 and max(top, bottom) <= 0:
        return array
    if mode == "reflect":
        height, width = array.shape[:2]
        if left >= width or right >= width or top >= height or bottom >= height:
            mode = "edge"
    pad_width = ((top, bottom), (left, right)) + ((0, 0),) * (array.ndim - 2)
    return np.pad(array, pad_width, mode=mode)


def crop_zoom(image: np.ndarray, mask: np.ndarray, requested_factor: float,
              ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Object-centric crop-and-resize; returns image, mask and what really happened."""
    from PIL import Image

    height, width = mask.shape
    box = bbox_of(mask)
    if box is None:
        raise ValueError("crop_zoom needs a non-empty ground-truth mask")
    bx0, by0, bx1, by1 = box
    box_width, box_height = bx1 - bx0, by1 - by0
    centre_x, centre_y = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0

    zoom = math.sqrt(float(requested_factor))
    clamped = False
    if zoom > 1.0:
        # A window smaller than the bounding box would cut the object, so the
        # zoom is capped before anything is cropped.
        room_x = width / float(box_width + MARGIN_PIXELS)
        room_y = height / float(box_height + MARGIN_PIXELS)
        cap = min(room_x, room_y)
        if zoom > cap:
            zoom = cap
            clamped = True

    window_w = int(round(width / zoom))
    window_h = int(round(height / zoom))
    window_w = max(window_w, 2)
    window_h = max(window_h, 2)

    if window_w <= width and window_h <= height:
        # Fully inside the frame: keep the window inside, but never leave the object.
        lower_x = max(bx1 - window_w, 0)
        upper_x = min(bx0, width - window_w)
        lower_y = max(by1 - window_h, 0)
        upper_y = min(by0, height - window_h)
        x0 = float(np.clip(centre_x - window_w / 2.0, lower_x, upper_x)) if lower_x <= upper_x else 0.0
        y0 = float(np.clip(centre_y - window_h / 2.0, lower_y, upper_y)) if lower_y <= upper_y else 0.0
        x0, y0 = int(round(x0)), int(round(y0))
        x1, y1 = x0 + window_w, y0 + window_h
        padded = False
        pad_mode = "none"
    else:
        # Zooming out: the window reaches past the frame, so the missing context
        # is synthesised (reflected where possible, replicated at the edges).
        x0 = int(round(centre_x - window_w / 2.0))
        y0 = int(round(centre_y - window_h / 2.0))
        x1, y1 = x0 + window_w, y0 + window_h
        padded = True
        pad_mode = "reflect"

    left, top = max(0, -x0), max(0, -y0)
    right, bottom = max(0, x1 - width), max(0, y1 - height)
    image_padded = _pad_array(image, left, top, right, bottom, pad_mode)
    # The mask is padded with background, never with reflections: a mirrored
    # copy of the object in the padding would silently grow the ground truth.
    mask_padded = _pad_array((mask.astype(np.uint8) * 255), left, top, right, bottom,
                             "constant")

    crop = (x0 + left, y0 + top, x0 + left + window_w, y0 + top + window_h)
    window_rgb = image_padded[crop[1]:crop[3], crop[0]:crop[2]]
    window_mask = mask_padded[crop[1]:crop[3], crop[0]:crop[2]] > 127

    resized_image = Image.fromarray(window_rgb, mode="RGB").resize(
        (width, height), Image.BICUBIC)
    resized_mask = Image.fromarray((window_mask * 255).astype(np.uint8), mode="L").resize(
        (width, height), Image.NEAREST)

    new_image = np.asarray(resized_image, dtype=np.uint8)
    new_mask = np.asarray(resized_mask, dtype=np.uint8) > 127

    before = float(mask.mean())
    after = float(new_mask.mean())
    achieved = (after / before) if before > 0 else None
    info = {
        "requested_factor": round(float(requested_factor), 4),
        "achieved_factor": round(float(achieved), 4) if achieved else None,
        "zoom": round(float(zoom), 4),
        "clamped": bool(clamped),
        "padded": bool(padded),
        "pad_mode": pad_mode,
        "window": [int(v) for v in (x0, y0, x1, y1)],
        "object_area_fraction_before": round(before, 6),
        "object_area_fraction_after": round(after, 6),
    }
    return new_image, new_mask, info


def apply(array: np.ndarray, mask: np.ndarray, variant: Variant,
          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply one scale variant. `rng` is unused: the crop is a pure function."""
    del rng
    factor = float(variant.parameters["requested_factor"])
    return crop_zoom(array, mask, factor)
