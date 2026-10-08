"""Viewpoint robustness: reproject the target through a synthetic camera rotation.

There is no multi-view ground truth anywhere in COCO 2017 val - it has no video,
no burst and no identity links - so a "real" viewpoint ladder cannot be built
from the official data. This family therefore does the honest synthetic
approximation: a homography induced by rotating the camera about its optical
centre, with the scene shifted so that the *object* sits at the principal point
while the rotation is applied, and shifted back afterwards:

    H = T(p - H_rot(c)) . K R K^-1 . T(c - p),   R = Rz(roll) . Rx(pitch) . Ry(yaw)

where `p` is the object's bbox centre and `c` the frame centre. The two
translations matter as much as the rotation:

* without the re-centring pair, the object rotates wherever it happens to lie
  in the frame - and a rotation homography's local scale is
  `f^2 / (f cos(yaw) - x sin(yaw))^2`, which runs towards infinity as `x`
  approaches the vanishing line. An object near the frame edge at yaw -40deg
  inflated 6x in area in early builds. Shifting it to `c` before rotating pins
  the scale field at its most benign point (floor `1/cos^2(yaw)`), making the
  two yaw directions symmetric;
* together they keep the pivot a fixed point, so the subject stays framed at
  every severity.

A pure rotation has no parallax, so every surface keeps its projective
appearance; what changes is perspective, foreshortening and the framing of the
object inside the frame. Objects behave as if painted on a plane - that is a
*weaker and slightly different* change than a real viewpoint (explicitly not
claimed otherwise anywhere in the results), but it is reproducible,
label-exact, and it isolates the pipeline's sensitivity to geometry rather than
to occlusion by newly exposed/disappeared surfaces.

Ladder (yaw sign is drawn per sample and recorded):

| variant     | condition | severity | yaw   | pitch | roll |
| ----------- | --------- | -------- | ----- | ----- | ---- |
| `yaw_1`     | `yaw`     | 1        | 12deg | 0deg  | 0deg |
| `yaw_2`     | `yaw`     | 2        | 25deg | 4deg  | 3deg |
| `yaw_3`     | `yaw`     | 3        | 40deg | 6deg  | 6deg |
| `mirror`    | `mirror`  | 1        | -     | -     | -    |

`mirror` (a horizontal flip) is a separate condition rather than a fourth yaw
level: it is an orientation change, not a continuous viewpoint step, and the
analysis must be able to keep it apart.

Implementation notes:

* the output canvas keeps its size (W, H); the rotation's out-of-frame
  excursions are served by reflect-padding the input first, with the padding
  computed exactly from the inverse-mapped canvas corners (capped at 1.25x the
  frame, recorded in `parameters.padding_px`). If the cap ever binds,
  `parameters.padding_capped` says so and PIL falls back to `fillcolor`.
* image resamples bicubic, mask nearest neighbour, so ground truth stays
  quantised.
* coefficients passed to `PIL.Image.transform` map output pixels *back* to
  input pixels, which is why the matrix inverted here is `H`, not `H^-1`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from augmentation.common import Variant, bbox_of

FAMILY = "viewpoint"

# (variant, severity, yaw, pitch, roll) in degrees
YAW_LADDER: Tuple[Tuple[str, int, float, float, float], ...] = (
    ("yaw_1", 1, 12.0, 0.0, 0.0),
    ("yaw_2", 2, 25.0, 4.0, 3.0),
    ("yaw_3", 3, 40.0, 6.0, 6.0),
)
FRACTION_OF_FRAME = 0.45   # focal length as a fraction of max(W, H)
PADDING_CAP = 1.25         # hard cap on padding as a fraction of max(W, H)
PADDING_MARGIN_PX = 3


def build_variants() -> List[Variant]:
    variants: List[Variant] = []
    for name, severity, yaw, pitch, roll in YAW_LADDER:
        variants.append(Variant(
            family=FAMILY,
            condition="yaw",
            severity=severity,
            variant=name,
            transformation="rotation",
            parameters={"yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll},
        ))
    variants.append(Variant(FAMILY, "mirror", 1, "mirror", "mirror", {}))
    return variants


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Camera rotation as `Rz(roll) . Rx(pitch) . Ry(yaw)` (radians)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return rz @ rx @ ry


def _output_to_input(image: np.ndarray, yaw: float, pitch: float, roll: float,
                     pivot: Tuple[float, float],
                     ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """3x3 matrix mapping output pixels back to original input pixels, plus the
    two translations that were composed into it (recorded in metadata).

    Construction: `G = T(p - M(c)) . M . T(c - p)` with `M = K R K^-1`, `p` the
    object's bbox centre, `c` the frame centre; the matrix returned is `G^-1`.

    Rotating at the frame centre keeps the local scale field at its benign end
    (floor `1/cos^2(yaw)`, symmetric in the yaw sign); the second translation
    restores the object to where it started, so the pivot is fixed by
    construction. See the module docstring for why both shifts are needed.
    """
    height, width = image.shape[:2]
    focal = FRACTION_OF_FRAME * max(width, height)
    rotation = _rotation_matrix(yaw, pitch, roll)
    intrinsic = np.array([[focal, 0.0, width / 2.0],
                          [0.0, focal, height / 2.0],
                          [0.0, 0.0, 1.0]])
    forward = intrinsic @ rotation @ np.linalg.inv(intrinsic)
    centre = np.array([width / 2.0, height / 2.0, 1.0])
    # The pre-shift places the object at the frame centre, so the rotation acts
    # on it there; `moved` is where the rotation then sends it.
    moved = forward @ centre
    moved = moved[:2] / moved[2]
    post_shift = (pivot[0] - moved[0], pivot[1] - moved[1])
    pre_to_centre = (centre[0] - pivot[0], centre[1] - pivot[1])
    pre = np.array([[1.0, 0.0, pre_to_centre[0]],
                    [0.0, 1.0, pre_to_centre[1]],
                    [0.0, 0.0, 1.0]])
    post = np.array([[1.0, 0.0, post_shift[0]],
                     [0.0, 1.0, post_shift[1]],
                     [0.0, 0.0, 1.0]])
    in_to_out = post @ forward @ pre
    details = {
        "focal_px": round(float(focal), 2),
        "pre_shift_px": [round(float(pre_to_centre[0]), 2),
                         round(float(pre_to_centre[1]), 2)],
        "post_shift_px": [round(float(post_shift[0]), 2),
                          round(float(post_shift[1]), 2)],
    }
    return np.linalg.inv(in_to_out), focal, details


def _padding_for(matrix: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    """Uncapped padding (left, top, right, bottom) so every output corner samples
    real pixels. The caller applies `PADDING_CAP` and records whether it bound."""
    corners = np.array([[0.0, 0.0, 1.0],
                        [width, 0.0, 1.0],
                        [0.0, height, 1.0],
                        [width, height, 1.0]]).T
    mapped = matrix @ corners
    mapped = mapped[:2] / np.where(mapped[2] == 0.0, 1e-12, mapped[2])
    left = int(max(0, np.ceil(-mapped[0].min() + PADDING_MARGIN_PX)))
    top = int(max(0, np.ceil(-mapped[1].min() + PADDING_MARGIN_PX)))
    right = int(max(0, np.ceil(mapped[0].max() - width + PADDING_MARGIN_PX)))
    bottom = int(max(0, np.ceil(mapped[1].max() - height + PADDING_MARGIN_PX)))
    return left, top, right, bottom


def _pad(array: np.ndarray, left: int, top: int, right: int, bottom: int,
         mode: str) -> np.ndarray:
    """Pad on the four sides; numpy's `reflect` repeats past the edge, which
    handles pads wider than the array (needed for the pre-shift translation)."""
    if max(left, right, top, bottom) <= 0:
        return array
    pad_width = ((top, bottom), (left, right)) + ((0, 0),) * (array.ndim - 2)
    return np.pad(array, pad_width, mode=mode)


def rotate(image: np.ndarray, mask: np.ndarray, yaw_deg: float, pitch_deg: float,
           roll_deg: float, rng: np.random.Generator,
           ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Reproject image and mask through a camera rotation of the given angles."""
    from PIL import Image

    height, width = mask.shape
    box = bbox_of(mask)
    if box is None:
        raise ValueError("rotate needs a non-empty ground-truth mask")
    pivot = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    yaw_sign = 1 if rng.random() < 0.5 else -1
    yaw = np.deg2rad(yaw_deg * yaw_sign)
    pitch = np.deg2rad(pitch_deg * yaw_sign)
    roll = np.deg2rad(roll_deg * yaw_sign)

    matrix, _focal, transform_details = _output_to_input(
        image, float(yaw), float(pitch), float(roll), pivot)
    left, top, right, bottom = _padding_for(matrix, width, height)
    cap = int(round(PADDING_CAP * max(width, height)))
    padding_capped = any(side > cap for side in (left, top, right, bottom))
    left, top = min(left, cap), min(top, cap)
    right, bottom = min(right, cap), min(bottom, cap)
    pad_mode = "reflect"

    image_padded = _pad(image, left, top, right, bottom, pad_mode)
    mask_padded = _pad((mask.astype(np.uint8) * 255), left, top, right, bottom,
                       "constant")
    offset = np.array([[1.0, 0.0, float(left)],
                       [0.0, 1.0, float(top)],
                       [0.0, 0.0, 1.0]])
    mapper = offset @ matrix
    mapper = mapper / mapper[2, 2]
    coefficients = [float(v) for v in
                    (mapper[0, 0], mapper[0, 1], mapper[0, 2],
                     mapper[1, 0], mapper[1, 1], mapper[1, 2],
                     mapper[2, 0], mapper[2, 1])]

    warped_image = Image.fromarray(image_padded, mode="RGB").transform(
        (width, height), Image.PERSPECTIVE, coefficients,
        resample=Image.BICUBIC, fillcolor=(0, 0, 0))
    warped_mask = Image.fromarray(mask_padded, mode="L").transform(
        (width, height), Image.PERSPECTIVE, coefficients,
        resample=Image.NEAREST, fillcolor=0)

    ground_truth = np.asarray(warped_mask, dtype=np.uint8) > 127
    if not ground_truth.any():
        raise ValueError(
            "rotation pushed the whole object out of the frame "
            f"(yaw={yaw_deg}, pivot={pivot}); refusing to write empty ground truth"
        )

    info: Dict[str, Any] = {
        "yaw_deg": round(float(yaw_deg * yaw_sign), 3),
        "pitch_deg": round(float(pitch_deg * yaw_sign), 3),
        "roll_deg": round(float(roll_deg * yaw_sign), 3),
        "pivot": [round(pivot[0], 2), round(pivot[1], 2)],
        **transform_details,
        "padding_px": [left, top, right, bottom],
        "padding_capped": bool(padding_capped),
        "pad_mode": pad_mode,
        "visible_object_fraction": round(float(ground_truth.mean()) / max(float(mask.mean()), 1e-9), 4),
    }
    return np.asarray(warped_image, dtype=np.uint8), ground_truth, info


def mirror(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                          Dict[str, Any]]:
    """Horizontal flip: an orientation change, kept as its own condition."""
    info = {"transformation": "mirror", "axis": "horizontal"}
    return image[:, ::-1].copy(), mask[:, ::-1].copy(), info


def apply(array: np.ndarray, mask: np.ndarray, variant: Variant,
          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply one viewpoint variant."""
    if variant.transformation == "mirror":
        transformed_image, transformed_mask, info = mirror(array, mask)
        return transformed_image, transformed_mask, info
    parameters = variant.parameters
    return rotate(
        array,
        mask,
        float(parameters["yaw_deg"]),
        float(parameters["pitch_deg"]),
        float(parameters["roll_deg"]),
        rng,
    )
