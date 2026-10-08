"""Occlusion robustness: hide a graded share of the object with real texture.

Levels 1-5 hide 10-50% of the object's pixels. Every occluder is built from
photographic texture taken from elsewhere in the *same* target image, so the
covering region looks like scene content rather than a flat black box.

Four occluder shapes rotate across the ladder:

    OCCLUSIONS = (rect, ellipse, polygon, feathered patch)

and the shape used at a given level depends on the sample index
(`OCCLUSIONS[(level - 1 + sample_index) % 4]`), so across the benchmark each
level is served by every shape roughly equally often, instead of level 1 always
being squares.

Procedure per variant:

1. anchor the occluder centre on a random *object* pixel (a centre outside the
   object would occlude nothing);
2. draw a random shape aspect ratio, then bisect on the shape's radius until the
   part of the shape overlapping the ground truth reaches the level's target
   share to within +/- 0.02;
3. sample a texture window of the same size from a random image location that
   avoids the object when the image allows it;
4. paste the texture through the shape mask (hard edge for rect/ellipse/polygon,
   Gaussian-feathered for `patch`).

The ground truth is deliberately left **unchanged**: the pixels of the object
are still the object, they are merely invisible now, which is exactly the
condition the pipeline has to survive. Because a system cannot recover what it
cannot see, the analyzer additionally computes visible-region metrics against
`GT & ~occluder`, which is why the occluder mask is stored next to each variant
(`occluder_mask.png`) instead of being thrown away.

`info` reports `target_fraction` and `achieved_fraction`, so a level whose
target was unreachable (clipping at the image border) is visible in the data
rather than silently mislabelled.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from augmentation.common import Variant, bbox_of

FAMILY = "occlusion"

OCCLUSIONS: Tuple[str, ...] = ("rect", "ellipse", "polygon", "patch")
TARGET_FRACTIONS: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)
COVERAGE_TOLERANCE = 0.02
BISECTION_STEPS = 16


def build_variants(sample_index: int = 0) -> List[Variant]:
    """The five occlusion levels for one sample.

    `sample_index` only decides which shape each level gets, so the result is a
    pure function of the sample id and stays stable across rebuilds.
    """
    variants: List[Variant] = []
    for level, fraction in enumerate(TARGET_FRACTIONS, start=1):
        occluder_type = OCCLUSIONS[(level - 1 + int(sample_index)) % len(OCCLUSIONS)]
        variants.append(Variant(
            family=FAMILY,
            condition=occluder_type,
            severity=level,
            variant=f"occlude_{level}",
            transformation="occlude",
            parameters={"target_fraction": fraction, "occluder_type": occluder_type},
        ))
    return variants


def _shape_mask(shape: str, width: int, height: int, rx: float, ry: float,
                rng: np.random.Generator) -> np.ndarray:
    """Binary shape covering a (width, height) window centred in it."""
    xs = np.arange(width, dtype=np.float64) + 0.5 - width / 2.0
    ys = np.arange(height, dtype=np.float64) + 0.5 - height / 2.0
    grid_x, grid_y = np.meshgrid(xs, ys)
    if shape == "ellipse":
        return ((grid_x / max(rx, 1e-6)) ** 2 + (grid_y / max(ry, 1e-6)) ** 2) <= 1.0
    if shape == "rect":
        return np.ones((height, width), dtype=bool)
    if shape == "patch":
        # Feathering is applied later; the coverage search uses the solid core.
        return np.ones((height, width), dtype=bool)
    if shape == "polygon":
        from PIL import Image, ImageDraw

        sides = int(rng.integers(5, 9))
        angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
        angles += rng.uniform(0.0, 2.0 * np.pi / sides)
        jitter = rng.uniform(0.65, 1.0, size=sides)
        points = [
            (width / 2.0 + float(np.cos(a)) * rx * float(j),
             height / 2.0 + float(np.sin(a)) * ry * float(j))
            for a, j in zip(angles, jitter)
        ]
        canvas = Image.new("L", (width, height), 0)
        ImageDraw.Draw(canvas).polygon(points, fill=255)
        return np.asarray(canvas, dtype=np.uint8) > 0
    raise ValueError(f"unknown occluder shape: {shape!r}")


def _feather(mask: np.ndarray) -> np.ndarray:
    """Soft edges for the patch shape, so it blends like a shadowed surface."""
    from PIL import Image, ImageFilter

    height, width = mask.shape
    radius = max(2.0, min(height, width) / 6.0)
    canvas = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    blurred = canvas.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred, dtype=np.float64) / 255.0


def _pick_texture_window(width: int, height: int, size: Tuple[int, int],
                         avoid, rng: np.random.Generator) -> Tuple[int, int]:
    """Top-left corner of a same-sized texture window that avoids `avoid` when possible."""
    window_w, window_h = size
    span_x = max(1, width - window_w + 1)
    span_y = max(1, height - window_h + 1)
    best = (int(rng.integers(0, span_x)), int(rng.integers(0, span_y)))
    if avoid is None:
        return best
    ax0, ay0, ax1, ay1 = avoid
    for _ in range(32):
        candidate = (int(rng.integers(0, span_x)), int(rng.integers(0, span_y)))
        cx0, cy0 = candidate
        cx1, cy1 = cx0 + window_w, cy0 + window_h
        overlaps = not (cx1 <= ax0 or cx0 >= ax1 or cy1 <= ay0 or cy0 >= ay1)
        if not overlaps:
            return candidate
        best = candidate
    return best


def occlude(image: np.ndarray, mask: np.ndarray, target_fraction: float,
            occluder_type: str, rng: np.random.Generator,
            ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Hide ~`target_fraction` of the object; returns image, unchanged mask, info."""
    from PIL import Image

    height, width = mask.shape
    box = bbox_of(mask)
    if box is None:
        raise ValueError("occlude needs a non-empty ground-truth mask")
    object_pixels = np.flatnonzero(mask)
    if object_pixels.size == 0:
        raise ValueError("occlude needs at least one object pixel")

    anchor = int(rng.choice(object_pixels))
    anchor_x, anchor_y = anchor % width, anchor // width
    box_width = max(1, box[2] - box[0])
    box_height = max(1, box[3] - box[1])
    aspect = float(rng.uniform(0.6, 1.5))
    object_area = float(mask.sum())
    # The shape must be *identical* at every radius evaluation, otherwise the
    # bisection chases a moving target. One seed, re-created per render.
    shape_seed = int(rng.integers(0, 2**32 - 1))

    def render(radius: float) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
        rng_shape = np.random.default_rng(shape_seed)
        rx = max(1.0, radius)
        ry = max(1.0, radius * aspect)
        gx0 = int(np.floor(anchor_x - rx))
        gy0 = int(np.floor(anchor_y - ry))
        gx1 = int(np.ceil(anchor_x + rx))
        gy1 = int(np.ceil(anchor_y + ry))
        gx0c, gy0c = max(gx0, 0), max(gy0, 0)
        gx1c, gy1c = min(gx1, width), min(gy1, height)
        window_w, window_h = max(1, gx1c - gx0c), max(1, gy1c - gy0c)
        shape = _shape_mask(occluder_type, window_w, window_h,
                            (gx1 - gx0) / 2.0, (gy1 - gy0) / 2.0, rng_shape)
        if gy0c < gy1c and gx0c < gx1c:
            local_object = mask[gy0c:gy1c, gx0c:gx1c]
        else:
            local_object = np.zeros((window_h, window_w), dtype=bool)
        alpha = _feather(shape) if occluder_type == "patch" else shape.astype(np.float64)
        covered = float((alpha * local_object).sum())
        return alpha, (gx0c, gy0c, gx1c, gy1c), covered

    def coverage(radius: float) -> float:
        return render(radius)[2] / max(object_area, 1.0)

    lo, hi = 1.0, float(4 * max(width, height))
    if coverage(hi) < target_fraction:
        # Even a huge shape cannot hide this share (object spans the frame edge);
        # take the maximum and let `achieved_fraction` tell the truth.
        chosen_radius = hi
    else:
        best_radius, best_err = hi, abs(coverage(hi) - target_fraction)
        for _ in range(BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            current = coverage(mid)
            error = abs(current - target_fraction)
            if error < best_err:
                best_radius, best_err = mid, error
            if current >= target_fraction:
                hi = mid
            else:
                lo = mid
        chosen_radius = best_radius

    alpha, (gx0, gy0, gx1, gy1), covered = render(chosen_radius)
    window_w, window_h = gx1 - gx0, gy1 - gy0

    texture_x, texture_y = _pick_texture_window(width, height, (window_w, window_h),
                                                (box[0], box[1], box[2], box[3]), rng)
    texture = image[texture_y:texture_y + window_h, texture_x:texture_x + window_w]
    if texture.shape[:2] != (window_h, window_w):
        texture = np.asarray(
            Image.fromarray(texture, mode="RGB").resize((window_w, window_h),
                                                        Image.BILINEAR),
            dtype=np.uint8)

    region = image[gy0:gy1, gx0:gx1].astype(np.float64)
    if region.shape[:2] == alpha.shape:
        blended = region * (1.0 - alpha[..., None]) + texture.astype(np.float64) * alpha[..., None]
        image = image.copy()
        image[gy0:gy1, gx0:gx1] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    occluder_mask = np.zeros((height, width), dtype=bool)
    occluder_mask[gy0:gy1, gx0:gx1] = alpha > 0.5

    info: Dict[str, Any] = {
        "target_fraction": round(float(target_fraction), 4),
        "achieved_fraction": round(float(covered / object_area), 4),
        "within_tolerance": bool(
            abs(covered / object_area - target_fraction) <= COVERAGE_TOLERANCE),
        "occluder_type": occluder_type,
        "anchor": [anchor_x, anchor_y],
        "window": [gx0, gy0, gx1, gy1],
        "texture_origin": [texture_x, texture_y],
        "feathered": occluder_type == "patch",
        "aspect": round(aspect, 4),
        "artifacts": {"occluder_mask": occluder_mask},
    }
    return image, mask, info


def apply(array: np.ndarray, mask: np.ndarray, variant: Variant,
          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply one occlusion variant."""
    return occlude(
        array,
        mask,
        float(variant.parameters["target_fraction"]),
        str(variant.parameters["occluder_type"]),
        rng,
    )
