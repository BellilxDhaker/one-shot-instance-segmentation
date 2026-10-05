"""Deterministic demo pair for local runs and for the Colab notebook.

The notebook must run end to end with no uploads and no dataset, so it needs an
example where the answer is known. `make_demo_pair` builds one: a single striped
object is pasted into two different canvases, at a different position and a
smaller scale, and the exact rectangle it occupies is returned as the mask in
both images.

The target is not a copy-paste detection trick: the object is rescaled, the
background texture and noise change, and the object is the only element with that
colour signature. That is enough to exercise every stage (mask pooling,
correspondence, localisation, prompted segmentation) and to print a real IoU,
while staying small enough to reason about when something looks wrong.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
from PIL import Image


def make_demo_pair(seed: int = 7) -> Tuple[Image.Image, np.ndarray, Image.Image, np.ndarray]:
    """Return `(reference_image, reference_mask, target_image, target_mask)`."""
    rng = np.random.default_rng(seed)
    height, width = 384, 512

    canvas = rng.integers(28, 46, size=(height, width, 3), dtype=np.uint8)
    ys, xs = np.mgrid[0:height, 0:width]
    canvas = canvas + (((xs + ys) // 24) % 2 * 14).astype(np.uint8)[..., None]

    patch_h, patch_w = 168, 150
    yy, xx = np.mgrid[0:patch_h, 0:patch_w]
    stripes = (((xx // 7) + (yy // 11) * 2) % 3).astype(np.uint8)
    texture = np.stack(
        [
            np.where(stripes == 0, 236, 62),
            np.where(stripes == 1, 128, 92),
            np.where(stripes == 2, 44, 206),
        ],
        axis=-1,
    ).astype(np.uint8)
    texture = np.clip(texture.astype(np.int16) + rng.integers(-18, 18, texture.shape), 0, 255).astype(np.uint8)

    def paste(image: Image.Image, origin: Tuple[int, int], scale: float) -> Tuple[int, int]:
        piece = Image.fromarray(texture).resize(
            (int(patch_w * scale), int(patch_h * scale)), Image.BICUBIC
        )
        image.paste(piece, origin)
        return piece.size

    reference = Image.fromarray(canvas.copy())
    reference_size = paste(reference, (44, 96), 1.0)
    reference_mask = np.zeros((height, width), dtype=bool)
    reference_mask[96:96 + reference_size[1], 44:44 + reference_size[0]] = True

    shifted = np.clip(canvas.astype(np.int16) + rng.integers(-10, 10, canvas.shape), 0, 255).astype(np.uint8)
    target = Image.fromarray(shifted)
    target_size = paste(target, (286, 150), 0.78)
    target_mask = np.zeros((height, width), dtype=bool)
    target_mask[150:150 + target_size[1], 286:286 + target_size[0]] = True

    return reference, reference_mask, target, target_mask


def save_demo_pair(directory: str = "data", seed: int = 7) -> dict:
    """Write the demo pair to `directory` and return the file paths."""
    os.makedirs(directory, exist_ok=True)
    reference, reference_mask, target, target_mask = make_demo_pair(seed)
    paths = {
        "reference_image": os.path.join(directory, "reference_image.png"),
        "reference_mask": os.path.join(directory, "reference_mask.png"),
        "target_image": os.path.join(directory, "target_image.png"),
        "target_mask": os.path.join(directory, "target_mask.png"),
    }
    reference.save(paths["reference_image"])
    Image.fromarray((reference_mask * 255).astype(np.uint8), mode="L").save(paths["reference_mask"])
    target.save(paths["target_image"])
    Image.fromarray((target_mask * 255).astype(np.uint8), mode="L").save(paths["target_mask"])
    return paths
