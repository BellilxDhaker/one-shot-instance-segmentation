"""Steps 3 and 4 - target representation and feature correspondence.

Step 3 wraps the target image features in a `TargetRepresentation`: every patch
of the target keeps its features and its grid position, so a patch-level
comparison is a plain matrix product.

Step 4 turns that into a *correspondence field*. For each target patch we
measure how similar it is to the reference object and resample the result to
pixel resolution, producing the `similarity_map` the later stages consume.

Two aggregations are available:

* `max` - the single best-matching reference patch per target patch. Sharp and
  good for texture repetition, but sensitive to a few spuriously high scores.
* `prototype` - similarity to the coverage-weighted mean reference token. More
  robust, blurrier, and it describes the object as a whole rather than by its
  most distinctive patch.

`both` returns the two maps separately plus their average, which is the default
in the pipeline: agreement between the two is a cheap, honest confidence signal.

Similarity is computed on L2-normalised tokens, so the cosine similarity is a
dot product. Patch scores are finally min-max normalised using robust percentile
clips, which keeps a handful of extreme values from flattening the rest of the
map.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from models.feature_encoder import FeatureMap
from matching.reference_representation import ReferenceRepresentation

AGGREGATIONS = ("max", "prototype", "both")


@dataclass
class TargetRepresentation:
    """Features of the target image, flattened for patch-wise comparison.

    Attributes:
        tokens: (M, C) L2-normalised target patch tokens, row-major over the grid.
        features: the underlying `FeatureMap`, kept for visualisation.
    """

    tokens: torch.Tensor
    features: FeatureMap

    @property
    def grid_size(self) -> Tuple[int, int]:
        return self.features.grid_size

    @property
    def num_patches(self) -> int:
        return int(self.tokens.shape[0])

    def positions(self) -> torch.Tensor:
        """(M, 2) normalised (x, y) centres of the target patches."""
        positions = self.features.grid_positions()
        grid_h, grid_w = self.grid_size
        return positions.reshape(grid_h * grid_w, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.features.model_id,
            "image_size": list(self.features.image_size),
            "grid_size": list(self.grid_size),
            "num_patches": self.num_patches,
            "channels": int(self.tokens.shape[1]),
        }


def build_target_representation(features: FeatureMap) -> TargetRepresentation:
    """Flatten the target feature map into row-major patch rows."""
    grid_h, grid_w, _ = features.tokens_hwc().shape
    tokens = features.tokens_hwc().reshape(grid_h * grid_w, -1).contiguous()
    return TargetRepresentation(tokens=tokens, features=features)


@dataclass
class CorrespondenceResult:
    """Pixel-level agreement between the reference object and the target.

    Attributes:
        similarity_map: (H, W) float map in [0, 1] at the target's pixel size.
        patch_scores: (M,) normalised score per target patch.
        patch_best_match: (M,) index of the best-matching reference patch.
        prototype_map: (H, W) prototype-only map when `aggregate="both"`.
        aggregate: which aggregation produced `similarity_map`.
        reference_centroid_distance: normalised distance between the similarity
            peak and the location the reference centroid would project to; used
            as a sanity signal, not as ground truth.
    """

    similarity_map: np.ndarray
    patch_scores: np.ndarray
    patch_best_match: np.ndarray
    grid_size: Tuple[int, int]
    aggregate: str
    prototype_map: Optional[np.ndarray] = None
    reference_centroid_distance: Optional[float] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.similarity_map.shape[:2])

    def peak_pixel(self) -> Tuple[int, int]:
        """(x, y) pixel coordinate of the most similar target patch."""
        grid_h, grid_w = self.grid_size
        index = int(np.argmax(self.patch_scores))
        row, col = divmod(index, grid_w)
        height, width = self.similarity_map.shape[:2]
        scale_x = width / grid_w
        scale_y = height / grid_h
        return ((col + 0.5) * scale_x, (row + 0.5) * scale_y)

    def peak_score(self) -> float:
        return float(self.patch_scores.max())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregate": self.aggregate,
            "peak_score": self.peak_score(),
            "peak_pixel": list(self.peak_pixel()),
            "grid_size": list(self.grid_size),
            "reference_centroid_distance": self.reference_centroid_distance,
        }


def align_reference(
    reference: "ReferenceRepresentation", device: Union[str, torch.device]
) -> "ReferenceRepresentation":
    """Return `reference` with every tensor on `device`.

    A representation restored with `ReferenceRepresentation.load` is always on the
    CPU, because that is where NumPy arrays live. Comparing it with GPU target
    features then fails deep inside a matmul with a bare "expected all tensors to
    be on the same device", which says nothing about the cause. Reconciling here
    keeps the cache usable across devices and names the move in a warning.

    Returns `reference` unchanged when it is already on `device`.
    """
    wanted = torch.device(device)
    tensors = (reference.tokens, reference.weights, reference.coverage)
    if all(tensor.device == wanted for tensor in tensors):
        return reference
    warnings.warn(
        f"moving the reference representation from {reference.tokens.device} to {wanted}; "
        "a cached reference is stored on the CPU, so this is expected on a GPU run",
        stacklevel=2,
    )
    return reference.to(wanted)


def cosine_similarity(
    target: TargetRepresentation,
    reference: ReferenceRepresentation,
    aggregate: str = "both",
    smooth_sigma: Optional[float] = 1.5,
    clip_percentile: float = 99.0,
    min_percentile: float = 1.0,
) -> CorrespondenceResult:
    """Compare target patches with the reference object patches.

    Args:
        target: features of the target image.
        reference: representation of the reference object.
        aggregate: one of `"max"`, `"prototype"`, `"both"`.
        smooth_sigma: Gaussian sigma in pixels applied to the similarity map;
            `None` disables smoothing.
        clip_percentile: upper percentile used before the min-max normalisation.
        min_percentile: lower percentile used before the min-max normalisation.

    Returns:
        The `CorrespondenceResult` with a pixel-resolution similarity map.
    """
    if aggregate not in AGGREGATIONS:
        raise ValueError(f"aggregate must be one of {AGGREGATIONS}, got {aggregate!r}")
    if target.tokens.shape[1] != reference.tokens.shape[1]:
        raise ValueError(
            "Reference and target features are not comparable: channel counts differ "
            f"({reference.tokens.shape[1]} vs {target.tokens.shape[1]}). Use the same encoder for both."
        )
    reference = align_reference(reference, target.tokens.device)

    scores = target.tokens @ reference.tokens.permute(1, 0).contiguous()
    best_index = scores.argmax(dim=1)
    max_scores = scores.max(dim=1).values
    prototype = reference.weighted_mean_token()
    prototype_scores = target.tokens @ prototype

    image_size = tuple(target.features.image_size)
    max_map = _to_pixel_map(max_scores, target.features, image_size)
    prototype_map = _to_pixel_map(prototype_scores, target.features, image_size)

    if aggregate == "max":
        patch_scores, combined = max_scores, max_map
    elif aggregate == "prototype":
        patch_scores, combined = prototype_scores, prototype_map
    else:
        patch_scores = 0.5 * (max_scores + prototype_scores)
        combined = 0.5 * (_normalise(max_map) + _normalise(prototype_map))

    combined = _smooth(combined, smooth_sigma)
    combined = _normalise(combined, clip_percentile, min_percentile)

    if aggregate == "both":
        return CorrespondenceResult(
            similarity_map=combined,
            patch_scores=_normalise_series(patch_scores, clip_percentile, min_percentile),
            patch_best_match=best_index.cpu().numpy(),
            grid_size=target.features.grid_size,
            aggregate=aggregate,
            prototype_map=prototype_map,
            reference_centroid_distance=_centroid_distance(combined, reference.centroid),
        )
    return CorrespondenceResult(
        similarity_map=combined,
        patch_scores=_normalise_series(patch_scores, clip_percentile, min_percentile),
        patch_best_match=best_index.cpu().numpy(),
        grid_size=target.features.grid_size,
        aggregate=aggregate,
        prototype_map=prototype_map if aggregate == "prototype" else None,
        reference_centroid_distance=_centroid_distance(combined, reference.centroid),
    )


def _to_pixel_map(scores: torch.Tensor, features: FeatureMap, image_size: Tuple[int, int]) -> np.ndarray:
    grid_h, grid_w = features.grid_size
    grid = scores.reshape(1, 1, grid_h, grid_w)
    resized = F.interpolate(grid, size=image_size, mode="bilinear", align_corners=False)
    return resized.squeeze().detach().cpu().numpy().astype(np.float32)


def _normalise(array: np.ndarray, clip_percentile: float = 99.0, min_percentile: float = 1.0) -> np.ndarray:
    high = float(np.percentile(array, clip_percentile))
    low = float(np.percentile(array, min_percentile))
    if high - low < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _normalise_series(scores: torch.Tensor, clip_percentile: float, min_percentile: float) -> np.ndarray:
    values = scores.detach().cpu().numpy()
    return _normalise(values, clip_percentile, min_percentile)


def _smooth(array: np.ndarray, sigma: Optional[float]) -> np.ndarray:
    if sigma is None or sigma <= 0:
        return array
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(array, sigma=sigma, mode="nearest").astype(np.float32)


def _centroid_distance(similarity_map: np.ndarray, reference_centroid: Tuple[float, float]) -> float:
    height, width = similarity_map.shape[:2]
    ys, xs = np.nonzero(similarity_map > 0.5)
    if xs.size == 0:
        return float("nan")
    centre_x = float(xs.mean()) / max(width - 1, 1)
    centre_y = float(ys.mean()) / max(height - 1, 1)
    return float(np.hypot(centre_x - reference_centroid[0], centre_y - reference_centroid[1]))
