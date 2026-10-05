"""Step 2 - the reference object representation.

The reference image carries a binary mask that says which object we are looking
for. This module turns that mask plus the image's dense features into a compact
but spatially faithful description of the object:

1. the mask is resized to the exact input grid the encoder saw, so no geometry
   is guessed;
2. it is average-pooled by the patch size, giving the fraction of every patch
   that belongs to the object;
3. the covered patches become the reference token set, keeping their grid
   coordinates and their coverage as weights.

Spatial information is preserved twice over: every reference token still knows
where it came from (`grid`), and the full coverage map is stored so callers can
visualise or re-threshold the selection. Nothing is collapsed into a single
vector, because a single vector would destroy the layout that later stages need.

The representation is a plain dataclass holding torch tensors, so it can be
cached in memory or written to disk with `save()` / `load()` and reused across
experiments.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from models.feature_encoder import FeatureMap

MaskInput = Union[np.ndarray, torch.Tensor, "Any"]


def to_binary_mask(mask: MaskInput) -> torch.Tensor:
    """Return the mask as a float tensor in [0, 1] shaped (1, 1, H, W)."""
    if isinstance(mask, torch.Tensor):
        tensor = mask.detach().float()
    elif isinstance(mask, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()
    else:
        array = np.asarray(mask)
        if array.dtype != np.float32 and array.dtype != np.float64:
            array = array.astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[1] > 1:
        tensor = tensor[:, :1]
    return tensor.clamp(0.0, 1.0)


@dataclass
class ReferenceRepresentation:
    """Description of the reference object, anchored on the patch grid.

    Attributes:
        tokens: (N, C) L2-normalised features of the object patches.
        weights: (N,) coverage of each selected patch, sums to the object area
            expressed in patches.
        grid: (N, 2) `(row, col)` position of each selected patch.
        coverage: (h, w) fraction of every patch covered by the reference mask.
        image_size: (H, W) size of the reference image in pixels.
        input_size: (H_in, W_in) size the encoder received.
        model_id: identifier of the encoder that produced the tokens.
        patch_size: patch side in encoder-input pixels.
        area_fraction: share of the reference image covered by the mask.
        centroid: mask centroid as normalised (x, y) in [0, 1].
        thresholded: whether any patch reached `min_coverage`; `False` means the
            most-covered patches were used instead and a warning was emitted.
    """

    tokens: torch.Tensor
    weights: torch.Tensor
    grid: torch.Tensor
    coverage: torch.Tensor
    image_size: Tuple[int, int]
    input_size: Tuple[int, int]
    model_id: str
    patch_size: int
    area_fraction: float
    centroid: Tuple[float, float]
    thresholded: bool = True

    @property
    def num_patches(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def channels(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def grid_size(self) -> Tuple[int, int]:
        return int(self.coverage.shape[0]), int(self.coverage.shape[1])

    def centroid_pixels(self) -> Tuple[float, float]:
        """Centroid in reference-image pixels."""
        return (self.centroid[0] * self.image_size[1], self.centroid[1] * self.image_size[0])

    def weighted_mean_token(self) -> torch.Tensor:
        """(C,) coverage-weighted average of the reference tokens."""
        weights = self.weights.to(device=self.tokens.device, dtype=self.tokens.dtype)
        weights = weights / weights.sum().clamp_min(1e-8)
        return F.normalize((self.tokens * weights.unsqueeze(1)).sum(dim=0), dim=0)

    def to(self, device: Union[str, torch.device]) -> "ReferenceRepresentation":
        """Copy of this representation with every tensor on `device`."""
        return ReferenceRepresentation(
            tokens=self.tokens.to(device),
            weights=self.weights.to(device),
            grid=self.grid.to(device),
            coverage=self.coverage.to(device),
            image_size=self.image_size,
            input_size=self.input_size,
            model_id=self.model_id,
            patch_size=self.patch_size,
            area_fraction=self.area_fraction,
            centroid=self.centroid,
            thresholded=self.thresholded,
        )

    def coverage_map(self, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Coverage resampled to pixel resolution, as a numpy array."""
        target = tuple(size) if size is not None else self.image_size
        resized = F.interpolate(
            self.coverage.unsqueeze(0).unsqueeze(0), size=target, mode="bilinear", align_corners=False
        )
        return resized.squeeze().detach().cpu().numpy()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "patch_size": self.patch_size,
            "image_size": list(self.image_size),
            "input_size": list(self.input_size),
            "grid_size": list(self.grid_size),
            "num_patches": self.num_patches,
            "channels": self.channels,
            "area_fraction": self.area_fraction,
            "centroid": list(self.centroid),
            "thresholded": self.thresholded,
            "coverage": self.coverage.detach().cpu().numpy().tolist(),
        }

    def save(self, path: str) -> str:
        """Write the representation to an `.npz` file next to a small `.json`."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            tokens=self.tokens.detach().cpu().numpy(),
            weights=self.weights.detach().cpu().numpy(),
            grid=self.grid.detach().cpu().numpy(),
            coverage=self.coverage.detach().cpu().numpy(),
        )
        with open(f"{os.path.splitext(path)[0]}.json", "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return os.path.abspath(path)

    @classmethod
    def load(cls, path: str) -> "ReferenceRepresentation":
        """Read back a representation written by `save()`."""
        with np.load(path) as data:
            tensors = {key: torch.from_numpy(data[key]) for key in data.files}
        with open(f"{os.path.splitext(path)[0]}.json", "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        return cls(
            tokens=tensors["tokens"].float(),
            weights=tensors["weights"].float(),
            grid=tensors["grid"].long(),
            coverage=tensors["coverage"].float(),
            image_size=tuple(meta["image_size"]),
            input_size=tuple(meta["input_size"]),
            model_id=meta["model_id"],
            patch_size=int(meta["patch_size"]),
            area_fraction=float(meta["area_fraction"]),
            centroid=tuple(meta["centroid"]),
            thresholded=bool(meta["thresholded"]),
        )


def build_reference_representation(
    features: FeatureMap,
    reference_mask: MaskInput,
    min_coverage: float = 0.5,
    fallback_fraction: float = 0.01,
) -> ReferenceRepresentation:
    """Select the reference-object patches described by `reference_mask`.

    Args:
        features: dense features of the reference image.
        reference_mask: binary mask of the reference object.
        min_coverage: minimum share of a patch that must be covered for the patch
            to belong to the object.
        fallback_fraction: when no patch reaches `min_coverage` (a very small
            object), the best `fallback_fraction` of patches is used instead.

    Returns:
        The `ReferenceRepresentation` for this reference.
    """
    mask = to_binary_mask(reference_mask)
    if mask.shape[-2:] != tuple(features.image_size):
        mask = F.interpolate(mask, size=tuple(features.image_size), mode="nearest")
    if mask.shape[-2:] != tuple(features.input_size):
        mask = F.interpolate(mask, size=tuple(features.input_size), mode="bilinear", align_corners=False)

    mask = mask.to(device=features.tokens.device)
    coverage = F.avg_pool2d(mask, kernel_size=features.patch_size, stride=features.patch_size)
    coverage = coverage.squeeze(0).squeeze(0).clamp(0.0, 1.0)

    area_fraction = float(mask.mean().item())

    ys, xs = torch.nonzero(coverage >= min_coverage, as_tuple=True)
    thresholded = bool(ys.numel() > 0)
    if not thresholded:
        keep = max(1, int(round(fallback_fraction * coverage.numel())))
        flat = coverage.flatten()
        top = torch.topk(flat, k=min(keep, flat.numel())).indices
        ys = torch.div(top, coverage.shape[1], rounding_mode="floor")
        xs = top % coverage.shape[1]
        warnings.warn(
            "No patch reached min_coverage="
            f"{min_coverage}; using the {ys.numel()} most covered patches instead. "
            "Lower min_coverage or enlarge the reference object.",
            stacklevel=2,
        )

    grid = torch.stack([ys, xs], dim=1)
    tokens = features.tokens[:, ys, xs].permute(1, 0).contiguous()
    weights = coverage[ys, xs].contiguous()
    if tokens.device != weights.device:  # pragma: no cover - guarded by the mask move above
        raise RuntimeError(
            f"reference tokens are on {tokens.device} but the pooled mask weights are on "
            f"{weights.device}; the mask was not moved onto the feature device"
        )

    centroid_y, centroid_x = _mask_centroid(mask)
    return ReferenceRepresentation(
        tokens=tokens,
        weights=weights,
        grid=grid,
        coverage=coverage,
        image_size=tuple(features.image_size),
        input_size=tuple(features.input_size),
        model_id=features.model_id,
        patch_size=features.patch_size,
        area_fraction=area_fraction,
        centroid=(centroid_x, centroid_y),
        thresholded=thresholded,
    )


def _mask_centroid(mask: torch.Tensor) -> Tuple[float, float]:
    binary = (mask > 0.5).float()
    total = binary.sum()
    if float(total) <= 0.0:
        return 0.5, 0.5
    rows = torch.arange(binary.shape[-2], dtype=torch.float32, device=binary.device).view(-1, 1)
    cols = torch.arange(binary.shape[-1], dtype=torch.float32, device=binary.device).view(1, -1)
    centre_y = float((binary * rows).sum().item() / total.item())
    centre_x = float((binary * cols).sum().item() / total.item())
    height, width = binary.shape[-2], binary.shape[-1]
    return centre_y / height, centre_x / width


class ReferenceRepresentationStore:
    """In-memory cache of reference representations, keyed by an explicit id."""

    def __init__(self) -> None:
        self._items: Dict[str, ReferenceRepresentation] = {}

    def put(self, key: str, representation: ReferenceRepresentation) -> None:
        self._items[key] = representation

    def get(self, key: str) -> Optional[ReferenceRepresentation]:
        return self._items.get(key)

    def keys(self) -> List[str]:
        return sorted(self._items)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)