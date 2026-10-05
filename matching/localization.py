"""Step 5 - candidate localisation, independent of segmentation.

The correspondence map is noisy at the pixel level, so localisation turns it
into a small number of ranked candidate regions:

1. threshold the similarity map;
2. clean the binary map with a morphological closing (and optional opening) to
   remove speckle and close pinholes;
3. label the connected components and drop the tiny ones;
4. turn each component into a `Candidate` carrying a box, a region mask, a
   score (mean similarity inside the region) and a centroid.

This stage is deliberately separate from segmentation. It answers *where* the
object is and *how much* the features agree there; the segmenter in Step 6
answers *which pixels belong to it*. Keeping them apart makes both stages
inspectable on their own in the notebook, and it means the boxes can be
evaluated without loading a segmentation model at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Candidate:
    """One ranked region proposed by the correspondence map.

    Attributes:
        box: (x1, y1, x2, y2) pixel bounding box, integers, `x2`/`y2` exclusive.
        region: (H, W) boolean mask of the component.
        score: mean similarity inside the region, in [0, 1].
        peak: highest similarity inside the region, in [0, 1].
        area: component area in pixels.
        area_fraction: share of the image covered by the component.
        centroid: (x, y) centroid of the component in pixels.
    """

    box: Tuple[int, int, int, int]
    region: np.ndarray
    score: float
    peak: float
    area: int
    area_fraction: float
    centroid: Tuple[float, float]

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": [int(value) for value in self.box],
            "score": float(self.score),
            "peak": float(self.peak),
            "area": int(self.area),
            "area_fraction": float(self.area_fraction),
            "centroid": [float(value) for value in self.centroid],
        }


@dataclass
class LocalizationResult:
    """All candidate regions for one target image, best first.

    Attributes:
        candidates: ranked `Candidate` list.
        binary_map: (H, W) boolean thresholded similarity map before filtering.
        threshold: threshold that produced `binary_map`.
        similarity_map: the map the threshold was applied to, for debugging.
    """

    candidates: List[Candidate]
    binary_map: np.ndarray
    threshold: float
    similarity_map: Optional[np.ndarray] = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def best(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def best_region(self) -> Optional[np.ndarray]:
        best = self.best
        return None if best is None else best.region

    @property
    def confidence(self) -> float:
        return 0.0 if self.best is None else float(self.best.score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "num_candidates": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def localize(
    similarity_map: np.ndarray,
    threshold: float = 0.5,
    min_area_fraction: float = 0.002,
    closing: int = 7,
    opening: int = 0,
    max_candidates: int = 5,
) -> LocalizationResult:
    """Extract ranked candidate regions from a similarity map.

    Args:
        similarity_map: (H, W) map in [0, 1] from Step 4.
        threshold: similarity above which a pixel belongs to a candidate.
        min_area_fraction: components smaller than this share of the image are
            discarded as noise.
        closing: side of the square structuring element for the morphological
            closing; `0` disables it.
        opening: side for the morphological opening; `0` disables it.
        max_candidates: how many components to keep after ranking.

    Returns:
        A `LocalizationResult` whose candidates are sorted by descending score.
    """
    if similarity_map.ndim != 2:
        raise ValueError(f"similarity_map must be 2-D, got shape {similarity_map.shape}")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    from scipy.ndimage import binary_closing, binary_opening, label as connected_components

    binary = similarity_map >= threshold
    cleaned = binary
    if closing:
        cleaned = binary_closing(cleaned, structure=np.ones((closing, closing), dtype=bool))
    if opening:
        cleaned = binary_opening(cleaned, structure=np.ones((opening, opening), dtype=bool))
    cleaned = np.asarray(cleaned, dtype=bool)

    labels, count = connected_components(cleaned)
    height, width = cleaned.shape
    min_area = max(1, int(round(min_area_fraction * height * width)))

    candidates: List[Candidate] = []
    for label in range(1, count + 1):
        region = labels == label
        if int(region.sum()) < min_area:
            continue
        candidates.append(_make_candidate(region, similarity_map))

    candidates.sort(key=lambda candidate: (candidate.score, candidate.peak, candidate.area), reverse=True)
    return LocalizationResult(
        candidates=candidates[:max_candidates],
        binary_map=binary,
        threshold=threshold,
        similarity_map=similarity_map,
    )


def _make_candidate(region: np.ndarray, similarity_map: np.ndarray) -> Candidate:
    ys, xs = np.nonzero(region)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    values = similarity_map[region]
    area = int(region.sum())
    total = float(region.size)
    return Candidate(
        box=(x1, y1, x2, y2),
        region=region,
        score=float(values.mean()),
        peak=float(values.max()),
        area=area,
        area_fraction=area / total,
        centroid=(float(xs.mean()), float(ys.mean())),
    )


def merge_boxes(
    boxes: Sequence[Tuple[int, int, int, int]], iou_threshold: float = 0.5
) -> List[Tuple[int, int, int, int]]:
    """Greedy non-maximum suppression over boxes, used to combine prompts."""
    remaining = sorted(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    kept: List[Tuple[int, int, int, int]] = []
    for box in remaining:
        if all(_box_iou(box, other) < iou_threshold for other in kept):
            kept.append(box)
    return kept


def _box_iou(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0
