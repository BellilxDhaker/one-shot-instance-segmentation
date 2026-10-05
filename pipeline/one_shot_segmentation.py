"""Step 7 - the one-shot pipeline that ties the stages together.

    result = segment_object(reference_image, reference_mask, target_image)
    result.predicted_mask      # (H, W) bool
    result.similarity_map      # (H, W) float in [0, 1]
    result.candidate_region    # (H, W) bool or None
    result.confidence          # float
    result.runtime             # per-stage seconds plus the total

The data flow is strictly linear, and each stage only depends on the output of
the previous one:

    reference image + mask  ->  feature encoder  ->  reference representation
    target image             ->  feature encoder  ->  target representation
    the two representations ->  correspondence   ->  similarity map
    similarity map           ->  localisation     ->  candidate regions
    best candidate           ->  segmenter (SAM)  ->  predicted mask

Nothing is trained, nothing is fitted per object, and the reference
representation is the only object-specific state, so the same weights serve any
query. Every stage can be replaced through the constructor, which is how the
test suite exercises the pipeline without downloading checkpoints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from matching.correspondence import (
    CorrespondenceResult,
    TargetRepresentation,
    build_target_representation,
    cosine_similarity,
)
from matching.localization import LocalizationResult, localize
from matching.reference_representation import (
    MaskInput,
    ReferenceRepresentation,
    build_reference_representation,
)
from models.feature_encoder import FeatureEncoder, ImageInput, get_default_encoder
from models.segmenter import Segmenter, SegmentationResult

STAGES = ("features", "reference", "correspondence", "localization", "segmentation")


@dataclass
class OneShotResult:
    """Everything one run produced, with the two artifacts required downstream.

    Attributes:
        predicted_mask: (H, W) boolean mask of the reference object in the target
            image, at the target's own resolution. `None` when localisation found
            nothing, so the failure is visible instead of silently empty.
        similarity_map: (H, W) float map in [0, 1].
        candidate_region: (H, W) boolean region of the selected candidate, or
            `None` when there was none.
        confidence: agreement of the selected region with the reference object,
            in [0, 1]; `0.0` when no candidate was found.
        runtime: seconds spent in each stage plus `total`, and the encoder/segmenter
            devices for provenance.
        candidate: the selected `Candidate` (box, score, area, centroid).
        candidates: every candidate that survived localisation, best first.
        segmentation: the raw `SegmentationResult` from the segmenter.
        correspondence: the raw `CorrespondenceResult`.
        localization: the raw `LocalizationResult`.
        reference: the reference representation, so a run can be inspected or its
            features reused.
    """

    predicted_mask: Optional[np.ndarray]
    similarity_map: np.ndarray
    candidate_region: Optional[np.ndarray]
    confidence: float
    runtime: Dict[str, Any]
    candidate: Optional[Any] = None
    candidates: Tuple[Any, ...] = ()
    segmentation: Optional[SegmentationResult] = None
    correspondence: Optional[CorrespondenceResult] = field(default=None, repr=False)
    localization: Optional[LocalizationResult] = field(default=None, repr=False)
    reference: Optional[ReferenceRepresentation] = field(default=None, repr=False)

    @property
    def found(self) -> bool:
        return self.predicted_mask is not None and bool(self.predicted_mask.any())

    @property
    def mask_area_fraction(self) -> float:
        if self.predicted_mask is None or self.predicted_mask.size == 0:
            return 0.0
        return float(self.predicted_mask.mean())

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "confidence": float(self.confidence),
            "found": self.found,
            "mask_area_fraction": self.mask_area_fraction,
            "runtime": self.runtime,
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "segmentation": None if self.segmentation is None else self.segmentation.to_dict(),
            "correspondence": None if self.correspondence is None else self.correspondence.to_dict(),
            "localization": None if self.localization is None else self.localization.to_dict(),
            "reference": None if self.reference is None else {
                key: value for key, value in self.reference.to_dict().items() if key != "coverage"
            },
        }
        if include_arrays:
            payload["predicted_mask"] = self.predicted_mask
            payload["similarity_map"] = self.similarity_map
            payload["candidate_region"] = self.candidate_region
        return payload

    def save(self, directory: str, stem: str = "one_shot_result") -> Dict[str, str]:
        """Write the mask, the similarity map and a JSON summary to `directory`."""
        import json
        import os

        os.makedirs(directory, exist_ok=True)
        paths: Dict[str, str] = {}

        if self.predicted_mask is not None:
            mask_path = os.path.join(directory, f"{stem}_mask.png")
            _write_mask(mask_path, self.predicted_mask)
            paths["mask"] = mask_path

        map_path = os.path.join(directory, f"{stem}_similarity.npy")
        np.save(map_path, self.similarity_map.astype(np.float32))
        paths["similarity_map"] = map_path

        json_path = os.path.join(directory, f"{stem}_summary.json")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, default=float)
        paths["summary"] = json_path
        return paths


class OneShotPipeline:
    """Stage-agnostic facade around the seven steps.

    Args:
        encoder: a `FeatureEncoder`; built from `encoder_kwargs` if omitted.
        segmenter: a `Segmenter`; built from `segmenter_kwargs` if omitted.
        device: device passed to whichever model is created here.
        encoder_kwargs / segmenter_kwargs: forwarded to the two constructors.
        min_coverage: patch coverage needed for a reference patch to count.
        aggregate: correspondence aggregation, see `matching.correspondence`.
        localization_kwargs: defaults for Step 5, overridable per call.
        segmentation_kwargs: defaults for Step 6, overridable per call.
    """

    def __init__(
        self,
        encoder: Optional[FeatureEncoder] = None,
        segmenter: Optional[Segmenter] = None,
        device: Optional[Union[str, Any]] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        segmenter_kwargs: Optional[Dict[str, Any]] = None,
        min_coverage: float = 0.5,
        aggregate: str = "both",
        localization_kwargs: Optional[Dict[str, Any]] = None,
        segmentation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.encoder = encoder or FeatureEncoder(device=device, **(encoder_kwargs or {}))
        self.segmenter = segmenter or Segmenter(device=device, **(segmenter_kwargs or {}))
        self.min_coverage = min_coverage
        self.aggregate = aggregate
        self.localization_defaults = dict(localization_kwargs or {})
        self.segmentation_defaults = dict(segmentation_kwargs or {})

    def run(
        self,
        reference_image: ImageInput,
        reference_mask: MaskInput,
        target_image: ImageInput,
        localization_kwargs: Optional[Dict[str, Any]] = None,
        segmentation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> OneShotResult:
        """Run the full pipeline for one reference/target pair."""
        timings: Dict[str, float] = {}

        started = time.perf_counter()
        reference_features, target_features = self._extract(reference_image, target_image)
        timings["features"] = time.perf_counter() - started

        started = time.perf_counter()
        reference = build_reference_representation(
            reference_features, reference_mask, min_coverage=self.min_coverage
        )
        target = build_target_representation(target_features)
        timings["reference"] = time.perf_counter() - started

        started = time.perf_counter()
        correspondence = cosine_similarity(target, reference, aggregate=self.aggregate)
        timings["correspondence"] = time.perf_counter() - started

        started = time.perf_counter()
        localization = localize(similarity_map=correspondence.similarity_map, **{
            **self.localization_defaults, **(localization_kwargs or {})
        })
        timings["localization"] = time.perf_counter() - started

        started = time.perf_counter()
        segmentation, mask = self._segment(target_image, localization)
        timings["segmentation"] = time.perf_counter() - started

        best = localization.best
        runtime: Dict[str, Any] = dict(timings)
        runtime["total"] = float(sum(timings.values()))
        runtime["encoder_device"] = str(getattr(self.encoder, "device", "unknown"))
        runtime["segmenter_device"] = str(getattr(self.segmenter, "device", "unknown"))
        runtime["encoder_model"] = getattr(self.encoder, "model_id", "custom")
        runtime["segmenter_model"] = getattr(self.segmenter, "model_id", "custom")

        return OneShotResult(
            predicted_mask=mask,
            similarity_map=correspondence.similarity_map,
            candidate_region=None if best is None else best.region,
            confidence=localization.confidence,
            runtime=runtime,
            candidate=best,
            candidates=tuple(localization.candidates),
            segmentation=segmentation,
            correspondence=correspondence,
            localization=localization,
            reference=reference,
        )

    def segment_object(
        self,
        reference_image: ImageInput,
        reference_mask: MaskInput,
        target_image: ImageInput,
        **kwargs: Any,
    ) -> OneShotResult:
        """Documented alias of `run`, the public entry point of the project."""
        return self.run(reference_image, reference_mask, target_image, **kwargs)

    def describe(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder.describe(),
            "segmenter": self.segmenter.describe(),
            "min_coverage": self.min_coverage,
            "aggregate": self.aggregate,
            "localization": self.localization_defaults,
            "segmentation": self.segmentation_defaults,
        }

    def _extract(self, reference_image: ImageInput, target_image: ImageInput):
        reference, target = self.encoder.extract_batch([reference_image, target_image])
        return reference, target

    def _segment(
        self, target_image: ImageInput, localization: LocalizationResult
    ) -> Tuple[Optional[SegmentationResult], Optional[np.ndarray]]:
        best = localization.best
        if best is None:
            return None, None
        options = {**self.segmentation_defaults, "region": best.region}
        segmentation = self.segmenter.segment_region(target_image, **options)
        return segmentation, segmentation.mask


_DEFAULT_PIPELINE: Optional[OneShotPipeline] = None


def get_default_pipeline(device: Optional[Union[str, Any]] = None, **kwargs: Any) -> OneShotPipeline:
    """Process-wide pipeline so the two checkpoints load only once."""
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = OneShotPipeline(device=device, **kwargs)
    return _DEFAULT_PIPELINE


def segment_object(
    reference_image: ImageInput,
    reference_mask: MaskInput,
    target_image: ImageInput,
    device: Optional[Union[str, Any]] = None,
    pipeline: Optional[OneShotPipeline] = None,
    **kwargs: Any,
) -> OneShotResult:
    """Segment `target_image` at the object described by `reference_image`/`reference_mask`.

    Args:
        reference_image: image containing the query object.
        reference_mask: binary mask of that object in the reference image.
        target_image: image to segment.
        device: device for the models when a fresh pipeline is created.
        pipeline: reuse an existing pipeline instead of the default one.
        **kwargs: forwarded to `OneShotPipeline.run`.

    Returns:
        A `OneShotResult` with `predicted_mask`, `candidate_region`,
        `similarity_map`, `confidence` and `runtime`.
    """
    active = pipeline or get_default_pipeline(device=device)
    return active.segment_object(reference_image, reference_mask, target_image, **kwargs)


def _write_mask(path: str, mask: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8), mode="L").save(path)
