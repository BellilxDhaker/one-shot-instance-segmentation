"""Step 6 - promptable segmentation.

Model choice
------------
`facebook/sam-vit-base` (Segment Anything ViT-B) through `transformers`.

* it is promptable: a box, a point, or both are exactly the weak supervision a
  one-shot setting can afford - the pipeline only has a rough region, not a
  mask;
* no training, no LoRA, no per-object optimisation, so the whole method stays a
  preprocessing pipeline plus two frozen forward passes;
* the hub checkpoint loads with `transformers` alone, so no external research
  repository is cloned or imported;
* ViT-B is the smallest SAM variant and runs in about a second per prompt on a
  T4, which matters because Phase 1 evaluates a handful of examples.

SAM refines the box from localisation; it never sees the reference mask, so the
two stages remain independent and each can be inspected on its own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from models.feature_encoder import ImageInput, to_pil_image

DEFAULT_SEGMENTER_ID = "facebook/sam-vit-base"
Point = Union[Tuple[float, float], Tuple[float, float, int]]


@dataclass
class SegmentationResult:
    """Output of one prompted segmentation call.

    Attributes:
        mask: (H, W) boolean mask at the original image resolution.
        score: predicted IoU of the selected mask, as reported by the model.
        scores: predicted IoU of every returned candidate mask.
        prompt: the prompt actually sent to the model, for provenance.
        multimask_output: whether several candidates were requested.
    """

    mask: np.ndarray
    score: float
    scores: Tuple[float, ...]
    prompt: Dict[str, Any]
    multimask_output: bool = True

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "scores": [float(value) for value in self.scores],
            "area": self.area,
            "prompt": self.prompt,
            "multimask_output": self.multimask_output,
        }


class Segmenter:
    """Frozen SAM wrapper.

    Args:
        model_id: Hugging Face identifier of the SAM checkpoint.
        device: torch device; `None` picks CUDA when available.
        dtype: inference dtype.
        cache_dir: optional directory for the downloaded weights.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_SEGMENTER_ID,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device) if device is not None else _default_device()
        self.dtype = dtype
        self.cache_dir = cache_dir
        self._model: Optional[torch.nn.Module] = None
        self._processor: Optional[Any] = None

    def load(self) -> "Segmenter":
        """Load the checkpoint and its processor, reusing them if already loaded."""
        if self._model is not None:
            return self
        from transformers import SamModel, SamProcessor

        kwargs: Dict[str, Any] = {}
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            kwargs["cache_dir"] = self.cache_dir
        self._processor = SamProcessor.from_pretrained(self.model_id, **kwargs)
        self._model = SamModel.from_pretrained(self.model_id, **kwargs)
        self._model.eval().to(device=self.device, dtype=self.dtype)
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        return self

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def describe(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "loaded": self.is_loaded,
        }

    def segment(
        self,
        image: ImageInput,
        point: Optional[Point] = None,
        box: Optional[Tuple[float, float, float, float]] = None,
        multimask_output: bool = True,
    ) -> SegmentationResult:
        """Segment one object given a point and/or a box.

        Args:
            image: target image as a path, array or PIL image.
            point: (x, y) in pixels, optionally (x, y, label) with label 1 for
                foreground and 0 for background.
            box: (x1, y1, x2, y2) in pixels.
            multimask_output: ask SAM for three candidate masks (part, sub-part,
                whole) and keep the highest scoring one.

        Returns:
            The `SegmentationResult` at the original image resolution.
        """
        if point is None and box is None:
            raise ValueError("segment() needs at least one prompt: a point or a box")
        self.load()
        pil_image = to_pil_image(image)
        model_inputs, meta = self._build_inputs(pil_image, point, box)

        with torch.inference_mode():
            outputs = self._model(**model_inputs, multimask_output=multimask_output)
        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            meta["original_sizes"],
            meta["reshaped_input_sizes"],
        )
        scores = outputs.iou_scores.detach().cpu().numpy().reshape(-1)
        selected = int(np.argmax(scores))
        masks_for_image = np.asarray(masks[0])
        candidates = masks_for_image.reshape(-1, *masks_for_image.shape[-2:])
        if candidates.shape[0] != scores.shape[0]:
            raise RuntimeError(
                f"SAM returned {candidates.shape[0]} masks and {scores.shape[0]} scores for one image; "
                "the model output layout changed and `selected` would be ambiguous"
            )
        return SegmentationResult(
            mask=np.asarray(candidates[selected], dtype=bool),
            score=float(scores[selected]),
            scores=tuple(float(value) for value in scores),
            prompt=self._prompt_record(point, box),
            multimask_output=multimask_output,
        )

    def segment_region(
        self,
        image: ImageInput,
        region: np.ndarray,
        add_center_point: bool = True,
        multimask_output: bool = True,
    ) -> SegmentationResult:
        """Segment from a binary region: its box, and optionally its centre point.

        This is the bridge from Step 5 to Step 6. The box alone can settle on a
        neighbouring object, so the centre of the region is sent as a foreground
        point as well.
        """
        box = region_bbox(region)
        point: Optional[Point] = None
        if add_center_point and box is not None:
            point = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, 1)
        return self.segment(image, point=point, box=box, multimask_output=multimask_output)

    def _build_inputs(
        self, image, point: Optional[Point], box: Optional[Tuple[float, float, float, float]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return `(model_inputs, post_processing_metadata)`.

        The two are kept apart on purpose: `original_sizes` and
        `reshaped_input_sizes` are processor bookkeeping and must never be
        forwarded to the model.
        """
        kwargs: Dict[str, Any] = {}
        if point is not None:
            kwargs["input_points"] = [[[float(point[0]), float(point[1])]]]
            if len(point) > 2:
                kwargs["input_labels"] = [[[int(point[2])]]]
        if box is not None:
            kwargs["input_boxes"] = [[[float(value) for value in box]]]
        processed = self._processor(image, return_tensors="pt", **kwargs)

        model_inputs: Dict[str, Any] = {
            "pixel_values": processed["pixel_values"].to(device=self.device, dtype=self.dtype),
        }
        for key in ("input_points", "input_labels", "input_boxes"):
            if processed.get(key) is not None:
                model_inputs[key] = processed[key].to(self.device)
        meta = {
            "original_sizes": processed["original_sizes"],
            "reshaped_input_sizes": processed["reshaped_input_sizes"],
        }
        return model_inputs, meta

    def _prompt_record(
        self, point: Optional[Point], box: Optional[Tuple[float, float, float, float]]
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {}
        if point is not None:
            record["point"] = [float(value) for value in point]
        if box is not None:
            record["box"] = [float(value) for value in box]
        return record


def region_bbox(region: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box (x1, y1, x2, y2) of a boolean region, or `None` if empty."""
    ys, xs = np.nonzero(np.asarray(region, dtype=bool))
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
