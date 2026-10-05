"""End-to-end smoke test that runs anywhere, with or without model downloads.

Two modes:

* `--stub` (default): every stage runs against deterministic test doubles, so the
  plumbing, the mask pooling, the correspondence maths, the localisation, the
  metrics and the plotting are verified on a laptop CPU with no network. The
  stub encoder is a classic colour-and-gradient patch descriptor, and the stub
  segmenter fills the prompted box - neither pretends to be a foundation model.
* `--real`: the frozen DINOv2 and SAM checkpoints are loaded and the demo pair
  is segmented for real. Needs `transformers` plus hub access.

    python tools/smoke_test.py
    python tools/smoke_test.py --real --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.metrics import evaluate_mask, evaluate_results
from matching.correspondence import build_target_representation, cosine_similarity
from matching.localization import localize
from matching.reference_representation import ReferenceRepresentation, build_reference_representation
from models.feature_encoder import FeatureEncoder, FeatureMap, to_pil_image
from models.segmenter import SegmentationResult, Segmenter
from pipeline.one_shot_segmentation import OneShotPipeline, segment_object
from tools.make_demo_pair import make_demo_pair


class PatchDescriptorEncoder(FeatureEncoder):
    """Test double: per-patch colour and gradient statistics, L2-normalised.

    It implements the same `extract` / `extract_batch` contract as
    `FeatureEncoder`, which is what lets the pipeline be exercised end to end
    without a checkpoint. Descriptors: mean RGB, luminance standard deviation,
    absolute luminance gradient in x and y.
    """

    def __init__(self, patch_size: int = 16, max_size: int = 384, **kwargs) -> None:
        super().__init__(model_id="stub-patch-descriptor", max_size=max_size, **kwargs)
        self._patch = patch_size

    @property
    def patch_size(self) -> int:
        return self._patch

    def extract_batch(self, images: Sequence) -> list:
        return [self._extract_one(to_pil_image(image)) for image in images]

    def _extract_one(self, image) -> FeatureMap:
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        input_size = self._target_size(image.height, image.width)
        resized = F.interpolate(tensor, size=input_size, mode="bilinear", align_corners=False)
        pooled = F.avg_pool2d(resized, kernel_size=self._patch, stride=self._patch)
        luminance = resized.mean(dim=1, keepdim=True)
        variance = F.avg_pool2d(luminance**2, self._patch, self._patch) - pooled.mean(dim=1, keepdim=True) ** 2
        diff_y = (luminance[:, :, 1:] - luminance[:, :, :-1]).abs()
        diff_x = (luminance[:, :, :, 1:] - luminance[:, :, :, :-1]).abs()
        gradient_y = F.avg_pool2d(F.pad(diff_y, (0, 0, 0, 1), mode="replicate"), self._patch, self._patch)
        gradient_x = F.avg_pool2d(F.pad(diff_x, (0, 1, 0, 0), mode="replicate"), self._patch, self._patch)
        features = torch.cat([pooled, variance.clamp_min(0).sqrt(), gradient_x, gradient_y], dim=1)[0]
        return FeatureMap(
            tokens=F.normalize(features, dim=0),
            image_size=(image.height, image.width),
            input_size=input_size,
            patch_size=self._patch,
            model_id=self.model_id,
            normalised=True,
        )


class BoxSegmenter(Segmenter):
    """Test double: return the prompted box as the mask, with a fixed score."""

    def __init__(self, **kwargs) -> None:
        super().__init__(model_id="stub-box-segmenter", **kwargs)

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> "BoxSegmenter":
        return self

    def segment(self, image, point=None, box=None, multimask_output=True) -> SegmentationResult:
        if box is None and point is not None:
            box = (point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8)
        if box is None:
            raise ValueError("the stub segmenter needs a box or a point")
        pil = to_pil_image(image)
        mask = np.zeros((pil.height, pil.width), dtype=bool)
        x1, y1, x2, y2 = (int(value) for value in box)
        mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = True
        return SegmentationResult(
            mask=mask,
            score=0.9,
            scores=(0.9,),
            prompt={"box": [float(value) for value in box], "point": point},
            multimask_output=multimask_output,
        )

    def segment_region(self, image, region, add_center_point=True, multimask_output=True) -> SegmentationResult:
        from models.segmenter import region_bbox

        box = region_bbox(region)
        point = None
        if add_center_point and box is not None:
            point = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, 1)
        return self.segment(image, point=point, box=box, multimask_output=multimask_output)


def check_stages(reference_image, reference_mask, target_image, encoder) -> None:
    reference_features, target_features = encoder.extract_batch([reference_image, target_image])
    assert reference_features.grid_size == target_features.grid_size, "demo pair should share a grid"
    assert reference_features.tokens.shape[0] == 6, reference_features.tokens.shape
    assert torch.allclose(
        reference_features.tokens.norm(dim=0), torch.ones(reference_features.grid_size), atol=1e-4
    )

    reference = build_reference_representation(reference_features, reference_mask, min_coverage=0.5)
    assert reference.num_patches > 0
    assert reference.tokens.device == reference.weights.device, (
        f"tokens on {reference.tokens.device}, mask weights on {reference.weights.device}"
    )
    assert reference.coverage.device == reference.tokens.device
    assert 0.0 < reference.area_fraction < 1.0
    expected_centroid = (
        (np.nonzero(reference_mask)[1].mean()) / reference_mask.shape[1],
        (np.nonzero(reference_mask)[0].mean()) / reference_mask.shape[0],
    )
    assert abs(reference.centroid[0] - expected_centroid[0]) < 0.02, (reference.centroid, expected_centroid)

    target = build_target_representation(target_features)
    correspondence = cosine_similarity(target, reference, aggregate="both")
    assert correspondence.similarity_map.shape == target_image.size[::-1]
    assert 0.0 <= correspondence.similarity_map.min() and correspondence.similarity_map.max() <= 1.0
    assert isinstance(correspondence.similarity_map, np.ndarray), "the similarity map must be numpy"

    _check_device_alignment(reference, target)

    localization = localize(correspondence.similarity_map, threshold=0.5)
    assert localization.best is not None, "stub encoder should find the pasted object"
    print("stages ok:", {
        "grid": reference_features.grid_size,
        "reference_patches": reference.num_patches,
        "peak": round(correspondence.peak_score(), 3),
        "candidates": len(localization),
        "best_box": localization.best.box,
    })


def _check_device_alignment(reference, target) -> None:
    """A reference restored from disk is CPU-only; it must survive a GPU run.

    The `meta` device is the one second device available without a GPU, and it is
    enough to reproduce the failure: multiplying a real tensor by a `meta` one
    raises the same "expected all tensors to be on the same device" as a CPU/GPU
    mix. The prototype line that used to fail is exercised directly.
    """
    from matching.correspondence import align_reference

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "reference.npz")
        reference.save(path)
        reloaded = ReferenceRepresentation.load(path)

    assert reloaded.tokens.device.type == "cpu", "a saved reference reads back on the CPU"

    aligned = align_reference(reloaded, "meta")
    for name in ("tokens", "weights", "coverage", "grid"):
        assert getattr(aligned, name).device.type == "meta", f"{name} was not moved"
    assert aligned.num_patches == reloaded.num_patches
    assert aligned.image_size == reloaded.image_size
    assert aligned.model_id == reloaded.model_id

    prototype = aligned.weighted_mean_token()
    assert prototype.device.type == "meta"
    assert prototype.shape == (reloaded.channels,)

    assert align_reference(reloaded, "cpu") is reloaded, "a matching device must not copy"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        align_reference(reloaded, "meta")
    assert any("moving the reference representation" in str(item.message) for item in caught), (
        "moving devices should be announced, not silent"
    )
    print("device alignment ok: a CPU reference is moved once, with a warning")


def run_stub(device: str = "cpu") -> int:
    reference_image, reference_mask, target_image, target_mask = make_demo_pair()
    encoder = PatchDescriptorEncoder()
    segmenter = BoxSegmenter()

    check_stages(reference_image, reference_mask, target_image, encoder)

    pipeline = OneShotPipeline(encoder=encoder, segmenter=segmenter, device=device)
    result = pipeline.segment_object(reference_image, reference_mask, target_image)

    assert result.predicted_mask is not None and result.predicted_mask.any(), "no mask produced"
    assert result.similarity_map.shape == target_mask.shape
    assert set(result.runtime) >= {"features", "reference", "correspondence", "localization", "segmentation", "total"}
    assert result.candidate is not None and result.confidence > 0.0

    metrics = evaluate_mask(result.predicted_mask, target_mask)
    print("stub pipeline:", metrics)
    assert metrics.iou > 0.5, metrics

    report = evaluate_results([("demo", result.predicted_mask, target_mask, result)])
    print(report.to_table())

    module_level = segment_object(
        reference_image, reference_mask, target_image, device=device,
        pipeline=OneShotPipeline(encoder=PatchDescriptorEncoder(), segmenter=BoxSegmenter()),
    )
    assert module_level.predicted_mask is not None
    assert abs(module_level.confidence - result.confidence) < 1e-6, "segment_object must be deterministic"

    paths = result.save(os.path.join(PROJECT_ROOT, "results"), stem="smoke_stub")
    print("saved:", {key: os.path.basename(value) for key, value in paths.items()})
    assert all(os.path.exists(value) for value in paths.values())

    try:
        import matplotlib

        matplotlib.use("Agg")
        from visualization.results import save_result_figure

        figure_path = save_result_figure(
            os.path.join(PROJECT_ROOT, "results", "smoke_stub.png"),
            reference_image, reference_mask, target_image, result, ground_truth=target_mask,
        )
        assert os.path.exists(figure_path)
        print("figure:", os.path.relpath(figure_path, PROJECT_ROOT))
    except ImportError as error:
        print("skipped figure:", error)

    print("STUB SMOKE TEST PASSED")
    return 0


def run_real(device: Optional[str] = None) -> int:
    from models.segmenter import region_bbox

    reference_image, reference_mask, target_image, target_mask = make_demo_pair()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    started = time.perf_counter()
    result = segment_object(reference_image, reference_mask, target_image, device=device)
    print(f"real pipeline finished in {time.perf_counter() - started:.2f} s on {device}")
    print(json_summary(result))
    print(evaluate_mask(result.predicted_mask, target_mask))
    box = None if result.candidate is None else result.candidate.box
    print("candidate box:", box, "region_bbox agrees:", box == region_bbox(result.candidate_region))
    print("REAL SMOKE TEST PASSED")
    return 0


def json_summary(result) -> dict:
    payload = result.to_dict()
    payload["runtime"] = {
        key: (round(value, 3) if isinstance(value, float) else value)
        for key, value in payload["runtime"].items()
    }
    return payload


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="load the actual checkpoints")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    arguments = parser.parse_args(argv)
    if arguments.real:
        return run_real(arguments.device)
    return run_stub(arguments.device or "cpu")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
