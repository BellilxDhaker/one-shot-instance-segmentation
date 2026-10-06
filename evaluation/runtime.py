"""Timing, seeding and provenance for benchmark runs.

Phase 1 reported a runtime per example, which is enough to see that the pipeline
works. A benchmark needs more than that, because a mean IoU is meaningless
without knowing which machine produced it and how long each stage took. This
module collects the three things a result depends on: wall-clock time per stage,
GPU memory when a GPU is present, and a description of the environment that
produced the numbers.

Nothing here changes how the pipeline computes anything. The stages are measured
around calls the caller already makes, so timing never alters the algorithm:

    with Stopwatch() as watch:
        result = pipeline.segment_object(reference_image, reference_mask, target_image)
    record = watch.to_record()

GPU timings are only meaningful when device work is finished, so every measurement
synchronises CUDA first when it is available; without that, `cuda.synchronize`
timings drift with queueing rather than with compute.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

EPSILON = 1e-9


def synchronize(device: Any = None) -> None:
    """Wait for queued GPU work so a timing measures compute, not queueing."""
    import torch

    if not torch.cuda.is_available():
        return
    if device is None or "cuda" in str(device):
        torch.cuda.synchronize()


def gpu_memory_mb(device: Any = None) -> Optional[float]:
    """Peak CUDA memory in megabytes, or `None` when there is no GPU to measure."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        if device is not None and "cuda" not in str(device):
            return None
        index = 0 if device is None else torch.device(device).index or 0
        return float(torch.cuda.max_memory_allocated(index)) / (1024 ** 2)
    except Exception:  # noqa: BLE001 - reporting memory must never fail a run
        return None


def reset_gpu_memory(device: Any = None) -> None:
    """Clear the peak-memory counter so each sample is measured on its own."""
    try:
        import torch

        if torch.cuda.is_available() and (device is None or "cuda" in str(device)):
            index = 0 if device is None else torch.device(device).index or 0
            torch.cuda.reset_peak_memory_stats(index)
    except Exception:  # noqa: BLE001
        pass


@dataclass
class RuntimeRecord:
    """Wall-clock seconds for one unit of work, split by stage.

    Attributes:
        stages: seconds per named stage, in insertion order.
        total_seconds: sum of every stage measured.
        peak_gpu_mb: peak CUDA allocation during the work, if any.
        device: the device string the work ran on, for the record.
    """

    stages: Dict[str, float] = field(default_factory=dict)
    peak_gpu_mb: Optional[float] = None
    device: str = "cpu"

    @property
    def total_seconds(self) -> float:
        return float(sum(self.stages.values()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stages": {name: float(value) for name, value in self.stages.items()},
            "total_seconds": self.total_seconds,
            "peak_gpu_mb": self.peak_gpu_mb,
            "device": self.device,
        }


class Stopwatch:
    """Context manager that times the block and attributes it to one stage.

        with Stopwatch() as watch:
            with watch.stage("features"):
                features = encoder(image)
    """

    def __init__(self, device: Any = None, reset_memory: bool = False) -> None:
        self.device = str(device) if device is not None else "cpu"
        self.record = RuntimeRecord(device=self.device)
        self._started = 0.0
        self._open = False
        if reset_memory:
            reset_gpu_memory(device)

    def __enter__(self) -> "Stopwatch":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        # The record is finished by `to_record`, so leaving the block only has to
        # hand control back: an exception inside a stage must propagate unchanged.
        return False

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one named stage and add it to the record."""
        if self._open:
            raise RuntimeError("Stopwatch stages cannot be nested")
        synchronize(self.device)
        started = time.perf_counter()
        self._open = True
        try:
            yield
        finally:
            synchronize(self.device)
            self.record.stages[name] = self.record.stages.get(name, 0.0) + (
                time.perf_counter() - started
            )
            self._open = False

    def to_record(self) -> RuntimeRecord:
        """Finish the record, sampling peak GPU memory once at the end."""
        synchronize(self.device)
        self.record.peak_gpu_mb = gpu_memory_mb(self.device)
        return self.record


def set_seed(seed: int) -> int:
    """Seed every source of randomness this project can reach.

    Returns the seed so a caller can record it next to the results.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001 - torch is optional for pure-CPU bookkeeping
        pass
    return seed


def git_revision(path: Optional[str] = None) -> Optional[str]:
    """Current commit hash, or `None` when git is unavailable or this is not a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path or os.getcwd(), capture_output=True, text=True, timeout=10, check=False,
        )
        revision = result.stdout.strip()
        return revision or None
    except (OSError, subprocess.SubprocessError):
        return None


def package_version(name: str) -> Optional[str]:
    """Version of an installed distribution, or `None` when it is absent."""
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return None


def describe_environment(device: Any = None) -> Dict[str, Any]:
    """Record everything about the machine that could change a number.

    The point is that a reader can tell whether two result files are comparable
    without trusting the person who produced them.
    """
    environment: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {
            name: package_version(name)
            for name in ("torch", "torchvision", "transformers", "numpy", "scipy",
                         "matplotlib", "PIL")
        },
        "git_revision": git_revision(),
    }
    try:
        import torch

        environment["torch_cuda_build"] = str(torch.version.cuda)
        environment["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            environment["gpu_count"] = int(torch.cuda.device_count())
            environment["gpu_names"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
        environment["device"] = str(device) if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        if hasattr(torch.backends, "cudnn"):
            environment["cudnn_version"] = str(torch.backends.cudnn.version())
    except Exception as error:  # noqa: BLE001
        environment["torch_error"] = str(error)
    try:
        import transformers

        environment["transformers_cache"] = os.environ.get("HF_HOME", "")
    except Exception:  # noqa: BLE001
        pass
    return environment


def describe_models(pipeline: Any) -> Dict[str, Any]:
    """Model identifiers and parameter counts for the frozen encoders."""
    described: Dict[str, Any] = {}
    for name in ("encoder", "segmenter"):
        module = getattr(pipeline, name, None)
        if module is None:
            continue
        model = getattr(module, "model", None)
        parameters = None
        trainable = None
        if model is not None and hasattr(model, "parameters"):
            try:
                parameters = int(sum(p.numel() for p in model.parameters()))
                trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            except Exception:  # noqa: BLE001
                parameters = None
        described[name] = {
            "model_id": str(getattr(module, "model_id", "unknown")),
            "parameters": parameters,
            "trainable_parameters": trainable,
            "device": str(getattr(module, "device", "unknown")),
        }
    for key in ("candidate_threshold", "candidate_min_area", "candidate_top_k",
                "max_candidates", "blend_alpha"):
        if hasattr(pipeline, key):
            described[key] = getattr(pipeline, key)
    return described


def build_provenance(pipeline: Any = None, device: Any = None, seed: Optional[int] = None,
                     dataset: Optional[Dict[str, Any]] = None,
                     extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the provenance block saved beside a set of results."""
    provenance: Dict[str, Any] = {
        "environment": describe_environment(device),
        "created_unix": int(time.time()),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if seed is not None:
        provenance["seed"] = int(seed)
    if pipeline is not None:
        provenance["pipeline"] = describe_models(pipeline)
    if dataset is not None:
        provenance["dataset"] = dataset
    if extra:
        provenance.update(extra)
    return provenance


def write_json(path: str, payload: Any) -> str:
    """Write JSON with sorted keys and a trailing newline, creating parent folders."""
    import json

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    return path


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:  # noqa: BLE001
        pass
    return str(value)


def summarize_times(values: Sequence[float]) -> Dict[str, float]:
    """Mean, median, spread and extremes of a list of durations in seconds."""
    import numpy as np

    if not values:
        return {}
    array = np.asarray(list(values), dtype=float)
    return {
        "count": int(array.size),
        "mean_seconds": float(array.mean()),
        "median_seconds": float(np.median(array)),
        "std_seconds": float(array.std(ddof=0)),
        "min_seconds": float(array.min()),
        "max_seconds": float(array.max()),
        "p95_seconds": float(np.percentile(array, 95)),
    }
