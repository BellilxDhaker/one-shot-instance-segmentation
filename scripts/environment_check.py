"""Environment diagnostics for the one-shot instance segmentation project.

Phase 0 only. This module inspects the *runtime* (GPU, CUDA, PyTorch,
RAM, disk) and answers three questions:

1. What hardware/software are we actually running on?
2. Does CUDA work, i.e. can a tensor really be moved to the GPU and used?
3. Is this runtime good enough for the project (with actionable warnings)?

Nothing here is specific to a particular GPU model: Colab hardware changes over
time, so every value is queried at runtime.

Used as a library from notebooks/01_colab_setup.ipynb and as a CLI:

    python scripts/environment_check.py
    python scripts/environment_check.py --json results/env_report.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

GB = 1024**3

# Packages we care about for the phases planned so far. Reported even when a
# package is missing so the notebook can show one consolidated status table.
TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "cv2",
    "numpy",
    "scipy",
    "sklearn",
    "matplotlib",
    "PIL",
    "tqdm",
    "pandas",
    "yaml",
    "gradio",
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _torch():
    """Import torch lazily so the module works before installation."""
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:  # pragma: no cover - depends on environment
        return None


def _package_version(name: str) -> Optional[str]:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    for attr in ("__version__", "VERSION", "version"):
        value = getattr(module, attr, None)
        if isinstance(value, str):
            return value
        if value is not None:
            return str(value)
    try:
        from importlib import metadata as _metadata  # noqa: PLC0415

        return _metadata.version(name)
    except Exception:
        return None


def _run(cmd: List[str], timeout: int = 15) -> Optional[str]:
    """Run a command and return stripped stdout, or None on any failure."""
    try:
        out = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return out.stdout.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


# ---------------------------------------------------------------------------
# hardware / software probing
# ---------------------------------------------------------------------------
def get_python_info() -> Dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "in_colab": _in_colab(),
    }


def _in_colab() -> bool:
    if "google.colab" in sys.modules:
        return True
    return "google.colab" in os.environ.get("PYTHONPATH", "") or bool(
        os.environ.get("COLAB_RELEASE_TAG")
    )


def get_gpu_info() -> Dict[str, Any]:
    """GPU name, total/free memory, compute capability, multi-GPU count."""
    info: Dict[str, Any] = {
        "available": False,
        "count": 0,
        "names": [],
        "name": None,
        "total_memory_bytes": None,
        "total_memory_gb": None,
        "free_memory_bytes": None,
        "free_memory_gb": None,
        "capability": None,
        "multi_gpu": False,
    }
    torch = _torch()
    if torch is None:
        return info

    try:
        info["available"] = bool(torch.cuda.is_available())
    except Exception:
        return info
    if not info["available"]:
        return info

    count = torch.cuda.device_count()
    info["count"] = count
    info["multi_gpu"] = count > 1
    for idx in range(count):
        info["names"].append(torch.cuda.get_device_name(idx))
    info["name"] = info["names"][0]

    props = _safe(lambda: torch.cuda.get_device_properties(0))
    if props is not None:
        info["total_memory_bytes"] = int(props.total_memory)
        info["total_memory_gb"] = round(props.total_memory / GB, 2)
        info["capability"] = f"{props.major}.{props.minor}"

    # mem_get_info reports *free* VRAM (reserved vs actually free), nvidia-smi
    # is the fallback when the driver API is unavailable.
    mem = _safe(lambda: torch.cuda.mem_get_info(0))
    if mem is not None:
        free_bytes, _total_bytes = mem
        info["free_memory_bytes"] = int(free_bytes)
        info["free_memory_gb"] = round(free_bytes / GB, 2)

    if info["free_memory_gb"] is None:
        smi = _run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free,name",
                "--format=csv,noheader,nounits",
            ]
        )
        if smi:
            first = smi.splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 2:
                try:
                    info["total_memory_bytes"] = int(float(parts[0])) * 1024**3
                    info["total_memory_gb"] = round(float(parts[0]) / 1024, 2)
                    info["free_memory_bytes"] = int(float(parts[1])) * 1024**3
                    info["free_memory_gb"] = round(float(parts[1]) / 1024, 2)
                    info["name"] = info["name"] or parts[2]
                except ValueError:
                    pass
    return info


def get_cuda_info() -> Dict[str, Any]:
    """CUDA build/runtime versions. `cuda_available` is the hard gate."""
    torch = _torch()
    info: Dict[str, Any] = {
        "cuda_available": False,
        "cuda_build_version": None,
        "cuda_runtime_version": None,
        "cudnn_version": None,
        "nvidia_driver_version": None,
        "device_count": 0,
    }
    if torch is None:
        return info
    info["cuda_available"] = _safe(lambda: bool(torch.cuda.is_available()), False)
    info["cuda_build_version"] = _safe(lambda: torch.version.cuda)
    info["cudnn_version"] = _safe(lambda: torch.backends.cudnn.version())
    if info["cuda_available"]:
        info["device_count"] = _safe(lambda: torch.cuda.device_count(), 0)
        info["cuda_runtime_version"] = _safe(
            lambda: torch._C._cuda_getCompiledVersion() // 1000
        )
    smi = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if smi:
        info["nvidia_driver_version"] = smi.splitlines()[0].strip()
    return info


def get_torch_info() -> Dict[str, Any]:
    torch = _torch()
    if torch is None:
        return {"installed": False, "version": None}
    info: Dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build_version": _safe(lambda: torch.version.cuda),
        "cudnn_version": _safe(lambda: torch.backends.cudnn.version()),
        "num_threads": _safe(lambda: torch.get_num_threads()),
    }
    if _safe(lambda: bool(torch.cuda.is_available()), False):
        info["cuda_device"] = _safe(lambda: torch.cuda.get_device_name(0))
        info["cuda_allocated_gb"] = round(torch.cuda.memory_allocated(0) / GB, 3)
    return info


def get_system_memory() -> Dict[str, Any]:
    """System RAM in bytes/GB. Uses sysconf on Linux, falls back to psutil."""
    info: Dict[str, Any] = {"total_bytes": None, "total_gb": None, "available_gb": None}
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        info["total_bytes"] = int(page_size * pages)
    except (AttributeError, ValueError, OSError):
        pass
    if info["total_bytes"] is None:
        try:  # pragma: no cover - optional dependency
            import psutil  # noqa: PLC0415

            mem = psutil.virtual_memory()
            info["total_bytes"] = int(mem.total)
            info["available_gb"] = round(mem.available / GB, 2)
        except Exception:
            return info
    if info["total_bytes"] is not None:
        info["total_gb"] = round(info["total_bytes"] / GB, 2)
    return info


def get_disk_info(paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    disks: List[Dict[str, Any]] = []
    for path in paths or ["/content", "/content/drive"]:
        if not os.path.exists(path):
            disks.append({"path": path, "exists": False})
            continue
        usage = _safe(lambda p=path: shutil.disk_usage(p))
        entry: Dict[str, Any] = {"path": path, "exists": True}
        if usage is not None:
            entry.update(
                {
                    "total_gb": round(usage.total / GB, 2),
                    "used_gb": round(usage.used / GB, 2),
                    "free_gb": round(usage.free / GB, 2),
                }
            )
        disks.append(entry)
    return disks


def get_package_versions() -> Dict[str, Optional[str]]:
    return {name: _package_version(name) for name in TRACKED_PACKAGES}


# ---------------------------------------------------------------------------
# functional GPU test
# ---------------------------------------------------------------------------
def gpu_inference_test(matrix_size: int = 2048, repeats: int = 20) -> Dict[str, Any]:
    """Move a tensor to CUDA and do real matmuls. Proves inference works."""
    torch = _torch()
    result: Dict[str, Any] = {
        "ran": False,
        "success": False,
        "device": None,
        "matrix_size": matrix_size,
        "mean_abs": None,
        "peak_memory_mb": None,
        "ms_per_iteration": None,
        "error": None,
    }
    if torch is None:
        result["error"] = "PyTorch is not installed."
        return result
    if not _safe(lambda: bool(torch.cuda.is_available()), False):
        result["error"] = "CUDA is not available in this runtime."
        return result

    try:
        result["ran"] = True
        device = torch.device("cuda")
        result["device"] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats(0)

        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)

        c = a @ b  # warm-up (kernel autotuning, lazy init)
        torch.cuda.synchronize(0)

        start = time.perf_counter()
        for _ in range(repeats):
            c = a @ b
        torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - start

        expected = matrix_size  # mean of N(0,1) @ N(0,1) row sums
        result["mean_abs"] = round(float(c.abs().mean().item()), 4)
        result["ms_per_iteration"] = round(elapsed / repeats * 1000, 3)
        result["peak_memory_mb"] = round(torch.cuda.max_memory_allocated(0) / 1024**2, 1)
        result["checksum_ok"] = abs(result["mean_abs"] - expected) < 1.0
        result["success"] = bool(result["checksum_ok"])

        del a, b, c
        torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - hardware dependent
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# ---------------------------------------------------------------------------
# sufficiency assessment
# ---------------------------------------------------------------------------
def check_memory_sufficiency(
    min_vram_gb: float = 8.0,
    recommended_vram_gb: float = 15.0,
    min_system_ram_gb: float = 12.0,
    min_free_disk_gb: float = 20.0,
    disk_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Decide whether the session has enough memory/disk for the project.

    `ok`        -> usable, proceed.
    `degraded`  -> usable but expected to hurt later phases.
    `insufficient` -> switch the Colab runtime.
    """
    gpu = get_gpu_info()
    ram = get_system_memory()
    disks = get_disk_info(disk_paths)

    vram_gb = gpu["free_memory_gb"] or gpu["total_memory_gb"]
    checks: List[Dict[str, Any]] = []

    if not gpu["available"]:
        checks.append(
            {
                "name": "GPU present",
                "status": "fail",
                "detail": "No CUDA-capable GPU detected.",
            }
        )
        vram_gb = None
    else:
        checks.append(
            {
                "name": "GPU present",
                "status": "pass",
                "detail": f"{gpu['name']} ({gpu['capability']}), "
                f"{gpu['total_memory_gb']} GB total / {gpu['free_memory_gb']} GB free",
            }
        )

    if vram_gb is not None:
        if vram_gb < min_vram_gb:
            status, detail = (
                "fail",
                f"{vram_gb} GB VRAM available, {min_vram_gb} GB required.",
            )
        elif vram_gb < recommended_vram_gb:
            status, detail = (
                "warn",
                f"{vram_gb} GB VRAM available: enough for Phase 0, "
                f"{recommended_vram_gb} GB recommended for later phases.",
            )
        else:
            status, detail = "pass", f"{vram_gb} GB VRAM available."
        checks.append({"name": "VRAM", "status": status, "detail": detail})

    ram_gb = ram.get("total_gb")
    if ram_gb is None:
        checks.append(
            {
                "name": "System RAM",
                "status": "warn",
                "detail": "Could not be determined on this platform.",
            }
        )
    elif ram_gb < min_system_ram_gb:
        checks.append(
            {
                "name": "System RAM",
                "status": "warn",
                "detail": f"{ram_gb} GB RAM available, {min_system_ram_gb} GB preferred. "
                "Large datasets may need reduced batch sizes.",
            }
        )
    else:
        checks.append(
            {"name": "System RAM", "status": "pass", "detail": f"{ram_gb} GB RAM."}
        )

    drive_disk = next((d for d in disks if "drive" in d.get("path", "")), None)
    if drive_disk is not None and drive_disk.get("free_gb") is not None:
        if drive_disk["free_gb"] < min_free_disk_gb:
            checks.append(
                {
                    "name": "Drive free space",
                    "status": "fail",
                    "detail": f"{drive_disk['free_gb']} GB free on {drive_disk['path']}, "
                    f"{min_free_disk_gb} GB recommended.",
                }
            )
        else:
            checks.append(
                {
                    "name": "Drive free space",
                    "status": "pass",
                    "detail": f"{drive_disk['free_gb']} GB free on {drive_disk['path']}.",
                }
            )
    else:
        checks.append(
            {
                "name": "Drive free space",
                "status": "warn",
                "detail": "Google Drive is not mounted yet; free space unknown.",
            }
        )

    statuses = {c["status"] for c in checks}
    if "fail" in statuses:
        overall = "insufficient"
    elif "warn" in statuses:
        overall = "degraded"
    else:
        overall = "ok"

    return {
        "overall": overall,
        "checks": checks,
        "thresholds": {
            "min_vram_gb": min_vram_gb,
            "recommended_vram_gb": recommended_vram_gb,
            "min_system_ram_gb": min_system_ram_gb,
            "min_free_disk_gb": min_free_disk_gb,
        },
        "warnings": _memory_warnings(gpu, checks, python_in_colab()),
    }


def python_in_colab() -> bool:
    return _in_colab()


def _memory_warnings(
    gpu: Dict[str, Any], checks: List[Dict[str, Any]], in_colab: bool
) -> List[str]:
    """Warnings derived purely from the resource checks (no re-probing)."""
    warnings: List[str] = []
    failed = [c for c in checks if c["status"] == "fail"]
    for check in failed:
        warnings.append(f"Insufficient resource - {check['name']}: {check['detail']}")
    if not in_colab:
        warnings.append(
            "Not running inside Google Colab: Drive mounting is unavailable here."
        )
    return warnings


def _dedupe(items: List[str]) -> List[str]:
    seen: set = set()
    unique: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def build_warnings(report: Dict[str, Any]) -> List[str]:
    """Actionable messages for an insufficient Colab runtime."""
    warnings: List[str] = []
    gpu = report.get("gpu", {})
    cuda = report.get("cuda", {})
    memory = report.get("memory", {})

    if not cuda.get("cuda_available"):
        warnings.append(
            "No CUDA GPU is available. In Colab use Runtime > Change runtime type "
            "and pick a 'GPU' hardware accelerator (e.g. T4 / L4 / A100)."
        )
        return warnings

    free_gb = gpu.get("free_memory_gb")
    total_gb = gpu.get("total_memory_gb")
    if free_gb is not None and total_gb is not None and free_gb < 0.5 * total_gb:
        warnings.append(
            f"Only {free_gb} GB of {total_gb} GB VRAM is free. Restart the runtime "
            "(Runtime > Restart runtime) to release memory held by other sessions."
        )
    if memory.get("overall") == "insufficient":
        for check in memory.get("checks", []):
            if check["status"] == "fail":
                warnings.append(f"Insufficient resource - {check['name']}: {check['detail']}")
    if gpu.get("name") and any(
        tag in gpu["name"].upper() for tag in ("T4", "P4", "K80")
    ):
        warnings.append(
            f"{gpu['name']} is an older, low-VRAM accelerator. Phase 0 is fine, but "
            "expect to reduce image resolution / batch size in later phases."
        )
    if gpu.get("multi_gpu"):
        warnings.append(
            f"{gpu['count']} GPUs visible. This notebook uses GPU 0; pin "
            "CUDA_VISIBLE_DEVICES if you need deterministic single-GPU behaviour."
        )
    if not report.get("python", {}).get("in_colab"):
        warnings.append(
            "Not running inside Google Colab. Drive mounting and GPU checks "
            "outside Colab need a different setup."
        )
    return warnings


# ---------------------------------------------------------------------------
# aggregate report
# ---------------------------------------------------------------------------
def get_environment_report(
    matrix_size: int = 2048,
    run_gpu_test: bool = True,
    disk_paths: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Collect everything into one JSON-serialisable dictionary.

    `thresholds` overrides the memory-sufficiency limits, e.g. the `resources:`
    block of `configs/paths.yaml`.
    """
    python_info = get_python_info()
    gpu = get_gpu_info()
    cuda = get_cuda_info()
    torch_info = get_torch_info()
    inference = (
        gpu_inference_test(matrix_size=matrix_size)
        if run_gpu_test
        else {"ran": False, "success": False, "error": "skipped"}
    )
    memory_kwargs = {
        key: float(value)
        for key, value in (thresholds or {}).items()
        if key in ("min_vram_gb", "recommended_vram_gb", "min_system_ram_gb", "min_free_disk_gb")
    }
    memory = check_memory_sufficiency(disk_paths=disk_paths, **memory_kwargs)

    report: Dict[str, Any] = {
        "python": python_info,
        "gpu": gpu,
        "cuda": cuda,
        "torch": torch_info,
        "packages": get_package_versions(),
        "disks": get_disk_info(disk_paths),
        "gpu_test": inference,
        "memory": memory,
    }
    report["warnings"] = _dedupe(memory.get("warnings", []) + build_warnings(report))
    report["checks"] = summarise_checks(report)
    report["summary"] = {
        "status": report_status(report),
        "gpu_name": gpu.get("name"),
        "gpu_memory_gb": gpu.get("total_memory_gb"),
        "gpu_free_memory_gb": gpu.get("free_memory_gb"),
        "cuda_version": cuda.get("cuda_build_version"),
        "cuda_available": cuda.get("cuda_available"),
        "torch_version": torch_info.get("version"),
        "python_version": python_info.get("version"),
        "warning_count": len(report["warnings"]),
    }
    return report


def summarise_checks(report: Dict[str, Any]) -> List[Dict[str, str]]:
    """Flat PASS/WARN/FAIL list, used by the notebook verification table."""
    gpu = report.get("gpu", {})
    cuda = report.get("cuda", {})
    torch_info = report.get("torch", {})
    inference = report.get("gpu_test", {})
    packages = report.get("packages", {})

    checks: List[Dict[str, str]] = [
        {
            "check": "PyTorch installed",
            "status": "pass" if torch_info.get("installed") else "fail",
            "detail": str(torch_info.get("version")),
        },
        {
            "check": "CUDA available",
            "status": "pass" if cuda.get("cuda_available") else "fail",
            "detail": f"built against CUDA {cuda.get('cuda_build_version')}",
        },
        {
            "check": "GPU detected",
            "status": "pass" if gpu.get("available") else "fail",
            "detail": f"{gpu.get('name')} / {gpu.get('total_memory_gb')} GB",
        },
        {
            "check": "GPU inference",
            "status": "pass" if inference.get("success") else "fail",
            "detail": str(
                inference.get("error")
                or f"{inference.get('ms_per_iteration')} ms/iter, "
                f"peak {inference.get('peak_memory_mb')} MB"
            ),
        },
    ]

    # Phase 0 requires these; the heavy vision extras are checked later.
    required = ["numpy", "yaml", "tqdm", "pandas", "matplotlib", "cv2", "PIL", "scipy", "sklearn"]
    missing = [p for p in required if packages.get(p) is None]
    checks.append(
        {
            "check": "Phase 0 packages",
            "status": "pass" if not missing else "warn",
            "detail": "all present" if not missing else f"missing: {', '.join(missing)}",
        }
    )
    if packages.get("torchvision") is None:
        checks.append(
            {"check": "torchvision", "status": "warn", "detail": "not installed yet"}
        )
    if packages.get("transformers") is None:
        checks.append(
            {
                "check": "transformers (DINOv2)",
                "status": "warn",
                "detail": "not installed yet - required in the feature-extraction phase",
            }
        )

    checks.append(
        {
            "check": "Memory sufficiency",
            "status": {"ok": "pass", "degraded": "warn", "insufficient": "fail"}[
                report.get("memory", {}).get("overall", "insufficient")
            ],
            "detail": report.get("memory", {}).get("overall", "unknown"),
        }
    )
    return checks


def report_status(report: Dict[str, Any]) -> str:
    statuses = {c["status"] for c in report.get("checks", [])}
    if "fail" in statuses:
        return "FAIL"
    if "warn" in statuses:
        return "WARN"
    return "PASS"


# ---------------------------------------------------------------------------
# human-readable rendering
# ---------------------------------------------------------------------------
def format_environment_report(report: Dict[str, Any]) -> str:
    gpu = report.get("gpu", {})
    cuda = report.get("cuda", {})
    torch_info = report.get("torch", {})
    python_info = report.get("python", {})
    inference = report.get("gpu_test", {})
    ram = report.get("memory", {})

    def gb(value):
        return "n/a" if value is None else f"{value} GB"

    lines = [
        "=" * 66,
        " ENVIRONMENT REPORT - robust one-shot instance segmentation",
        "=" * 66,
        f"{'Runtime':<26}: {'Google Colab' if python_info.get('in_colab') else 'local'}",
        f"{'Python':<26}: {python_info.get('version')} ({python_info.get('implementation')})",
        f"{'Platform':<26}: {python_info.get('platform')}",
        "-" * 66,
        f"{'GPU name':<26}: {gpu.get('name') or 'NONE DETECTED'}",
        f"{'GPU memory (total)':<26}: {gb(gpu.get('total_memory_gb'))}",
        f"{'GPU memory (free)':<26}: {gb(gpu.get('free_memory_gb'))}",
        f"{'Compute capability':<26}: {gpu.get('capability') or 'n/a'}",
        f"{'GPU count':<26}: {gpu.get('count')}",
        "-" * 66,
        f"{'CUDA available':<26}: {cuda.get('cuda_available')}",
        f"{'CUDA version (build)':<26}: {cuda.get('cuda_build_version') or 'n/a'}",
        f"{'CUDA version (runtime)':<26}: {cuda.get('cuda_runtime_version') or 'n/a'}",
        f"{'cuDNN version':<26}: {cuda.get('cudnn_version') or 'n/a'}",
        f"{'NVIDIA driver':<26}: {cuda.get('nvidia_driver_version') or 'n/a'}",
        "-" * 66,
        f"{'PyTorch version':<26}: {torch_info.get('version') or 'not installed'}",
        f"{'torch.version.cuda':<26}: {torch_info.get('cuda_build_version') or 'n/a'}",
        "-" * 66,
    ]

    if inference.get("ran"):
        lines += [
            f"{'GPU inference test':<26}: "
            f"{'SUCCESS' if inference.get('success') else 'FAILED'}",
            f"{'  matrix size':<26}: {inference.get('matrix_size')}",
            f"{'  ms / iteration':<26}: {inference.get('ms_per_iteration')}",
            f"{'  peak VRAM (MB)':<26}: {inference.get('peak_memory_mb')}",
            f"{'  value check':<26}: {inference.get('checksum_ok')}",
        ]
        if inference.get("error"):
            lines.append(f"{'  error':<26}: {inference['error']}")
    else:
        lines.append(f"{'GPU inference test':<26}: not run ({inference.get('error')})")

    lines.append("-" * 66)
    ram_check = next(
        (c for c in report.get("memory", {}).get("checks", []) if c["name"] == "System RAM"),
        None,
    )
    lines.append(f"{'System RAM':<26}: {ram_check['detail'] if ram_check else 'n/a'}")
    for disk in report.get("disks", []):
        if disk.get("exists"):
            lines.append(
                f"{'Disk ' + disk['path']:<26}: {disk.get('free_gb')} GB free / "
                f"{disk.get('total_gb')} GB"
            )
        else:
            lines.append(f"{'Disk ' + disk['path']:<26}: not mounted")

    lines.append("-" * 66)
    lines.append(" Resource checks")
    for check in report.get("memory", {}).get("checks", []):
        lines.append(f"  [{check['status'].upper():<4}] {check['name']}: {check['detail']}")

    lines.append("-" * 66)
    lines.append(" Summary checks")
    for check in report.get("checks", []):
        lines.append(f"  [{check['status'].upper():<4}] {check['check']}: {check['detail']}")

    warnings = report.get("warnings", [])
    if warnings:
        lines.append("-" * 66)
        lines.append(" WARNINGS")
        lines.extend(f"  ! {w}" for w in warnings)
    else:
        lines.append("-" * 66)
        lines.append(" No warnings: this runtime is suitable for the project.")

    lines.append("=" * 66)
    lines.append(f" OVERALL: {report.get('summary', {}).get('status', 'UNKNOWN')}")
    lines.append("=" * 66)
    return "\n".join(lines)


def print_environment_report(report: Dict[str, Any]) -> str:
    text = format_environment_report(report)
    print(text)
    return text


def save_environment_report(report: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    return os.path.abspath(path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", default=None, help="write the report as JSON")
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--no-gpu-test", action="store_true")
    parser.add_argument("--drive", default="/content/drive", help="Drive mount to inspect")
    parser.add_argument(
        "--paths-config",
        default=None,
        help="configs/paths.yaml; when given, `resources:` supplies the thresholds",
    )
    args = parser.parse_args(argv)

    thresholds = None
    if args.paths_config:
        try:
            try:
                from scripts.project_setup import load_paths_config  # noqa: PLC0415
            except ImportError:
                from project_setup import load_paths_config  # noqa: PLC0415

            thresholds = load_paths_config(args.paths_config).get("resources")
        except Exception as exc:
            print(f"! Could not read thresholds from {args.paths_config}: {exc}")

    report = get_environment_report(
        matrix_size=args.matrix_size,
        run_gpu_test=not args.no_gpu_test,
        disk_paths=["/content", args.drive],
        thresholds=thresholds,
    )
    print(format_environment_report(report))
    if args.json_path:
        print(f"\nJSON report written to: {save_environment_report(report, args.json_path)}")

    return 0 if report["summary"]["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())