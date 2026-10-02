"""Reproducibility utilities: seeding, determinism and experiment metadata.

Every experiment in this project must be traceable. Two things are provided:

* `set_seed(...)`     - seeds Python, NumPy, PyTorch (CPU + all CUDA devices)
                        and optionally enables deterministic kernels.
* `ExperimentMetadata` - a small record of *how* a run was produced: UTC
                        timestamp, git commit/branch/dirty state, resolved
                        config, seed, GPU and software versions. Serialisable to
                        JSON and written next to the experiment results.

Nothing here is phase specific; it is used from the setup notebook and reused
by later experiment scripts.

CLI smoke test:

    python scripts/reproducibility.py --seed 1234 --out results/environment
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_SEED = 1234

# Environment variables that must be set *before* the first CUDA call for
# deterministic cuBLAS reductions. Set at import time so that importing this
# module early is enough.
_DETERMINISTIC_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def _torch():
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:
        return None


def apply_deterministic_env() -> Dict[str, str]:
    """Set determinism-related environment variables (idempotent)."""
    applied = {}
    for key, value in _DETERMINISTIC_ENV.items():
        os.environ.setdefault(key, value)
        applied[key] = os.environ[key]
    return applied


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------
def set_seed(
    seed: int = DEFAULT_SEED,
    deterministic: bool = True,
    benchmark: bool = False,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Seed every RNG this project can touch.

    Args:
        seed: the master seed.
        deterministic: enable deterministic algorithms / cudnn settings.
            Note that some CUDA ops have no deterministic implementation; the
            project calls `torch.use_deterministic_algorithms(..., warn_only=True)`
            so experiments keep running while still being as reproducible as
            possible.
        benchmark: keep cudnn autotuning on. Leave off when deterministic.
        quiet: suppress the confirmation print.

    Returns a small record of what was actually applied (used in metadata).
    """
    seed = int(seed)
    random.seed(seed)

    applied: Dict[str, Any] = {
        "seed": seed,
        "deterministic": bool(deterministic),
        "python_random": True,
        "numpy": False,
        "torch": False,
        "cuda": False,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
        "torch_use_deterministic_algorithms": None,
        "env": {},
    }

    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed)
        applied["numpy"] = True
    except Exception:
        pass

    torch = _torch()
    if torch is not None:
        torch.manual_seed(seed)
        applied["torch"] = True

        if _safe_cuda_available(torch):
            torch.cuda.manual_seed_all(seed)
            applied["cuda"] = True

        if deterministic:
            applied["env"] = apply_deterministic_env()
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = bool(benchmark)
            applied["cudnn_deterministic"] = True
            applied["cudnn_benchmark"] = bool(benchmark)
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                applied["torch_use_deterministic_algorithms"] = "warn_only"
            except Exception:
                applied["torch_use_deterministic_algorithms"] = False
        else:
            torch.backends.cudnn.benchmark = bool(benchmark)
            applied["cudnn_benchmark"] = bool(benchmark)

    if not quiet:
        bits = [f"seed={seed}"]
        bits.append("deterministic" if deterministic else "non-deterministic")
        bits.append(
            "python+"
            + "+".join(
                name
                for name, flag in (
                    ("numpy", applied["numpy"]),
                    ("torch", applied["torch"]),
                    ("cuda", applied["cuda"]),
                )
                if flag
            )
        )
        print(f"[reproducibility] {' | '.join(bits)}")
    return applied


def _safe_cuda_available(torch) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def seed_worker(worker_id: int) -> None:
    """`worker_init_fn` for PyTorch DataLoader workers (keeps loads repeatable)."""
    base_seed = torch.initial_seed() % 2**32 if _torch() is not None else 0
    np_seed = (base_seed + worker_id) % 2**32
    random.seed(np_seed)
    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(np_seed)
    except Exception:
        pass


def seeded_generator(seed: int = DEFAULT_SEED):
    """A torch.Generator for DataLoader shuffling / augmentation streams."""
    torch = _torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for seeded_generator().")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


# ---------------------------------------------------------------------------
# environment / provenance capture
# ---------------------------------------------------------------------------
def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_git_info(repo_root: Optional[str] = None) -> Dict[str, Any]:
    """Commit, branch and dirty state of the repository at `repo_root`."""
    info: Dict[str, Any] = {
        "available": False,
        "commit": None,
        "short_commit": None,
        "branch": None,
        "remote": None,
        "dirty": None,
        "describe": None,
    }
    if repo_root:
        info["repo_root"] = os.path.abspath(repo_root)
    if not repo_root or not os.path.isdir(os.path.join(repo_root, ".git")):
        return info

    def run(args: List[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git"] + args,
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
            if out.returncode != 0:
                return None
            return out.stdout.decode("utf-8", errors="replace").strip()
        except Exception:
            return None

    info["available"] = True
    info["commit"] = run(["rev-parse", "HEAD"])
    info["short_commit"] = run(["rev-parse", "--short", "HEAD"])
    info["branch"] = run(["rev-parse", "--abbrev-ref", "HEAD"])
    info["remote"] = run(["config", "--get", "remote.origin.url"])
    info["describe"] = run(["describe", "--always", "--dirty", "--tags"])
    status = run(["status", "--porcelain"])
    info["dirty"] = bool(status) if status is not None else None
    return info


def get_software_versions() -> Dict[str, Optional[str]]:
    """Versions of the stack that can change numerical results."""
    versions: Dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in ("torch", "torchvision", "transformers", "numpy", "cv2", "scipy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except Exception:
            versions[name] = None
    torch = _torch()
    if torch is not None:
        versions["torch_cuda_build"] = torch.version.cuda
        versions["cudnn"] = (
            str(torch.backends.cudnn.version())
            if _safe_cuda_available(torch)
            else None
        )
    return versions


def get_gpu_summary() -> Dict[str, Any]:
    """GPU name / memory, or a clear reason why there is none."""
    torch = _torch()
    if torch is None:
        return {"available": False, "reason": "PyTorch not installed"}
    try:
        if not torch.cuda.is_available():
            return {"available": False, "reason": "CUDA not available"}
        props = torch.cuda.get_device_properties(0)
        free_bytes, _ = torch.cuda.mem_get_info(0)
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "free_memory_gb": round(free_bytes / 1024**3, 2),
            "capability": f"{props.major}.{props.minor}",
            "device_count": torch.cuda.device_count(),
        }
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# experiment metadata
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ExperimentMetadata:
    """Everything needed to reproduce one experiment run."""

    experiment_name: str
    config: Dict[str, Any] = dataclasses.field(default_factory=dict)
    seed: int = DEFAULT_SEED
    deterministic: bool = True
    repo_root: Optional[str] = None
    notes: str = ""
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        git = get_git_info(self.repo_root or default_repo_root())
        return {
            "experiment_name": self.experiment_name,
            "timestamp_utc": utc_timestamp(),
            "seed": self.seed,
            "deterministic": self.deterministic,
            "notes": self.notes,
            "config": _jsonable(self.config),
            "git": git,
            "gpu": get_gpu_summary(),
            "software_versions": get_software_versions(),
            "extra": _jsonable(self.extra),
        }

    def save(self, directory: str, filename: Optional[str] = None) -> str:
        os.makedirs(directory, exist_ok=True)
        payload = self.to_dict()
        if filename is None:
            stamp = payload["timestamp_utc"].replace(":", "").replace("-", "")
            filename = f"{stamp}_{_slug(self.experiment_name)}.json"
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return os.path.abspath(path)

    def summary(self) -> str:
        payload = self.to_dict()
        git = payload["git"]
        gpu = payload["gpu"]
        return (
            f"{payload['experiment_name']} | {payload['timestamp_utc']} | "
            f"seed={payload['seed']} | commit={git.get('short_commit')} | "
            f"dirty={git.get('dirty')} | gpu={gpu.get('name', gpu.get('reason'))}"
        )


def start_experiment(
    experiment_name: str,
    config: Optional[Dict[str, Any]] = None,
    seed: int = DEFAULT_SEED,
    deterministic: bool = True,
    repo_root: Optional[str] = None,
    notes: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> ExperimentMetadata:
    """Seed everything and return a ready-to-save metadata record."""
    set_seed(seed=seed, deterministic=deterministic, quiet=True)
    meta = ExperimentMetadata(
        experiment_name=experiment_name,
        config=config or {},
        seed=seed,
        deterministic=deterministic,
        repo_root=repo_root or default_repo_root(),
        notes=notes,
        extra=extra or {},
    )
    print(f"[reproducibility] experiment '{experiment_name}' | seed={seed}")
    return meta


def log_experiment_result(
    meta: ExperimentMetadata,
    metrics: Dict[str, Any],
    directory: str,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist metadata + metrics as one JSON file inside an experiment folder."""
    payload = meta.to_dict()
    payload["metrics"] = _jsonable(metrics)
    if extra:
        payload["extra"].update(_jsonable(extra))
    os.makedirs(directory, exist_ok=True)
    stamp = payload["timestamp_utc"].replace(":", "").replace("-", "")
    path = os.path.join(directory, f"{stamp}_{_slug(meta.experiment_name)}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return os.path.abspath(path)


def default_repo_root() -> str:
    """Repository root inferred from this file's location (<repo>/scripts/...)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _slug(text: str) -> str:
    keep = [c if c.isalnum() or c in "-_" else "_" for c in str(text)]
    return "".join(keep).strip("_") or "experiment"


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of configs/metrics to JSON-serialisable values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if dataclasses.is_dataclass(obj):
        return _jsonable(dataclasses.asdict(obj))
    if isinstance(obj, datetime):
        return obj.isoformat(timespec="seconds")
    if hasattr(obj, "tolist"):  # numpy arrays / scalars
        try:
            return _jsonable(obj.tolist())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _jsonable(vars(obj))
        except Exception:
            pass
    return str(obj)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Seeding / metadata smoke test.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", default=None, help="directory for the metadata JSON")
    args = parser.parse_args(argv)

    applied = set_seed(args.seed, deterministic=True)
    print(json.dumps(applied, indent=2, default=str))

    meta = start_experiment("phase0_environment_check", config=applied, seed=args.seed)
    print(meta.summary())
    if args.out:
        print(f"metadata -> {meta.save(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())