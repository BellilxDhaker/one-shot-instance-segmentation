"""Project setup: Drive mounting, path resolution, directories, artefact checks.

Everything the notebook needs to know *where things live* lives here, so the
notebook config cell stays small and the logic is testable outside Colab.

Responsibilities
----------------
* resolve `configs/paths.yaml` into concrete absolute paths (${TOKEN} expansion
  plus environment-variable overrides);
* mount Google Drive once and report whether it is available;
* create the project directory tree idempotently (safe to re-run);
* report whether a dataset / checkpoint / feature cache already exists, so
  nothing is downloaded or recomputed twice.

Phase 0 performs no downloads. `DATASETS` is a *declaration* of what the
project will need later; the helpers only inspect what is present.

CLI:

    python scripts/project_setup.py --init --drive-root ./drive_root
    python scripts/project_setup.py --report --drive-root ./drive_root
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATHS_CONFIG = os.path.join(REPO_ROOT, "configs", "paths.yaml")

# Project code packages that later phases will fill in. Present from the start
# so imports and `scripts -m` style commands work immediately.
CODE_PACKAGES = ("models", "matching", "perturbations", "evaluation", "experiments", "visualization")

# Declared datasets - Phase 1 will fill in the download/verification logic.
DATASETS: Dict[str, Dict[str, Any]] = {
    "coco2017_val": {
        "directory": "coco2017",
        "split_dir": "splits/coco2017",
        "expected": ["annotations", "val2017"],
        "description": "COCO instance-segmentation validation subset (baseline evaluation).",
    },
    "lvis_v1_minival": {
        "directory": "lvis_v1",
        "split_dir": "splits/lvis_v1",
        "expected": ["annotations", "val"],
        "description": "LVIS v1 mini-val, long-tail instance segmentation.",
    },
    "custom_easy": {
        "directory": "custom_easy",
        "split_dir": "splits/custom_easy",
        "expected": [],
        "description": "Small curated set of single-instance images (sanity checks).",
    },
}

# Declared checkpoints - downloaded in later phases, only inspected here.
CHECKPOINTS: Dict[str, Dict[str, Any]] = {
    "dinov2_base": {
        "filename": "dinov2_vitb14_pretrain.pth",
        "source": "torch.hub / transformers",
        "used_by": "feature extraction",
    },
    "dinov2_large": {
        "filename": "dinov2_vitl14_pretrain.pth",
        "source": "torch.hub / transformers",
        "used_by": "feature extraction",
    },
    "sam_vit_h": {
        "filename": "sam_vit_h_4b8939.pth",
        "source": "Segment Anything (Meta)",
        "used_by": "SAM baseline (added in the SAM phase)",
    },
    "sam2_base_plus": {
        "filename": "sam2_hiera_base_plus.pt",
        "source": "SAM 2 (Meta)",
        "used_by": "SAM 2 baseline (added in the SAM 2 phase)",
    },
}


# ---------------------------------------------------------------------------
# YAML + token resolution
# ---------------------------------------------------------------------------
def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required. Install it with: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_paths_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Raw (unexpanded) contents of `configs/paths.yaml`.

    Useful for reading declarative blocks such as `resources:` (VRAM / RAM /
    disk thresholds) without resolving every path.
    """
    return _load_yaml(path or DEFAULT_PATHS_CONFIG)


_TOKEN_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_tokens(value: Any, tokens: Dict[str, str], _depth: int = 0) -> Any:
    """Recursively replace ${TOKEN} placeholders. Unresolved tokens are kept."""
    if _depth > 10:
        raise ValueError("Too many nested ${...} expansions; check paths.yaml.")
    if isinstance(value, dict):
        return {k: expand_tokens(v, tokens, _depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_tokens(v, tokens, _depth + 1) for v in value]
    if not isinstance(value, str):
        return value

    def replace(match: "re.Match[str]") -> str:
        key = match.group(1)
        return str(tokens.get(key, match.group(0)))

    expanded = _TOKEN_RE.sub(replace, value)
    if expanded != value:
        return expand_tokens(expanded, tokens, _depth + 1)
    return expanded


def _token_map(raw: Dict[str, Any], overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    tokens = dict(raw.get("tokens") or {})
    tokens["REPO_ROOT"] = str(raw.get("__repo_root__", REPO_ROOT))
    tokens["PROJECT_NAME"] = str(raw.get("project", {}).get("name", "one-shot-segmentation"))
    tokens.setdefault("PROJECT_DIRNAME", tokens["PROJECT_NAME"])
    for key, value in (overrides or {}).items():
        if value:
            tokens[key] = str(value)
    # Environment variables win over the file (OSS_DRIVE_ROOT etc.).
    for key in list(tokens):
        env_value = os.environ.get(f"OSS_{key}")
        if env_value:
            tokens[key] = env_value
    return {k: str(v) for k, v in tokens.items()}


@dataclass
class ProjectPaths:
    """Resolved absolute paths for every persistent artefact."""

    repo_root: str
    paths_config: str
    project_name: str
    drive_mount: str
    drive_root: str
    mydrive: str
    data_root: str
    raw_root: str
    processed_root: str
    cache_root: str
    splits_root: str
    checkpoints_root: str
    results_root: str
    results_baseline: str
    results_robustness: str
    results_proposed: str
    results_ablation: str
    results_final: str
    logs_root: str
    env_reports_root: str
    runtime_root: str
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- derived helpers ---------------------------------------------------
    def all_directories(self) -> Dict[str, str]:
        """Persistent directories the notebook guarantees to exist on Drive."""
        directories = {
            key: value
            for key, value in asdict(self).items()
            if key.endswith("_root") and isinstance(value, str)
        }
        directories.pop("repo_root", None)
        directories.pop("runtime_root", None)  # /content is recreated by git clone
        directories.update(self.results_directories())
        return directories

    def results_directories(self) -> Dict[str, str]:
        return {
            "results_baseline": self.results_baseline,
            "results_robustness": self.results_robustness,
            "results_proposed": self.results_proposed,
            "results_ablation": self.results_ablation,
            "results_final": self.results_final,
        }

    def code_directories(self) -> Dict[str, str]:
        return {package: os.path.join(self.repo_root, package) for package in CODE_PACKAGES}

    def result_dir_for(self, experiment_kind: str) -> str:
        """`baseline` | `robustness` | `proposed` | `ablation` | `final`."""
        mapping = {
            "baseline": self.results_baseline,
            "robustness": self.results_robustness,
            "proposed": self.results_proposed,
            "ablation": self.results_ablation,
            "final": self.results_final,
        }
        if experiment_kind not in mapping:
            raise KeyError(
                f"Unknown results group '{experiment_kind}'. "
                f"Choose one of {sorted(mapping)}."
            )
        return mapping[experiment_kind]

    def cache_dir_for(self, model_family: str, model_tag: str) -> str:
        """Feature-cache directory for one backbone configuration."""
        return os.path.join(self.cache_root, model_family, model_tag)

    def checkpoint_path(self, name: str) -> str:
        return os.path.join(self.checkpoints_root, CHECKPOINTS[name]["filename"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_paths(
    paths_config: Optional[str] = None,
    token_overrides: Optional[Dict[str, str]] = None,
    path_overrides: Optional[Dict[str, str]] = None,
    local_root: Optional[str] = None,
) -> ProjectPaths:
    """Load `paths.yaml` and return fully resolved absolute paths.

    Args:
        paths_config: path to the YAML file (default: repo `configs/paths.yaml`).
        token_overrides: notebook-level values, e.g.
            `{"DRIVE_FOLDER": "one-shot-segmentation", "DRIVE_MOUNT": "/content/drive"}`.
        path_overrides: direct per-path overrides, e.g. `{"DATA_ROOT": ...}`
            (upper-cased keys, mirroring the YAML section/key).
        local_root: when Google Drive is unavailable, redirect the persistent
            tree here (used by local testing and by the notebook fallback).
    """
    config_path = paths_config or DEFAULT_PATHS_CONFIG
    raw = _load_yaml(config_path)
    raw["__repo_root__"] = REPO_ROOT

    token_overrides = dict(token_overrides or {})
    if local_root:
        token_overrides["DRIVE_ROOT"] = os.path.abspath(local_root)

    tokens = _token_map(raw, token_overrides)
    resolved = expand_tokens(raw, tokens)
    paths_cfg = resolved.get("paths", {})

    drive_mount = str(token_overrides.get("DRIVE_MOUNT") or tokens["DRIVE_MOUNT"])
    mydrive = str(resolved.get("drive", {}).get("mydrive", os.path.join(drive_mount, "MyDrive")))
    drive_root = str(
        token_overrides.get("DRIVE_ROOT")
        or os.environ.get("OSS_DRIVE_ROOT")
        or resolved.get("drive", {}).get("root", os.path.join(mydrive, tokens["DRIVE_FOLDER"]))
    )
    if local_root:
        mydrive = drive_root

    # A locally rooted tree (Drive unavailable, or local testing) uses native
    # separators; Colab paths keep their POSIX form.
    local_tree = bool(re.match(r"^[A-Za-z]:", drive_root))

    def value(key: str, default_tail: str) -> str:
        override = (path_overrides or {}).get(key.upper())
        if override:
            return os.path.abspath(str(override))
        configured = str(paths_cfg.get(key) or os.path.join(drive_root, *default_tail.split("/")))
        return configured.replace("/", os.sep) if local_tree else configured

    runtime_cfg = resolved.get("runtime", {})
    runtime_root = str(
        runtime_cfg.get("clone_root")
        or os.path.join(drive_mount, os.path.basename(str(
            resolved.get("project", {}).get("repo_dirname", "one-shot-segmentation")
        )))
    )

    return ProjectPaths(
        repo_root=REPO_ROOT,
        paths_config=os.path.abspath(config_path),
        project_name=str(resolved.get("project", {}).get("name", "one-shot-segmentation")),
        drive_mount=drive_mount,
        drive_root=drive_root,
        mydrive=mydrive,
        data_root=value("data_root", "data"),
        raw_root=value("raw_root", "data/raw"),
        processed_root=value("processed_root", "data/processed"),
        cache_root=value("cache_root", "data/cache"),
        splits_root=value("splits_root", "data/splits"),
        checkpoints_root=value("checkpoints_root", "checkpoints"),
        results_root=value("results_root", "results"),
        results_baseline=value("results_baseline", "results/baseline"),
        results_robustness=value("results_robustness", "results/robustness"),
        results_proposed=value("results_proposed", "results/proposed"),
        results_ablation=value("results_ablation", "results/ablation"),
        results_final=value("results_final", "results/final"),
        logs_root=value("logs_root", "logs"),
        env_reports_root=value("env_reports_root", "results/environment"),
        runtime_root=runtime_root,
        extra={
            "code_packages": list(CODE_PACKAGES),
            "datasets": DATASETS,
            "checkpoints": CHECKPOINTS,
            "hf_home": resolved.get("runtime", {}).get("hf_home"),
            "torch_home": resolved.get("runtime", {}).get("torch_home"),
        },
    )


# ---------------------------------------------------------------------------
# Google Drive
# ---------------------------------------------------------------------------
def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401,PLC0415

        return True
    except Exception:
        return False


def mount_google_drive(mount_point: str = "/content/drive", timeout_seconds: int = 300) -> Dict[str, Any]:
    """Mount Google Drive, but only once.

    Returns a dict with `available`, `mount_point`, `mydrive`, `mounted_now` and
    `message`. Safe to call on every notebook run: an already-mounted Drive is
    detected and nothing is re-mounted.
    """
    result: Dict[str, Any] = {
        "available": False,
        "mount_point": mount_point,
        "mydrive": os.path.join(mount_point, "MyDrive"),
        "mounted_now": False,
        "message": "",
    }

    if os.path.isdir(result["mydrive"]):
        result["available"] = True
        result["message"] = f"Already mounted at {mount_point}."
        return result

    if not in_colab():
        result["message"] = (
            "Not running in Colab: mount Google Drive manually, e.g. "
            "`rclone` / `google-drive-ocamlfuse`, or set DRIVE_ROOT to a local path."
        )
        return result

    try:
        from google.colab import drive  # noqa: PLC0415

        # force_remount=False keeps an existing session from being interrupted.
        drive.mount(mount_point, force_remount=False, timeout_ms=timeout_seconds * 1000)
        result["mounted_now"] = True
        result["available"] = os.path.isdir(result["mydrive"])
        result["message"] = f"Mounted Google Drive at {mount_point}."
    except Exception as exc:
        result["message"] = (
            f"Drive mount failed: {type(exc).__name__}: {exc}. "
            "Re-run the Drive cell and authorise the account if prompted."
        )
    return result


def drive_free_space_gb(path: str) -> Optional[float]:
    if not os.path.isdir(path):
        return None
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    return round(usage.free / 1024**3, 2)


# ---------------------------------------------------------------------------
# directories
# ---------------------------------------------------------------------------
def ensure_directories(paths: ProjectPaths, include_code_packages: bool = True) -> Dict[str, Any]:
    """Create every directory. Idempotent: returns created vs already-existing."""
    created: List[str] = []
    existing: List[str] = []
    missing_drive: List[str] = []

    drive_ready = os.path.isdir(paths.drive_root)

    for name, directory in paths.all_directories().items():
        if not drive_ready and name != "repo_root" and directory.startswith(paths.drive_mount):
            missing_drive.append(directory)
            continue
        if os.path.isdir(directory):
            existing.append(directory)
        else:
            os.makedirs(directory, exist_ok=True)
            created.append(directory)

    if include_code_packages:
        for package, directory in paths.code_directories().items():
            if os.path.isdir(directory):
                existing.append(directory)
            else:
                os.makedirs(directory, exist_ok=True)
                created.append(directory)

    return {
        "drive_ready": drive_ready,
        "created": created,
        "existing": existing,
        "skipped_due_to_unmounted_drive": missing_drive,
        "total": len(created) + len(existing),
    }


def format_paths_table(paths: ProjectPaths) -> str:
    rows: List[Tuple[str, str]] = [
        ("PROJECT_NAME", paths.project_name),
        ("REPO_ROOT", paths.repo_root),
        ("DRIVE_ROOT", paths.drive_root),
    ]
    seen = {key.upper() for key, _ in rows}
    for key, directory in paths.all_directories().items():
        if key.upper() not in seen:
            rows.append((key.upper(), directory))
            seen.add(key.upper())

    width = max(len(key) for key, _ in rows)
    rule = "-" * (width + 62)
    lines = [rule, f"{'PATH':<{width}} : VALUE", rule]
    lines += [f"{key:<{width}} : {value}" for key, value in rows]
    lines.append(rule)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# artefact existence checks (the "do not download twice" helpers)
# ---------------------------------------------------------------------------
def dataset_dir(name: str, paths: ProjectPaths) -> str:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(DATASETS)}")
    return os.path.join(paths.raw_root, DATASETS[name]["directory"])


def dataset_exists(name: str, paths: ProjectPaths) -> bool:
    """True when the dataset directory and all its expected subfolders exist."""
    try:
        root = dataset_dir(name, paths)
    except KeyError:
        return False
    if not os.path.isdir(root):
        return False
    return all(
        os.path.exists(os.path.join(root, sub)) for sub in DATASETS[name].get("expected", [])
    )


def describe_datasets(paths: ProjectPaths) -> List[Dict[str, Any]]:
    """Presence report for every declared dataset (no downloads)."""
    report = []
    for name, spec in DATASETS.items():
        root = dataset_dir(name, paths)
        report.append(
            {
                "name": name,
                "path": root,
                "exists": dataset_exists(name, paths),
                "missing_subdirs": [
                    sub
                    for sub in spec.get("expected", [])
                    if not os.path.exists(os.path.join(root, sub))
                ],
                "size_gb": _directory_size_gb(root),
                "description": spec.get("description", ""),
                "split_file": os.path.join(paths.splits_root, f"{name}.txt"),
                "split_file_exists": os.path.isfile(
                    os.path.join(paths.splits_root, f"{name}.txt")
                ),
            }
        )
    return report


def checkpoint_exists(name: str, paths: ProjectPaths) -> bool:
    if name not in CHECKPOINTS:
        return False
    return os.path.isfile(paths.checkpoint_path(name))


def describe_checkpoints(paths: ProjectPaths) -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            "path": paths.checkpoint_path(name),
            "exists": checkpoint_exists(name, paths),
            "source": spec.get("source", ""),
            "used_by": spec.get("used_by", ""),
        }
        for name, spec in CHECKPOINTS.items()
    ]


def _directory_size_gb(path: str, sample_limit: int = 2000) -> Optional[float]:
    """Cheap size estimate: sums up to `sample_limit` files, then extrapolates."""
    if not os.path.isdir(path):
        return None
    total = 0
    count = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
            count += 1
            if count >= sample_limit:
                break
        if count >= sample_limit:
            break
    if count == 0:
        return 0.0
    estimated = total / count * max(1, count)
    return round(estimated / 1024**3, 3)


# ---------------------------------------------------------------------------
# repository
# ---------------------------------------------------------------------------
def clone_or_update_repo(
    repo_url: str,
    destination: str,
    branch: str = "main",
    token: Optional[str] = None,
    update: bool = False,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Clone the project repository, or fast-forward an existing clone.

    Idempotent: if `destination` already contains a git checkout the clone is
    skipped unless `update=True`, in which case `git pull --ff-only` runs.
    """
    result: Dict[str, Any] = {
        "repo_url": repo_url,
        "destination": destination,
        "branch": branch,
        "action": "none",
        "message": "",
    }

    if os.path.isdir(os.path.join(destination, ".git")):
        result["action"] = "already_present"
        if update:
            pull = _git(["pull", "--ff-only", "origin", branch], cwd=destination, timeout=timeout)
            result["action"] = "updated"
            result["message"] = pull
        else:
            head = _git(["rev-parse", "--short", "HEAD"], cwd=destination)
            result["message"] = f"Existing checkout kept at {head}."
        return result

    url = repo_url
    if token and repo_url.startswith("https://github.com/"):
        url = repo_url.replace("https://github.com/", f"https://{token}@github.com/")
        result["message"] = "Used a GitHub token from Colab secrets for cloning. "
        result["message"] += "Remove it from the remote URL before pushing."

    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    output = _git(["clone", "--branch", branch, "--depth", "1", url, destination], timeout=timeout)
    result["action"] = "cloned" if os.path.isdir(os.path.join(destination, ".git")) else "failed"
    result["message"] = _redact(output)
    if result["action"] == "cloned":
        _git(["remote", "set-url", "origin", repo_url], cwd=destination)
    return result


def _git(args: List[str], cwd: Optional[str] = None, timeout: int = 120) -> str:
    try:
        out = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return out.stdout.decode("utf-8", errors="replace").strip()
    except Exception as exc:
        return f"git {' '.join(args)} failed: {type(exc).__name__}: {exc}"


def _redact(text: str) -> str:
    return re.sub(r"https://[^/@\s]+@github\.com", "https://<token>@github.com", text or "")


def add_repo_to_sys_path(repo_root: Optional[str] = None) -> str:
    """Make `scripts`, `models`, ... importable from notebook cells."""
    root = os.path.abspath(repo_root or REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def set_runtime_env(paths: ProjectPaths) -> Dict[str, str]:
    """Point HF/torch caches and the OSS_* path overrides at the right places."""
    env = {}
    if paths.extra.get("hf_home"):
        os.environ.setdefault("HF_HOME", paths.extra["hf_home"])
        env["HF_HOME"] = os.environ["HF_HOME"]
    if paths.extra.get("torch_home"):
        os.environ.setdefault("TORCH_HOME", paths.extra["torch_home"])
        env["TORCH_HOME"] = os.environ["TORCH_HOME"]
    os.environ["OSS_DRIVE_ROOT"] = paths.drive_root
    os.environ["OSS_PATHS_CONFIG"] = paths.paths_config
    env["OSS_DRIVE_ROOT"] = paths.drive_root
    env["OSS_PATHS_CONFIG"] = paths.paths_config
    return env


# ---------------------------------------------------------------------------
# reports / CLI
# ---------------------------------------------------------------------------
def setup_report(paths: ProjectPaths) -> Dict[str, Any]:
    return {
        "paths": paths.to_dict(),
        "paths_table": format_paths_table(paths),
        "drive": {
            "mount_point": paths.drive_mount,
            "available": os.path.isdir(paths.mydrive),
            "free_space_gb": drive_free_space_gb(paths.mydrive),
        },
        "datasets": describe_datasets(paths),
        "checkpoints": describe_checkpoints(paths),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--drive-root", default=None, help="override the resolved Drive root")
    parser.add_argument("--init", action="store_true", help="create the directory tree")
    parser.add_argument("--report", action="store_true", help="print paths + artefact status")
    parser.add_argument("--json", default=None, help="write the setup report as JSON")
    args = parser.parse_args(argv)

    overrides = {"DRIVE_ROOT": args.drive_root} if args.drive_root else None
    paths = resolve_paths(paths_config=args.paths_config, token_overrides=overrides)
    print(format_paths_table(paths))

    if args.init:
        result = ensure_directories(paths)
        print(
            f"\n[init] drive_ready={result['drive_ready']} "
            f"created={len(result['created'])} existing={len(result['existing'])}"
        )
        for directory in result["created"]:
            print(f"  created : {directory}")
        if result["skipped_due_to_unmounted_drive"]:
            print(
                "  warning : Google Drive is not mounted, so persistent directories "
                "were not created (mount Drive first)."
            )

    if args.report or args.json:
        report = setup_report(paths)
        print("\nDatasets:")
        for item in report["datasets"]:
            print(f"  [{'x' if item['exists'] else ' '}] {item['name']:<18} {item['path']}")
        print("Checkpoints:")
        for item in report["checkpoints"]:
            print(f"  [{'x' if item['exists'] else ' '}] {item['name']:<18} {item['path']}")
        if args.json:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, default=str)
            print(f"\nSetup report written to {os.path.abspath(args.json)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())