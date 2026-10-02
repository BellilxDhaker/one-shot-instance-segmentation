"""Feature cache design for DINOv2 (and any other backbone) features.

Motivation
----------
DINOv2 inference is the dominant cost of this project. Perturbation sweeps
(viewpoint, illumination, scale, occlusion) multiply the number of images by
the number of perturbation settings, while Matcher / PerSAM style matching
re-reads the *same* features many times. Extracting features once and storing
them turns the inner loop into a memory load.

Layout
------
    data/cache/<model_family>/<model_tag>/<split>/<image_id>__<key>.pt

Each file is a dict:

    {
      "features": torch.Tensor | np.ndarray,   # the cached feature
      "meta": {
          "cache_version": int,
          "image_id": str, "image_path": str, "split": str,
          "model_family": str, "model_tag": str, "model_config": {...},
          "preprocess": {...}, "extra": {...},
          "dtype": str, "shape": [...], "content_hash": str,
          "created_utc": str, "runtime": {...}
      }
    }

The filename key is a short SHA1 over every input that can change the tensor
(image identity, model id/revision, preprocessing, patch/aggregation options).
That guarantees a cache hit only ever returns features computed with exactly
the same configuration, and a different configuration can never silently reuse
stale tensors.

Phase 0 deliberately does not run DINOv2: `demo_feature_cache()` exercises the
whole pipeline with a dummy tensor so the contract is fixed before any model
code exists.

CLI demo:

    python scripts/cache_utils.py --root /tmp/feature_cache_demo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

CACHE_VERSION = 1
DEFAULT_MODEL_FAMILY = "dinov2"
DEFAULT_MODEL_TAG = "dinov2-base"


# ---------------------------------------------------------------------------
# optional backends - numpy and torch are both supported so the cache can be
# demonstrated before torch is installed.
# ---------------------------------------------------------------------------
def _torch():
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:
        return None


def _numpy():
    try:
        import numpy as np  # noqa: PLC0415

        return np
    except Exception:
        return None


def backend_name() -> str:
    if _torch() is not None:
        return "torch"
    if _numpy() is not None:
        return "numpy"
    raise RuntimeError("Either PyTorch or NumPy is required for the feature cache.")


# ---------------------------------------------------------------------------
# keys and paths
# ---------------------------------------------------------------------------
def _sanitise(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")
    return cleaned[:80] or "unnamed"


def stable_json(payload: Dict[str, Any]) -> str:
    """Deterministic JSON so identical configs always hash identically."""
    return json.dumps(_sortable(payload), sort_keys=True, separators=(",", ":"), default=str)


def _sortable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sortable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_sortable(v) for v in obj]
    return obj


def cache_signature(
    model_family: str = DEFAULT_MODEL_FAMILY,
    model_tag: str = DEFAULT_MODEL_TAG,
    model_config: Optional[Dict[str, Any]] = None,
    preprocess: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """12-hex-character signature of everything that can change a feature."""
    payload = {
        "cache_version": CACHE_VERSION,
        "model_family": model_family,
        "model_tag": model_tag,
        "model_config": model_config or {},
        "preprocess": preprocess or {},
        "extra": extra or {},
    }
    return hashlib.sha1(stable_json(payload).encode("utf-8")).hexdigest()[:12]


def image_identity(
    image_path: Optional[str] = None,
    image_id: Optional[str] = None,
    split: Optional[str] = None,
    transform_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Identity of the *input* image.

    `image_id` should be stable and unique (e.g. "lvis_v1/val/00000042" or a
    content hash). When it is not supplied it is derived from the file path.
    `transform_tag` distinguishes perturbed variants of the same source image
    (e.g. "azimuth+30/elev-10/occ0.3"), so augmented copies get their own entry
    while still pointing at the same source.
    """
    if image_id is None:
        image_id = image_path
    if image_id is None:
        raise ValueError("image_identity() needs an image_path or an image_id.")
    return {
        "image_id": str(image_id),
        "image_path": str(image_path) if image_path else None,
        "split": str(split) if split else None,
        "transform_tag": str(transform_tag) if transform_tag else None,
    }


def build_cache_key(
    identity: Dict[str, Any],
    model_family: str = DEFAULT_MODEL_FAMILY,
    model_tag: str = DEFAULT_MODEL_TAG,
    model_config: Optional[Dict[str, Any]] = None,
    preprocess: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Full cache key: image identity + model configuration."""
    image_part = hashlib.sha1(
        stable_json(
            {
                "image_id": identity.get("image_id"),
                "split": identity.get("split"),
                "transform_tag": identity.get("transform_tag"),
            }
        ).encode("utf-8")
    ).hexdigest()[:10]
    image_slug = _sanitise(identity.get("image_id"))
    config_part = cache_signature(
        model_family=model_family,
        model_tag=model_tag,
        model_config=model_config,
        preprocess=preprocess,
        extra=extra,
    )
    return f"{image_slug}-{image_part}__{config_part}"


def cache_dir(
    cache_root: str,
    model_family: str = DEFAULT_MODEL_FAMILY,
    model_tag: str = DEFAULT_MODEL_TAG,
    split: Optional[str] = None,
    create: bool = True,
) -> str:
    """Directory holding one model's (optionally one split's) cached features."""
    parts = [str(cache_root), _sanitise(model_family), _sanitise(model_tag)]
    if split:
        parts.append(_sanitise(split))
    directory = os.path.join(*parts)
    if create:
        os.makedirs(directory, exist_ok=True)
    return directory


def cache_path(
    cache_root: str,
    identity: Dict[str, Any],
    model_family: str = DEFAULT_MODEL_FAMILY,
    model_tag: str = DEFAULT_MODEL_TAG,
    model_config: Optional[Dict[str, Any]] = None,
    preprocess: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    create_parents: bool = False,
) -> str:
    """Absolute path of the cache file for one (image, model config) pair."""
    key = build_cache_key(
        identity,
        model_family=model_family,
        model_tag=model_tag,
        model_config=model_config,
        preprocess=preprocess,
        extra=extra,
    )
    directory = cache_dir(
        cache_root,
        model_family=model_family,
        model_tag=model_tag,
        split=identity.get("split"),
        create=create_parents,
    )
    extension = ".pt" if _torch() is not None else ".npz"
    return os.path.join(directory, f"{key}{extension}")


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------
def is_cached(
    cache_root: str,
    identity: Dict[str, Any],
    model_family: str = DEFAULT_MODEL_FAMILY,
    model_tag: str = DEFAULT_MODEL_TAG,
    model_config: Optional[Dict[str, Any]] = None,
    preprocess: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    """True if a *usable* cache file exists (a cache miss costs a forward pass)."""
    path = cache_path(
        cache_root,
        identity,
        model_family=model_family,
        model_tag=model_tag,
        model_config=model_config,
        preprocess=preprocess,
        extra=extra,
    )
    return os.path.isfile(path) and os.path.getsize(path) > 0


def cache_status(
    cache_root: str,
    identities: Iterable[Dict[str, Any]],
    model_family: str = DEFAULT_MODEL_FAMILY,
    model_tag: str = DEFAULT_MODEL_TAG,
    model_config: Optional[Dict[str, Any]] = None,
    preprocess: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Coverage of the cache over a list of images: hits, misses, percentages."""
    identities = list(identities)
    results = [
        is_cached(
            cache_root,
            identity,
            model_family=model_family,
            model_tag=model_tag,
            model_config=model_config,
            preprocess=preprocess,
            extra=extra,
        )
        for identity in identities
    ]
    hits = [i for i, hit in zip(identities, results) if hit]
    misses = [i for i, hit in zip(identities, results) if not hit]
    return {
        "cache_root": cache_root,
        "model_family": model_family,
        "model_tag": model_tag,
        "total": len(identities),
        "hits": len(hits),
        "misses": len(misses),
        "hit_rate": round(len(hits) / len(identities), 4) if identities else 0.0,
        "missing_image_ids": [i.get("image_id") for i in misses][:20],
    }


def list_cached_features(
    cache_root: str,
    model_family: str = DEFAULT_MODEL_FAMILY,
    model_tag: str = DEFAULT_MODEL_TAG,
) -> List[str]:
    directory = cache_dir(
        cache_root, model_family=model_family, model_tag=model_tag, create=False
    )
    if not os.path.isdir(directory):
        return []
    files: List[str] = []
    for root, _dirs, names in os.walk(directory):
        for name in names:
            if name.endswith((".pt", ".npz")):
                files.append(os.path.join(root, name))
    return sorted(files)


def cache_size_bytes(cache_root: str) -> int:
    total = 0
    for root, _dirs, names in os.walk(cache_root):
        for name in names:
            if name.endswith((".pt", ".npz")):
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    return total


# ---------------------------------------------------------------------------
# read / write
# ---------------------------------------------------------------------------
def content_hash(array: Any, max_elements: int = 100_000) -> str:
    """Cheap fingerprint of tensor values, for integrity checks."""
    np = _numpy()
    if np is not None:
        values = np.asarray(array.detach().cpu().numpy() if hasattr(array, "detach") else array)
        flat = values.reshape(-1)[:max_elements]
        digest = hashlib.sha1(flat.tobytes()).hexdigest()
        return f"{digest[:12]}-{tuple(values.shape)}"
    digest = hashlib.sha1(repr(array).encode("utf-8")).hexdigest()
    return f"{digest[:12]}-unknown"


def save_features(
    features: Any,
    path: str,
    identity: Dict[str, Any],
    model_family: str = DEFAULT_MODEL_FAMILY,
    model_tag: str = DEFAULT_MODEL_TAG,
    model_config: Optional[Dict[str, Any]] = None,
    preprocess: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Tuple[str, bool]:
    """Persist one feature tensor. Returns (path, was_written).

    Writing is atomic (temp file + rename) so an interrupted run cannot leave a
    truncated cache entry that later reads as a valid hit.
    """
    if os.path.exists(path) and not overwrite:
        return path, False

    torch = _torch()
    np = _numpy()
    meta = {
        "cache_version": CACHE_VERSION,
        "identity": identity,
        "model_family": model_family,
        "model_tag": model_tag,
        "model_config": model_config or {},
        "preprocess": preprocess or {},
        "extra": extra or {},
        "content_hash": content_hash(features),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".pt"):
        if torch is None:
            raise RuntimeError("PyTorch is required to write .pt cache entries.")
        payload = {
            "features": features.detach().cpu(),
            "meta": meta,
            "dtype": str(features.dtype),
            "shape": list(features.shape),
            "device": str(features.device),
        }
        tmp = f"{path}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path, True

    if np is None:
        raise RuntimeError("NumPy is required to write .npz cache entries.")
    meta["dtype"] = str(np.asarray(features).dtype)
    meta["shape"] = list(np.asarray(features).shape)
    tmp = f"{path}.tmp.npz"
    np.savez_compressed(tmp, features=np.asarray(features), meta=json.dumps(meta, default=str))
    os.replace(tmp, path)
    return path, True


def load_features(path: str, map_location: str = "cpu") -> Tuple[Any, Dict[str, Any]]:
    """Load a cache entry -> (features, meta)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if path.endswith(".pt"):
        torch = _torch()
        if torch is None:
            raise RuntimeError("PyTorch is required to read .pt cache entries.")
        payload = torch.load(path, map_location=map_location, weights_only=False)
        return payload["features"], payload.get("meta", {})

    np = _numpy()
    if np is None:
        raise RuntimeError("NumPy is required to read .npz cache entries.")
    with np.load(path, allow_pickle=False) as data:
        features = data["features"]
        meta = json.loads(str(data["meta"]))
    return features, meta


def get_or_load(
    cache_root: str,
    identity: Dict[str, Any],
    extract_fn=None,
    model_family: str = DEFAULT_MODEL_FAMILY,
    model_tag: str = DEFAULT_MODEL_TAG,
    model_config: Optional[Dict[str, Any]] = None,
    preprocess: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Cache-aware entry point used by every feature consumer.

    `extract_fn()` is called only on a miss, so the model is instantiated and
    the forward pass paid for exactly once per (image, configuration).
    """
    path = cache_path(
        cache_root,
        identity,
        model_family=model_family,
        model_tag=model_tag,
        model_config=model_config,
        preprocess=preprocess,
        extra=extra,
        create_parents=True,
    )
    if os.path.isfile(path) and not overwrite:
        features, meta = load_features(path)
        meta = dict(meta)
        meta["cache_hit"] = True
        meta["cache_path"] = path
        return features, meta

    if extract_fn is None:
        raise FileNotFoundError(f"No cached features for {identity} and no extract_fn given.")

    features = extract_fn()
    save_features(
        features,
        path,
        identity,
        model_family=model_family,
        model_tag=model_tag,
        model_config=model_config,
        preprocess=preprocess,
        extra=extra,
        overwrite=overwrite,
    )
    meta = {
        "cache_hit": False,
        "cache_path": path,
        "model_family": model_family,
        "model_tag": model_tag,
        "identity": identity,
        "content_hash": content_hash(features),
    }
    return features, meta


# ---------------------------------------------------------------------------
# demo: prove the contract with a dummy tensor (no DINOv2 involved)
# ---------------------------------------------------------------------------
def dummy_feature(shape: Tuple[int, ...] = (2, 4, 8), seed: int = 0):
    """Deterministic stand-in for a DINOv2 feature map."""
    torch = _torch()
    if torch is not None:
        generator = torch.Generator().manual_seed(int(seed))
        return torch.randn(*shape, generator=generator)
    np = _numpy()
    if np is None:
        raise RuntimeError("PyTorch or NumPy is required for the cache demo.")
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal(shape).astype("float32")


def demo_feature_cache(cache_root: str, verbose: bool = True) -> Dict[str, Any]:
    """End-to-end exercise of the cache with dummy tensors.

    Demonstrates: key uniqueness per image, key change per model config,
    cache hit/miss accounting, persistence to disk, and identical values on
    reload.
    """
    def say(message: str) -> None:
        if verbose:
            print(message)

    os.makedirs(cache_root, exist_ok=True)
    model_config = {"hidden_size": 768, "patch_size": 14, "layers": [-1]}
    preprocess = {"resize": 518, "crop": None, "normalise": "imagenet"}

    identities = [
        image_identity(image_id="demo/0001", image_path="/data/raw/demo/0001.jpg", split="val"),
        image_identity(image_id="demo/0002", image_path="/data/raw/demo/0002.jpg", split="val"),
        image_identity(
            image_id="demo/0001",
            image_path="/data/raw/demo/0001.jpg",
            split="val",
            transform_tag="azimuth+30_occ0.3",
        ),
    ]

    say("[cache] backend: " + backend_name())
    say(f"[cache] root:   {cache_root}")
    for identity in identities:
        say(
            f"[cache] key:    {identity['image_id']}"
            f"{' @' + identity['transform_tag'] if identity['transform_tag'] else ''}"
            f" -> {os.path.basename(cache_path(cache_root, identity, model_config=model_config, preprocess=preprocess))}"
        )

    # 1) cold pass: every image is a miss and gets written
    hits, misses = 0, 0
    for index, identity in enumerate(identities):
        features, meta = get_or_load(
            cache_root,
            identity,
            extract_fn=lambda i=index: dummy_feature(seed=i),
            model_config=model_config,
            preprocess=preprocess,
        )
        if meta["cache_hit"]:
            hits += 1
        else:
            misses += 1
        say(f"[cache] store {identity['image_id']} shape={tuple(features.shape)} hit={meta['cache_hit']}")
    say(f"[cache] cold pass: {hits} hits / {misses} misses")

    # 2) warm pass: the extractor must never be called again
    warm_hits = 0
    for identity in identities:

        def explode(_identity=identity):
            raise AssertionError("extract_fn called although features were cached.")

        features, meta = get_or_load(
            cache_root,
            identity,
            extract_fn=explode,
            model_config=model_config,
            preprocess=preprocess,
        )
        warm_hits += int(meta["cache_hit"])
    say(f"[cache] warm pass: {warm_hits}/{len(identities)} cache hits, extractor never called")

    # 3) a different model configuration must NOT reuse the entry
    other_model_config = {"hidden_size": 1024, "patch_size": 14, "layers": [-1]}
    identity = identities[0]
    features, other_meta = get_or_load(
        cache_root,
        identity,
        extract_fn=lambda: dummy_feature(shape=(2, 4, 8), seed=99),
        model_tag="dinov2-large",
        model_config=other_model_config,
        preprocess=preprocess,
    )
    config_isolated = not other_meta.get("cache_hit", False)
    say(
        f"[cache] config isolation: dinov2-large miss={config_isolated} "
        f"(same image, different model)"
    )

    # 4) reload and compare values
    path = cache_path(
        cache_root, identities[0], model_config=model_config, preprocess=preprocess
    )
    reloaded, meta = load_features(path)
    torch = _torch()
    if torch is not None and torch.is_tensor(reloaded):
        equal = bool(torch.equal(reloaded, dummy_feature(seed=0)))
    else:
        equal = bool((reloaded == dummy_feature(seed=0)).all())
    say(f"[cache] reload {os.path.basename(path)} identical={equal} hash={meta['content_hash']}")

    status = cache_status(
        cache_root, identities, model_config=model_config, preprocess=preprocess
    )
    say(
        f"[cache] coverage: {status['hits']}/{status['total']} "
        f"(hit rate {status['hit_rate']:.0%}), "
        f"{cache_size_bytes(cache_root) / 1024:.1f} KiB on disk"
    )

    return {
        "cache_root": cache_root,
        "cold_pass_hits": hits,
        "cold_pass_misses": misses,
        "warm_pass_hits": warm_hits,
        "config_isolation_verified": config_isolated,
        "reload_identical": equal,
        "status": status,
        "files": list_cached_features(cache_root),
        "size_bytes": cache_size_bytes(cache_root),
        "demo_model_config": model_config,
        "demo_preprocess": preprocess,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Feature cache demo (dummy tensors).")
    parser.add_argument(
        "--root",
        default=os.path.join("/tmp", "one-shot-segmentation", "feature_cache_demo"),
        help="Directory for the demo cache.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = demo_feature_cache(args.root, verbose=not args.quiet)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2, default=str))
    ok = result["reload_identical"] and result["warm_pass_hits"] == result["status"]["total"]
    print(f"\nfeature cache demo: {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())