"""Registry: one table from family name to variant builder and apply function.

The rest of the tooling (`tools/build_robustness.py`, the runner, the analyzer)
never imports a family module directly - it asks here. That keeps the protocol
order, the per-sample variation rules and the dispatch in exactly one place,
so a new family is one row in `FAMILY_MODULES` instead of a new branch
scattered across the tools.

The control family is not a module: it is an identity variant whose "apply" is a
no-op, exposed through the same interface so the builder and the runner can
treat every family uniformly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from augmentation import combined, illumination, occlusion, scale, viewpoint
from augmentation.common import FAMILIES, Variant, derive_seed

# Per-sample variation lives here: these two families need to know which sample
# they are building for (the occluder shape schedule, the pair parameters).
PER_SAMPLE_FAMILIES = frozenset({"occlusion", "combined"})

_BUILDERS: Dict[str, Callable[..., List[Variant]]] = {
    "illumination": illumination.build_variants,
    "scale": scale.build_variants,
    "occlusion": occlusion.build_variants,
    "viewpoint": viewpoint.build_variants,
    "combined": combined.build_variants,
}

_APPLIERS: Dict[str, Callable[..., Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]] = {
    "illumination": illumination.apply,
    "scale": scale.apply,
    "occlusion": occlusion.apply,
    "viewpoint": viewpoint.apply,
    "combined": combined.apply,
}

CONTROL_VARIANT = Variant(
    family="control",
    condition="identity",
    severity=0,
    variant="identity",
    transformation="identity",
    parameters={},
)


def sample_index(sample_id: str) -> int:
    """Numeric suffix of a benchmark id (`sample_0042 -> 42`); drives the
    per-sample schedules so they never depend on build order."""
    digits = "".join(ch for ch in sample_id if ch.isdigit())
    return int(digits) if digits else 0


def build_variants(family: str, sample_index: int = 0) -> List[Variant]:
    """All variants of one family, in severity order."""
    if family == "control":
        return [CONTROL_VARIANT]
    if family not in _BUILDERS:
        raise KeyError(f"unknown robustness family: {family!r}")
    if family in PER_SAMPLE_FAMILIES:
        return _BUILDERS[family](sample_index)
    return _BUILDERS[family]()


def apply_variant(family: str, image: np.ndarray, mask: np.ndarray,
                  variant: Variant,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray,
                                                      Dict[str, Any]]:
    """Apply one variant; the control returns its inputs untouched."""
    if family == "control":
        return image, mask, {"identity": True}
    if family not in _APPLIERS:
        raise KeyError(f"unknown robustness family: {family!r}")
    return _APPLIERS[family](image, mask, variant, rng)


def variants_for_sample(sample_id: str) -> List[Variant]:
    """Every variant one sample must undergo, in protocol order."""
    index = sample_index(sample_id)
    variants: List[Variant] = []
    for family in FAMILIES:
        for variant in build_variants(family, index):
            variants.append(variant)
    return variants


def variant_seed(base_seed: int, sample_id: str, variant: Variant) -> int:
    """The stable integer seed for one (sample, variant) pair; recorded in the
    variant metadata so a rebuild can be verified without re-running it."""
    return derive_seed(base_seed, sample_id, variant.family, variant.variant)


def variant_rng(base_seed: int, sample_id: str, variant: Variant) -> np.random.Generator:
    """The deterministic RNG for one (sample, variant) pair."""
    return np.random.default_rng(variant_seed(base_seed, sample_id, variant))
