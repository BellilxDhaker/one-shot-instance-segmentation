"""Combination robustness: apply two failures at once, in a fixed order.

Real deployments do not fail one variable at a time: a dim scene is usually
also an awkward angle. This family is a deliberately small stress probe - four
pairs at two levels - not a full cross product (which would be hundreds of
runs per sample and would answer a question nobody asked).

    pair                        component at `mild`   component at `strong`
    viewpoint + illumination    yaw_2 + low_light_2   yaw_3 + low_light_3
    viewpoint + scale           yaw_2 + smaller_2     yaw_3 + smaller_3
    viewpoint + occlusion       yaw_2 + occlude_2     yaw_3 + occlude_3
    illumination + occlusion    low_light_2 + occlude_2   low_light_3 + occlude_3

Each pair is built from the *representative* member of its family - the
severity-graded hardest case, picked a priori before any results existed - so
the combined ladder stays interpretable instead of exploding into every
crossing. `severity` of the combined row is the max of the component
severities (2 for mild, 3 for strong), as specified in the protocol.

Application order is fixed and documented, because order changes the result:

    viewpoint -> scale -> illumination -> occlusion

Geometry first (the crop/zoom must operate on the rotated frame), then
photometry (the occluder must be sampled from the already-illuminated image, so
it is lit like its surroundings and the two effects cannot be told apart).

Component parameters - including the occlusion level's shape assignment - are
recorded per component inside `metadata.json`, so any combined row can be
reproduced without guessing which occluder shape went in.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from augmentation import illumination, occlusion, scale, viewpoint
from augmentation.common import Variant

FAMILY = "combined"

# (pair key, condition, component family names in application order)
PAIRS: Tuple[Tuple[str, str, Tuple[str, str]], ...] = (
    ("vp_illum", "viewpoint+illumination", ("viewpoint", "illumination")),
    ("vp_scale", "viewpoint+scale", ("viewpoint", "scale")),
    ("vp_occl", "viewpoint+occlusion", ("viewpoint", "occlusion")),
    ("illum_occl", "illumination+occlusion", ("illumination", "occlusion")),
)
LEVELS: Tuple[Tuple[str, int], ...] = (("mild", 2), ("strong", 3))

# One representative per family, fixed before any results were seen.
REPRESENTATIVE = {
    "viewpoint": lambda level, sample_index: f"yaw_{level}",
    "scale": lambda level, sample_index: f"smaller_{level}",
    "illumination": lambda level, sample_index: f"low_light_{level}",
    "occlusion": lambda level, sample_index: f"occlude_{level}",
}


def _component_variant(family_name: str, level: int, sample_index: int) -> Variant:
    """Resolve the representative variant of one family at one severity."""
    if family_name == "viewpoint":
        variants = viewpoint.build_variants()
    elif family_name == "scale":
        variants = scale.build_variants()
    elif family_name == "illumination":
        variants = illumination.build_variants()
    elif family_name == "occlusion":
        variants = occlusion.build_variants(sample_index)
    else:
        raise ValueError(f"unknown component family: {family_name!r}")
    wanted = REPRESENTATIVE[family_name](level, sample_index)
    for candidate in variants:
        if candidate.variant == wanted:
            return candidate
    raise ValueError(f"{family_name} has no variant named {wanted!r}")


def build_variants(sample_index: int = 0) -> List[Variant]:
    """The eight combination variants for one sample."""
    variants: List[Variant] = []
    for pair_key, condition, components in PAIRS:
        for level_name, level in LEVELS:
            specs = [_component_variant(family_name, level, sample_index)
                     for family_name in components]
            variants.append(Variant(
                family=FAMILY,
                condition=condition,
                severity=level,
                variant=f"{pair_key}_{level_name}",
                transformation="compose",
                parameters={
                    "level": level_name,
                    "components": [
                        {
                            "family": spec.family,
                            "variant": spec.variant,
                            "condition": spec.condition,
                            "severity": spec.severity,
                            "transformation": spec.transformation,
                            "parameters": dict(spec.parameters),
                        }
                        for spec in specs
                    ],
                },
            ))
    return variants


def _resolve(spec: Dict[str, Any]) -> Variant:
    return Variant(
        family=spec["family"],
        condition=spec["condition"],
        severity=int(spec["severity"]),
        variant=spec["variant"],
        transformation=spec["transformation"],
        parameters=dict(spec["parameters"]),
    )


def apply(array: np.ndarray, mask: np.ndarray, variant: Variant,
          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply the component chain in the documented order."""
    image, ground_truth = array, mask
    applied: List[Dict[str, Any]] = []
    artifacts: Dict[str, np.ndarray] = {}

    for spec in variant.parameters["components"]:
        component = _resolve(spec)
        module = {
            "viewpoint": viewpoint,
            "scale": scale,
            "illumination": illumination,
            "occlusion": occlusion,
        }[component.family]
        child_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        image, ground_truth, info = module.apply(image, ground_truth, component, child_rng)
        for artifact_name, artifact in info.pop("artifacts", {}).items():
            artifacts[artifact_name] = artifact
        applied.append({
            "family": component.family,
            "variant": component.variant,
            "info": info,
        })

    info = {"components": applied}
    if artifacts:
        info["artifacts"] = artifacts
    return image, ground_truth, info
