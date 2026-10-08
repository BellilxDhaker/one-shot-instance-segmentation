"""Illumination robustness: severity-graded photometric changes to the target.

Only the target image changes. The reference image and the ground-truth mask are
byte-identical to the benchmark sample, because the question is whether the
system still recognises the object when the *photographs* differ, not whether it
survives a moved annotation.

Six ladders, each with a stated level 0 (the unchanged control lives in
`data/processed/control/`):

| condition       | transformation | level 1        | level 2        | level 3         |
| --------------- | -------------- | -------------- | -------------- | --------------- |
| `brightness_up` | gain           | x1.15          | x1.35          | x1.60           |
| `brightness_down` | gain         | x0.80          | x0.60          | x0.40           |
| `contrast_up`   | contrast       | x1.25          | x1.50          | x2.00           |
| `contrast_down` | contrast       | x0.75          | x0.55          | x0.40           |
| `warm` / `cool` | white balance  | +/-8%          | +/-16%         | -               |
| `low_light`     | gamma + noise  | g1.6, s6       | g2.6, s11      | g4.5, s18       |

The gains are applied in linear 8-bit space and clipped, which is what a
camera's exposure and contrast controls do at the extremes; `low_light` applies
a gamma curve (darker mid-tones, preserved highlights) plus additive Gaussian
noise as a first-order model of sensor noise in a short exposure. It is an
approximation, stated as one, not a calibrated camera simulation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from augmentation.common import Variant

FAMILY = "illumination"

BRIGHTNESS_UP = (1.15, 1.35, 1.60)
BRIGHTNESS_DOWN = (0.80, 0.60, 0.40)
CONTRAST_UP = (1.25, 1.50, 2.00)
CONTRAST_DOWN = (0.75, 0.55, 0.40)
TEMPERATURE_STRENGTH = (0.08, 0.16)
LOW_LIGHT = ((1.6, 6.0), (2.6, 11.0), (4.5, 18.0))


def _luma(array: np.ndarray) -> float:
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    return float((array.astype(np.float64).mean(axis=-1) * weights).sum())


def brightness(array: np.ndarray, gain: float) -> np.ndarray:
    """Multiply every channel: `gain > 1` overexposes, `gain < 1` underexposes."""
    return np.clip(array.astype(np.float32) * float(gain), 0.0, 255.0).astype(np.uint8)


def contrast(array: np.ndarray, factor: float) -> np.ndarray:
    """Scale deviations from the image mean, then clip."""
    mean = float(array.mean())
    scaled = (array.astype(np.float32) - mean) * float(factor) + mean
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def color_temperature(array: np.ndarray, direction: str, strength: float) -> np.ndarray:
    """Shift the red/blue balance and renormalise to keep mean luminance fixed.

    Keeping the luminance fixed matters: without it a "warm" variant would also
    be a "slightly brighter" variant, and the experiment would test two things
    at once.
    """
    if direction not in ("warm", "cool"):
        raise ValueError(f"direction must be 'warm' or 'cool', got {direction!r}")
    strength = float(strength)
    gains = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if direction == "warm":
        gains[0] += strength
        gains[2] -= strength
    else:
        gains[2] += strength
        gains[0] -= strength
    luminance_gain = float((gains * np.array([0.299, 0.587, 0.114])).sum())
    gains /= luminance_gain
    return np.clip(array.astype(np.float32) * gains, 0.0, 255.0).astype(np.uint8)


def low_light(array: np.ndarray, gamma: float, noise_sigma: float,
              rng: np.random.Generator) -> np.ndarray:
    """Darken with a gamma curve, then add Gaussian noise as sensor approximation."""
    normalised = array.astype(np.float32) / 255.0
    darkened = np.power(np.clip(normalised, 0.0, 1.0), float(gamma))
    noisy = darkened * 255.0 + rng.normal(0.0, float(noise_sigma), size=darkened.shape)
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def build_variants() -> List[Variant]:
    """Every illumination variant, in severity order."""
    variants: List[Variant] = []
    for index, gain in enumerate(BRIGHTNESS_UP, start=1):
        variants.append(Variant(FAMILY, "brightness_up", index, f"brightness_up_{index}",
                                "brightness", {"gain": gain}))
    for index, gain in enumerate(BRIGHTNESS_DOWN, start=1):
        variants.append(Variant(FAMILY, "brightness_down", index, f"brightness_down_{index}",
                                "brightness", {"gain": gain}))
    for index, factor in enumerate(CONTRAST_UP, start=1):
        variants.append(Variant(FAMILY, "contrast_up", index, f"contrast_up_{index}",
                                "contrast", {"factor": factor}))
    for index, factor in enumerate(CONTRAST_DOWN, start=1):
        variants.append(Variant(FAMILY, "contrast_down", index, f"contrast_down_{index}",
                                "contrast", {"factor": factor}))
    for index, strength in enumerate(TEMPERATURE_STRENGTH, start=1):
        for direction in ("warm", "cool"):
            variants.append(Variant(FAMILY, direction, index, f"{direction}_{index}",
                                    "temperature",
                                    {"direction": direction, "strength": strength}))
    for index, (gamma, sigma) in enumerate(LOW_LIGHT, start=1):
        variants.append(Variant(FAMILY, "low_light", index, f"low_light_{index}",
                                "low_light", {"gamma": gamma, "noise_sigma": sigma}))
    return variants


def apply(array: np.ndarray, mask: np.ndarray, variant: Variant,
          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply one illumination variant. The mask always comes back unchanged."""
    parameters = variant.parameters
    name = variant.transformation
    if name == "brightness":
        transformed = brightness(array, float(parameters["gain"]))
    elif name == "contrast":
        transformed = contrast(array, float(parameters["factor"]))
    elif name == "temperature":
        transformed = color_temperature(array, str(parameters["direction"]),
                                        float(parameters["strength"]))
    elif name == "low_light":
        transformed = low_light(array, float(parameters["gamma"]),
                                float(parameters["noise_sigma"]), rng)
    else:
        raise ValueError(f"unknown illumination transformation: {name!r}")
    info = {
        "mean_before": round(float(array.mean()), 3),
        "mean_after": round(float(transformed.mean()), 3),
    }
    return transformed, mask, info
