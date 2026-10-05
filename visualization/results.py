"""Step 8 - visualising a one-shot result.

`plot_one_shot_result` renders a single row of panels that tell the whole story
of a run: the reference image with its mask, the similarity map, the localised
candidates with their boxes, the final mask, and an optional ground-truth
overlay for the evaluation examples. `save_result_figure` writes the same figure
to disk, and `plot_feature_grid` shows where the encoder's patches actually sit,
which is the quickest way to see whether the grid geometry is sane.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from models.feature_encoder import ImageInput, to_pil_image


def _figure(nrows: int = 1, ncols: int = 1, figsize: Optional[Tuple[float, float]] = None):
    import matplotlib.pyplot as plt

    return plt.subplots(nrows, ncols, figsize=figsize or (4.0 * ncols, 3.4 * nrows), squeeze=False)


def _show_image(axis, image: ImageInput) -> None:
    axis.imshow(np.asarray(to_pil_image(image)))
    axis.set_axis_off()


def _overlay_mask(
    axis,
    image: ImageInput,
    mask: Optional[np.ndarray],
    colour: str = "#ff2d55",
    alpha: float = 0.45,
) -> None:
    array = np.asarray(to_pil_image(image))
    axis.imshow(array)
    if mask is not None and np.asarray(mask).any():
        coloured = np.zeros((*np.asarray(mask).shape, 4), dtype=np.float32)
        rgb = np.array(_to_rgb(colour), dtype=np.float32) / 255.0
        coloured[..., :3] = rgb
        coloured[..., 3] = np.asarray(mask, dtype=np.float32) * alpha
        axis.imshow(coloured)
    axis.set_axis_off()


def _to_rgb(colour: str) -> Tuple[int, int, int]:
    from matplotlib.colors import to_rgb

    return tuple(int(channel * 255) for channel in to_rgb(colour))


def plot_one_shot_result(
    reference_image: ImageInput,
    reference_mask: np.ndarray,
    target_image: ImageInput,
    result: Any,
    ground_truth: Optional[np.ndarray] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = True,
):
    """Draw the panels for one `OneShotResult` and return the matplotlib figure."""
    ncols = 5 if ground_truth is not None else 4
    figure, axes = _figure(1, ncols, figsize=figsize or (4.0 * ncols, 3.6))

    _overlay_mask(axes[0][0], reference_image, reference_mask, colour="#00d1b2")
    axes[0][0].set_title("reference + mask", fontsize=10)

    _show_image(axes[0][1], target_image)
    axes[0][1].set_title("target", fontsize=10)

    similarity = np.asarray(result.similarity_map, dtype=np.float32)
    handle = axes[0][2].imshow(similarity, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0][2].set_axis_off()
    axes[0][2].set_title(f"similarity (peak {similarity.max():.2f})", fontsize=10)
    figure.colorbar(handle, ax=axes[0][2], fraction=0.046, pad=0.04)

    _show_image(axes[0][3], target_image)
    candidates = list(getattr(result, "candidates", ()) or ())
    for rank, candidate in enumerate(candidates):
        x1, y1, x2, y2 = candidate.box
        axes[0][3].add_patch(
            _rectangle((x1, y1), x2 - x1, y2 - y1, edge="#ffd166" if rank else "#ff2d55", label=f"C{rank + 1}")
        )
    if getattr(result, "candidate", None) is not None:
        x1, y1, x2, y2 = result.candidate.box
        axes[0][3].add_patch(_rectangle((x1, y1), x2 - x1, y2 - y1, edge="#ff2d55", label="selected"))
    confidence = float(getattr(result, "confidence", 0.0))
    axes[0][3].set_title(f"candidates (conf {confidence:.2f})", fontsize=10)
    if candidates:
        axes[0][3].legend(loc="lower right", fontsize=7)

    if ground_truth is not None:
        _overlay_mask(axes[0][4], target_image, result.predicted_mask, colour="#ff2d55", alpha=0.45)
        truth_box = _bounding_box(ground_truth)
        if truth_box is not None:
            x1, y1, x2, y2 = truth_box
            axes[0][4].add_patch(_rectangle((x1, y1), x2 - x1, y2 - y1, edge="#22c55e", label="ground truth"))
        if result.predicted_mask is not None:
            predicted = np.asarray(result.predicted_mask, dtype=bool)
            truth = np.asarray(ground_truth, dtype=bool)
            intersection = float(np.logical_and(predicted, truth).sum())
            union = float(np.logical_or(predicted, truth).sum())
            iou = intersection / union if union else 0.0
        else:
            iou = 0.0
        axes[0][4].set_title(f"prediction (magenta) vs GT (green) - IoU {iou:.2f}", fontsize=10)
        axes[0][4].legend(loc="lower right", fontsize=7)

    if getattr(result, "predicted_mask", None) is None:
        for axis in axes[0][3:]:
            axis.text(
                0.5, 0.5, "no candidate", ha="center", va="center", color="white",
                transform=axis.transAxes, fontsize=9,
            )

    runtime = getattr(result, "runtime", {}) or {}
    figure.suptitle(
        f"one-shot segmentation - total {float(runtime.get('total', 0.0)):.2f} s",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return figure


def _rectangle(origin: Tuple[float, float], width: float, height: float, edge: str, label: str = ""):
    from matplotlib.patches import Rectangle

    return Rectangle(origin, width, height, fill=False, edgecolor=edge, linewidth=2.0, label=label)


def _bounding_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def plot_feature_grid(
    features: Any,
    image: Optional[ImageInput] = None,
    figsize: Tuple[float, float] = (9.0, 4.2),
    show: bool = True,
):
    """Overlay the encoder patch grid on the image to check the geometry."""
    figure, axes = _figure(1, 2, figsize=figsize)
    if image is not None:
        _show_image(axes[0][0], image)
        axes[0][0].set_title("input", fontsize=10)
    grid_h, grid_w = features.grid_size
    height, width = features.input_size
    for row in range(grid_h + 1):
        y = row * height / grid_h
        axes[0][1].plot([0, width], [y, y], color="#38bdf8", linewidth=0.4)
    for col in range(grid_w + 1):
        x = col * width / grid_w
        axes[0][1].plot([x, x], [0, height], color="#38bdf8", linewidth=0.4)
    axes[0][1].set_xlim(0, width)
    axes[0][1].set_ylim(height, 0)
    axes[0][1].set_axis_off()
    axes[0][1].set_title(f"patch grid {grid_w} x {grid_h} (patch {features.patch_size}px)", fontsize=10)
    figure.tight_layout()
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return figure


def save_result_figure(
    path: str,
    reference_image: ImageInput,
    reference_mask: np.ndarray,
    target_image: ImageInput,
    result: Any,
    ground_truth: Optional[np.ndarray] = None,
    dpi: int = 140,
) -> str:
    """Render the result panels to `path` (png, pdf, ...)."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure = plot_one_shot_result(
        reference_image,
        reference_mask,
        target_image,
        result,
        ground_truth=ground_truth,
        show=False,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return os.path.abspath(path)


def plot_similarity_map(similarity_map: np.ndarray, title: str = "similarity map", show: bool = True):
    """Standalone heatmap of the similarity map."""
    figure, axes = _figure(1, 1, figsize=(4.4, 3.8))
    handle = axes[0][0].imshow(np.asarray(similarity_map, dtype=np.float32), cmap="magma", vmin=0.0, vmax=1.0)
    axes[0][0].set_axis_off()
    axes[0][0].set_title(title, fontsize=10)
    figure.colorbar(handle, ax=axes[0][0], fraction=0.046, pad=0.04)
    figure.tight_layout()
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return figure
