"""Benchmark figures: one row per pair, plus the distribution across a run.

`visualization/results.py` shows what a single run did. A benchmark needs two
more views. `plot_benchmark_pair` puts the whole story of one pair side by side
in the order a reader needs it: what was shown as the reference, what was being
looked for, what the encoder thought matched, which region was chosen, what came
out, and how that compares with the truth. `plot_benchmark_summary` covers a whole
run: the IoU distribution, mean IoU per category, and runtime per stage, which is
where a regression shows up first.

Two deliberate choices are worth stating. Every panel is captioned with the value
that produced it, so a figure stays readable without the CSV beside it. And
failures are drawn, not hidden: a run that found nothing still gets a full row
with an explicit "no candidate" caption, because those rows are the interesting
ones.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from evaluation.metrics import MaskMetrics
from visualization.results import _overlay_mask, _show_image, _to_rgb


def _figure(nrows: int, ncols: int, width: float = 3.1, height: float = 2.9):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows),
                               squeeze=False)
    return figure, axes


def _title(axis, text: str, colour: str = "black", size: float = 9.0) -> None:
    axis.set_title(text, fontsize=size, color=colour, pad=3)


def _mask_panel(axis, mask: Optional[np.ndarray], colour: str = "#ff2d55") -> None:
    """Show a mask on its own, as a flat silhouette."""
    if mask is None or not np.asarray(mask).any():
        axis.text(0.5, 0.5, "no prediction", ha="center", va="center",
                  transform=axis.transAxes, fontsize=9, color="#b0b0b0")
        axis.set_axis_off()
        return
    binary = np.asarray(mask, dtype=bool)
    rgba = np.zeros((*binary.shape, 4), dtype=np.float32)
    rgb = np.array(_to_rgb(colour), dtype=np.float32) / 255.0
    rgba[..., :3] = rgb
    rgba[..., 3] = binary.astype(np.float32)
    axis.imshow(rgba)
    axis.set_axis_off()


def _similarity_panel(axis, similarity: Optional[np.ndarray]) -> None:
    if similarity is None or not np.asarray(similarity).size:
        axis.text(0.5, 0.5, "no map", ha="center", va="center", transform=axis.transAxes,
                  fontsize=9, color="#b0b0b0")
        axis.set_axis_off()
        return
    axis.imshow(np.asarray(similarity, dtype=float), cmap="magma", vmin=0.0, vmax=1.0)
    axis.set_axis_off()


def plot_benchmark_pair(
    reference_image: Any,
    reference_mask: np.ndarray,
    target_image: Any,
    similarity_map: Optional[np.ndarray] = None,
    candidate_region: Optional[np.ndarray] = None,
    predicted_mask: Optional[np.ndarray] = None,
    target_mask: Optional[np.ndarray] = None,
    metrics: Optional[MaskMetrics] = None,
    caption: str = "",
    confidence: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """Draw the seven panels that explain one pair, left to right.

    Args:
        reference_image: the image the object was shown from.
        reference_mask: the annotated reference mask.
        target_image: the image the object was to be found in.
        similarity_map: (H, W) correspondence scores in [0, 1].
        candidate_region: the region localisation selected.
        predicted_mask: the final mask, or `None` when nothing was found.
        target_mask: the ground truth for the target, when it exists.
        metrics: the scored metrics, shown on the overlay panel.
        caption: dataset, category and condition labels for the title.
        confidence: the pipeline's confidence, shown on the candidate panel.
    """
    figure, axes = _figure(1, 7, width=3.0, height=3.1)
    if figsize is not None:
        figure.set_size_inches(*figsize)

    _show_image(axes[0][0], reference_image)
    _title(axes[0][0], f"reference image\n{reference_mask.mean() * 100:.1f}% of frame")

    _mask_panel(axes[0][1], reference_mask, colour="#2f7dff")
    _title(axes[0][1], "reference mask\nthe query object")

    _show_image(axes[0][2], target_image)
    _title(axes[0][2], "target image\nunseen photograph")

    _similarity_panel(axes[0][3], similarity_map)
    if similarity_map is not None and np.asarray(similarity_map).size:
        _title(axes[0][3], f"similarity map\npeak {float(np.max(similarity_map)):.2f}")
    else:
        _title(axes[0][3], "similarity map")

    _mask_panel(axes[0][4], candidate_region, colour="#f5a623")
    _title(axes[0][4],
           f"candidate region\nconf {confidence:.3f}" if confidence is not None
           else "candidate region\nnone found")

    _mask_panel(axes[0][5], predicted_mask, colour="#00c853")
    found = predicted_mask is not None and bool(np.asarray(predicted_mask).any())
    _title(axes[0][5], f"predicted mask\n{found * np.asarray(predicted_mask).mean() * 100:.1f}% of frame"
           if found else "predicted mask\nNOT FOUND",
           colour="black" if found else "#d32f2f", size=8.5)

    _overlay_mask(axes[0][6], target_image, target_mask, colour="#00c853", alpha=0.55)
    if predicted_mask is not None and np.asarray(predicted_mask).any():
        _overlay_mask(axes[0][6], target_image, predicted_mask, colour="#ff2d55", alpha=0.45)
    if metrics is not None:
        _title(axes[0][6],
               f"truth green / pred red\nIoU {metrics.iou:.3f}  Dice {metrics.dice:.3f}\n"
               f"P {metrics.precision:.3f}  R {metrics.recall:.3f}",
               colour="black" if metrics.iou >= 0.5 else "#d32f2f", size=8.5)
    else:
        _title(axes[0][6], "no ground truth")

    if caption:
        figure.suptitle(caption, fontsize=10, y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.94 if caption else 1.0))
    return figure


def save_benchmark_pair(path: str, **kwargs: Any) -> str:
    """Render one pair figure and write it to `path`."""
    figure = plot_benchmark_pair(**kwargs)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(path, dpi=140, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path


def plot_iou_distribution(values: Sequence[float], bins: int = 10, title: str = "IoU distribution",
                          path: Optional[str] = None):
    """Histogram of IoU with success-rate markers, as one axis."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    array = np.asarray([float(value) for value in values if value is not None], dtype=float)
    if array.size == 0:
        axis.text(0.5, 0.5, "no scored samples", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    else:
        axis.hist(np.clip(array, 0.0, 1.0), bins=bins, range=(0.0, 1.0),
                  color="#2f7dff", edgecolor="white", linewidth=0.8)
        for threshold, colour in ((0.5, "#2e7d32"), (0.7, "#f9a825"), (0.9, "#c62828")):
            share = float((array >= threshold).mean())
            axis.axvline(threshold, color=colour, linestyle="--", linewidth=1.0, alpha=0.8)
            axis.text(threshold, axis.get_ylim()[1] * 0.97, f"  IoU>={threshold:g}: {share * 100:.0f}%",
                      color=colour, fontsize=8, va="top")
        axis.set_xlabel("IoU between predicted and ground-truth mask")
        axis.set_ylabel("pairs")
        axis.set_xlim(0.0, 1.0)
        axis.set_title(f"{title}  (n={array.size}, mean {array.mean():.3f}, "
                      f"median {np.median(array):.3f})", fontsize=10)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    figure.tight_layout()
    if path is not None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        figure.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        return path
    return figure


def plot_group_means(rows: Sequence[Dict[str, Any]], column: str = "category",
                     metric: str = "iou", max_groups: int = 30, path: Optional[str] = None):
    """Mean metric per group, sorted worst first, with counts in the tick labels."""
    import matplotlib.pyplot as plt

    buckets: Dict[str, list] = {}
    for row in rows:
        if metric in row and row[metric] is not None:
            buckets.setdefault(str(row.get(column, "unknown")), []).append(float(row[metric]))
    if not buckets:
        return plot_iou_distribution([], path=path)
    ordered = sorted(buckets.items(), key=lambda item: float(np.mean(item[1])))[:max_groups]
    labels = [f"{name}\n(n={len(values)})" for name, values in ordered]
    means = [float(np.mean(values)) for _, values in ordered]
    spread = [float(np.std(values)) for _, values in ordered]

    figure, axis = plt.subplots(figsize=(max(7.0, 0.42 * len(labels)), 4.0))
    positions = np.arange(len(labels))
    axis.bar(positions, means, yerr=spread, color="#00acc1", edgecolor="white", linewidth=0.7,
             capsize=2)
    axis.axhline(float(np.mean([row[metric] for row in rows if metric in row
                                and row[metric] is not None])), color="#455a64",
                 linestyle="--", linewidth=1.0, label="overall mean")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    axis.set_ylabel(f"mean {metric}")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(f"mean {metric} by {column} (worst {len(ordered)} of {len(buckets)})", fontsize=10)
    axis.legend(fontsize=8, loc="upper right")
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    figure.tight_layout()
    if path is not None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        figure.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        return path
    return figure


def plot_stage_runtime(rows: Sequence[Dict[str, Any]], path: Optional[str] = None):
    """Where the time went: median seconds per pipeline stage, with a per-sample spread."""
    import matplotlib.pyplot as plt

    stages: Dict[str, list] = {}
    for row in rows:
        for name, value in (row.get("stage_seconds") or {}).items():
            if isinstance(value, (int, float)):
                stages.setdefault(name, []).append(float(value))
    if not stages:
        return None
    ordered = sorted(stages.items(), key=lambda item: float(np.median(item[1])), reverse=True)
    labels = [name.replace("_", " ") for name, _ in ordered]
    medians = [float(np.median(values)) for _, values in ordered]

    figure, axis = plt.subplots(figsize=(7.0, 0.6 * len(labels) + 1.6))
    positions = np.arange(len(labels))
    axis.barh(positions, medians, color="#7e57c2", edgecolor="white", linewidth=0.7)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("median seconds per pair")
    axis.set_title("pipeline cost by stage", fontsize=10)
    for index, (_, values) in enumerate(ordered):
        axis.text(medians[index], index, f"  p95 {np.percentile(values, 95):.2f}s",
                  va="center", fontsize=7, color="#424242")
    axis.grid(axis="x", alpha=0.25, linewidth=0.6)
    figure.tight_layout()
    if path is not None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        figure.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        return path
    return figure


def plot_benchmark_summary(rows: Sequence[Dict[str, Any]], directory: str,
                           method: str = "baseline") -> Dict[str, str]:
    """Write the three run-level figures and return their paths."""
    values = [row["iou"] for row in rows if row.get("iou") is not None]
    written = {
        "iou_distribution": plot_iou_distribution(
            values, title=f"IoU distribution, {method}", path=os.path.join(directory, "iou_distribution.png")),
        "per_category": plot_group_means(
            rows, column="category", metric="iou",
            path=os.path.join(directory, "iou_by_category.png")) or "",
        "runtime": plot_stage_runtime(
            rows, path=os.path.join(directory, "runtime_by_stage.png")) or "",
    }
    return written
