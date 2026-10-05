"""Step 9 - evaluation metrics for a small, verified set of examples.

The four metrics requested for Phase 1 are IoU, Dice, precision and recall.
They are computed from boolean masks, tolerate empty inputs by returning
defined values (1.0 when both sides are empty, 0.0 when only the prediction is
empty), and are paired with a `summarise` helper that reports the mean plus the
per-example rows, so a handful of results stays readable.

Phase 1 evaluates a handful of manually checked examples, so this module
deliberately has no dataset iteration, no sweeps and no aggregation tricks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from matching.reference_representation import MaskInput, to_binary_mask

EPSILON = 1e-7


@dataclass
class MaskMetrics:
    """Overlap metrics for one prediction against one ground-truth mask."""

    iou: float
    dice: float
    precision: float
    recall: float
    true_positive: int
    false_positive: int
    false_negative: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "iou": self.iou,
            "dice": self.dice,
            "precision": self.precision,
            "recall": self.recall,
            "true_positive": float(self.true_positive),
            "false_positive": float(self.false_positive),
            "false_negative": float(self.false_negative),
        }

    def __str__(self) -> str:
        return f"IoU {self.iou:.3f} | Dice {self.dice:.3f} | P {self.precision:.3f} | R {self.recall:.3f}"


def _as_bool(mask: Optional[MaskInput]) -> np.ndarray:
    if mask is None:
        return np.zeros((0, 0), dtype=bool)
    tensor = to_binary_mask(mask)
    return (tensor.squeeze() > 0.5).cpu().numpy().astype(bool)


def evaluate_mask(prediction: Optional[MaskInput], ground_truth: MaskInput) -> MaskMetrics:
    """IoU, Dice, precision and recall for one predicted mask."""
    predicted = _as_bool(prediction)
    truth = _as_bool(ground_truth)

    if predicted.shape != truth.shape:
        if predicted.size == 0:
            predicted = np.zeros(truth.shape, dtype=bool)
        else:
            import torch.nn.functional as F

            resized = F.interpolate(
                to_binary_mask(prediction), size=truth.shape[-2:], mode="nearest"
            )
            predicted = (resized.squeeze() > 0.5).cpu().numpy().astype(bool)

    intersection = int(np.logical_and(predicted, truth).sum())
    predicted_area = int(predicted.sum())
    truth_area = int(truth.sum())
    union = predicted_area + truth_area - intersection

    if union == 0:
        iou = 1.0
        dice = 1.0
    else:
        iou = intersection / union
        dice = (2.0 * intersection) / (predicted_area + truth_area) if (predicted_area + truth_area) else 0.0

    precision = intersection / predicted_area if predicted_area else (1.0 if truth_area == 0 else 0.0)
    recall = intersection / truth_area if truth_area else (1.0 if predicted_area == 0 else 0.0)

    return MaskMetrics(
        iou=float(iou),
        dice=float(dice),
        precision=float(precision),
        recall=float(recall),
        true_positive=intersection,
        false_positive=predicted_area - intersection,
        false_negative=truth_area - intersection,
    )


@dataclass
class EvaluationReport:
    """Per-example rows plus the mean of each metric.

    Attributes:
        rows: one dict per example, containing the metrics, the confidence and
            the runtime when available.
        metric_names: the metrics that the mean is computed over.
    """

    rows: List[Dict[str, Any]]
    metric_names: Tuple[str, ...] = ("iou", "dice", "precision", "recall")

    def mean(self, metric: str = "iou") -> float:
        values = [float(row[metric]) for row in self.rows if metric in row]
        return float(np.mean(values)) if values else float("nan")

    def summary(self) -> Dict[str, float]:
        return {name: self.mean(name) for name in self.metric_names}

    def to_table(self) -> str:
        if not self.rows:
            return "no evaluated examples"
        header = f"{'example':<24}{'IoU':>8}{'Dice':>8}{'P':>8}{'R':>8}{'conf':>8}{'sec':>8}"
        lines = [header, "-" * len(header)]
        for row in self.rows:
            lines.append(
                f"{str(row.get('example', '')):<24}"
                f"{row.get('iou', float('nan')):>8.3f}"
                f"{row.get('dice', float('nan')):>8.3f}"
                f"{row.get('precision', float('nan')):>8.3f}"
                f"{row.get('recall', float('nan')):>8.3f}"
                f"{row.get('confidence', float('nan')):>8.3f}"
                f"{row.get('seconds', float('nan')):>8.2f}"
            )
        lines.append("-" * len(header))
        summary = self.summary()
        lines.append(
            f"{'mean':<24}" + "".join(f"{summary[name]:>8.3f}" for name in self.metric_names)
        )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"rows": self.rows, "summary": self.summary(), "num_examples": len(self.rows)}


def evaluate_results(
    examples: Sequence[Tuple[str, Optional[MaskInput], MaskInput, Any]],
) -> EvaluationReport:
    """Score `(name, prediction, ground_truth, result)` tuples into a report.

    The fourth element is the `OneShotResult` the prediction came from, used only
    to attach the confidence and the runtime; `None` is accepted.
    """
    rows: List[Dict[str, Any]] = []
    for name, prediction, ground_truth, result in examples:
        metrics = evaluate_mask(prediction, ground_truth)
        row: Dict[str, Any] = {"example": name, **metrics.to_dict()}
        if result is not None:
            row["confidence"] = float(getattr(result, "confidence", float("nan")))
            row["seconds"] = float(getattr(result, "runtime", {}).get("total", float("nan")))
        rows.append(row)
    return EvaluationReport(rows=rows)
