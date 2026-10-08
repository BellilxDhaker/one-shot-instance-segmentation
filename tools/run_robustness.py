"""Run the pipeline over the Phase 3 robustness dataset and save results.

Phase 2 asked whether one frozen pipeline survives real re-photographs. Phase 3
asks how it degrades *gradually*: every controlled variant of a sample goes
through the unchanged pipeline, and each row is compared with both its own
ground truth and the sample's clean baseline result, so a number like
`retention = 0.42` means "42% of the IoU this sample scored without the
transformation".

    python tools/run_robustness.py --selection 12            # the full ladder
    python tools/run_robustness.py --report-only             # reassemble tables
    python tools/run_robustness.py --shard 1/2               # split the work

Which samples are used is decided once, from the baseline results: samples are
ranked by baseline IoU and picked at evenly spaced ranks, so retention is
measured across the whole performance range instead of only on the easy end.
The choice is recorded in `data/processed/selection.json` and reused on every
later run - deleting that file is how you ask for a re-selection.

Outputs under `--output` (default `results/robustness`):

    samples/<record_id>.json     one record per variant, the unit of resume
    <family>.csv                 the table of record, one file per family
    summary.json                 aggregates, severity table, found rates
    provenance.json              seed, model ids, versions, dataset, selection
    progress.json                live counter (ignored by git)

Resume works exactly like Phase 2: a completed record is never recomputed, and
deleting a record's JSON is how you ask for it to be redone. Rows are sorted so
all variants of one sample run back to back, which is what keeps the encoder's
image cache effective.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmentation.common import FAMILIES  # noqa: E402
from evaluation.cases import cases_from_manifest  # noqa: E402
from evaluation.cases import load_mask as load_evaluation_mask  # noqa: E402
from evaluation.dataset_integrity import check_dataset  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    aggregate_rows,
    evaluate_mask,
    iou_histogram,
    summarize_values,
)
from evaluation.runtime import (  # noqa: E402
    build_provenance,
    reset_gpu_memory,
    set_seed,
    summarize_times,
    write_json,
)

CSV_COLUMNS = (
    # required by the protocol, in order
    "sample_id", "condition", "severity", "method", "iou", "dice", "precision",
    "recall", "runtime",
    # provenance and context
    "family", "variant", "transformation", "original_sample_id", "random_seed",
    "category", "object_id", "clean_iou", "retention", "clean_dice",
    "retention_dice", "found", "confidence", "iou_visible", "dice_visible",
    "achieved_factor", "achieved_fraction", "target_fraction", "occluder_type",
    "visible_object_fraction", "yaw_deg",
    "feature_time", "segmentation_time", "peak_gpu_mb", "device",
)

FAMILY_ORDER = {family: index for index, family in enumerate(FAMILIES)}
SELECTION_FILE = "selection.json"
SELECTION_RATIONALE = (
    "samples ranked by baseline IoU (results/baseline/samples), then picked at "
    "evenly spaced ranks so retention is measured across the whole performance "
    "range instead of only on the easy end"
)
BASELINE_RECORD_FIELDS = ("iou", "dice")


@dataclass
class RobustnessOptions:
    """Everything a robustness run needs to know, in one place, for the record."""

    root: str = "data"
    output: str = "results/robustness"
    baseline: str = "results/baseline"
    families: Tuple[str, ...] = FAMILIES
    selection: int = 12
    samples: Optional[Tuple[str, ...]] = None
    reselect: bool = False
    method: str = "dinov2-base+sam-base"
    device: Optional[str] = None
    encoder_model: str = "facebook/dinov2-base"
    segmenter_model: str = "facebook/sam-vit-base"
    seed: int = 0
    build_seed: int = 7
    limit: Optional[int] = None
    shard: Optional[Tuple[int, int]] = None
    resume: bool = True
    force: bool = False
    report_only: bool = False
    strict_integrity: bool = False


def _stage_seconds(result: Any) -> Dict[str, float]:
    runtime = getattr(result, "runtime", {}) or {}
    stages: Dict[str, float] = {}
    for name, value in runtime.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            stages[name] = float(value)
    return stages


def _csv_row(record: Dict[str, Any]) -> Dict[str, Any]:
    row = {column: record.get(column) for column in CSV_COLUMNS}
    stages = record.get("stage_seconds") or {}
    row["runtime"] = stages.get("total")
    row["feature_time"] = stages.get("features")
    row["segmentation_time"] = stages.get("segmentation")
    return row


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    """Write one family's table with a fixed column order."""
    import csv

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            formatted = {}
            for column in CSV_COLUMNS:
                value = row.get(column)
                if isinstance(value, float):
                    formatted[column] = f"{value:.6g}"
                elif isinstance(value, (bool, np.bool_)):
                    formatted[column] = int(value)
                elif value is None:
                    formatted[column] = ""
                else:
                    formatted[column] = value
            writer.writerow(formatted)
    return path


def load_baseline(baseline: str) -> Dict[str, Dict[str, Any]]:
    """Clean per-sample metrics from the Phase 2 run, keyed by sample id."""
    directory = os.path.join(baseline, "samples")
    if not os.path.isdir(directory):
        raise SystemExit(f"baseline records not found in {directory}\n"
                         f"run the Phase 2 benchmark first: python tools/run_benchmark.py")
    clean: Dict[str, Dict[str, Any]] = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(directory, name), "r", encoding="utf-8") as handle:
            record = json.load(handle)
        clean[str(record.get("sample_id") or name[:-5])] = record
    return clean


def select_samples(root: str, baseline: str, size: int,
                   reselect: bool = False) -> List[str]:
    """Pick the samples the ladder runs on; persist the choice and its reason."""
    selection_path = os.path.join(root, "processed", SELECTION_FILE)
    if os.path.isfile(selection_path) and not reselect:
        with open(selection_path, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
        return [str(entry["sample_id"]) for entry in stored["samples"]]

    clean = load_baseline(baseline)
    scored = [(sample_id, float(record["iou"]))
              for sample_id, record in clean.items()
              if record.get("iou") is not None]
    scored.sort(key=lambda item: (item[1], item[0]))
    if len(scored) < size:
        raise SystemExit(f"baseline has {len(scored)} scored samples, "
                         f"cannot select {size}")
    ranks = np.linspace(0, len(scored) - 1, size)
    chosen = [scored[int(round(rank))] for rank in ranks]
    payload = {
        "rationale": SELECTION_RATIONALE,
        "baseline": baseline,
        "baseline_samples": len(scored),
        "selection_size": size,
        "samples": [
            {"sample_id": sample_id, "baseline_iou": round(iou, 6),
             "rank": int(round(rank))}
            for (sample_id, iou), rank in zip(chosen, ranks)
        ],
    }
    os.makedirs(os.path.dirname(selection_path), exist_ok=True)
    write_json(selection_path, payload)
    print(f"selected {size} samples across baseline IoU "
          f"{chosen[0][1]:.3f}..{chosen[-1][1]:.3f} -> {selection_path}")
    return [sample_id for sample_id, _ in chosen]


def _occluder_path(root: str, case: Any) -> Optional[str]:
    original = case.meta.get("original_sample_id")
    variant = case.meta.get("variant")
    if not original or not variant:
        return None
    path = os.path.join(root, "processed", "occlusion",
                        f"{original}__{variant}", "occluder_mask.png")
    return path if os.path.isfile(path) else None


def _clean_metrics(clean: Dict[str, Dict[str, Any]],
                   case: Any) -> Tuple[Optional[float], Optional[float]]:
    record = clean.get(str(case.meta.get("original_sample_id")))
    if record is None:
        return None, None
    iou = record.get("iou")
    dice = record.get("dice")
    return (float(iou) if iou is not None else None,
            float(dice) if dice is not None else None)


def _retention(value: Optional[float], clean: Optional[float]) -> Optional[float]:
    """Transformed / clean; undefined when the clean score is zero or missing."""
    if value is None or clean is None or clean <= 0.0:
        return None
    return float(value / clean)


def _score(result: Any, case: Any, occluder: Optional[np.ndarray]) -> Dict[str, Any]:
    metrics = evaluate_mask(result.predicted_mask, case.target_mask)
    values: Dict[str, Any] = {
        "iou": float(metrics.iou),
        "dice": float(metrics.dice),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
    }
    if occluder is not None:
        visible = ~occluder
        visible_metrics = evaluate_mask(result.predicted_mask & visible,
                                        case.target_mask & visible)
        values["iou_visible"] = float(visible_metrics.iou)
        values["dice_visible"] = float(visible_metrics.dice)
    else:
        values["iou_visible"] = None
        values["dice_visible"] = None
    return values


def severity_table(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean IoU and retention per (family, condition, severity) - the shape the
    severity curves are drawn from."""
    buckets: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for record in records:
        if record.get("iou") is None:
            continue
        key = (str(record.get("family")), str(record.get("condition")),
               int(record.get("severity") or 0))
        buckets.setdefault(key, []).append(record)
    rows: List[Dict[str, Any]] = []
    for (family, condition, severity), group in sorted(buckets.items()):
        ious = [float(record["iou"]) for record in group]
        retentions = [float(record["retention"]) for record in group
                      if record.get("retention") is not None]
        rows.append({
            "family": family,
            "condition": condition,
            "severity": severity,
            "count": len(group),
            "mean_iou": round(float(np.mean(ious)), 6),
            "median_iou": round(float(np.median(ious)), 6),
            "mean_retention": round(float(np.mean(retentions)), 6) if retentions else None,
            "found_rate": round(float(np.mean([bool(r.get("found")) for r in group])), 6),
        })
    return rows


def load_records(output: str) -> List[Dict[str, Any]]:
    """Every record on disk, sorted for stable tables."""
    directory = os.path.join(output, "samples")
    records: List[Dict[str, Any]] = []
    if not os.path.isdir(directory):
        return records
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json") or name == "progress.json":
            continue
        with open(os.path.join(directory, name), "r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    records.sort(key=lambda record: (
        FAMILY_ORDER.get(str(record.get("family")), len(FAMILIES)),
        str(record.get("original_sample_id") or ""),
        str(record.get("sample_id") or ""),
    ))
    return records


def assemble(options: RobustnessOptions, elapsed: Optional[float] = None,
             reused: int = 0) -> Dict[str, Any]:
    """Rebuild the CSVs, summary and provenance from the records on disk."""
    records = load_records(options.output)
    scored = [record for record in records if record.get("iou") is not None]

    csv_paths: Dict[str, str] = {}
    for family in options.families:
        family_rows = [record for record in records if record.get("family") == family]
        if family_rows:
            csv_paths[family] = write_csv(
                os.path.join(options.output, f"{family}.csv"),
                [_csv_row(record) for record in family_rows])

    summary = aggregate_rows(scored, by=("family", "condition"))
    summary["severity_table"] = severity_table(records)
    summary["iou_histogram"] = iou_histogram([record["iou"] for record in scored])
    summary["retention"] = summarize_values(
        [float(record["retention"]) for record in scored
         if record.get("retention") is not None])
    summary["runtime"] = {
        "wall": summarize_times([record.get("wall_seconds", 0.0) for record in records]),
        "per_stage": {
            stage: summarize_times([record["stage_seconds"][stage] for record in records
                                    if stage in (record.get("stage_seconds") or {})])
            for stage in ("features", "segmentation")
        },
    }
    summary["found_rate"] = (
        float(np.mean([bool(record.get("found")) for record in records]))
        if records else 0.0)
    summary["method"] = options.method
    summary["families"] = list(options.families)
    summary["records"] = len(records)
    summary["scored"] = len(scored)
    summary["reused_from_cache"] = reused
    summary["samples_with_records"] = sorted({
        str(record.get("original_sample_id")) for record in records
        if record.get("original_sample_id")
    })
    if elapsed is not None:
        summary["elapsed_seconds"] = round(elapsed, 1)

    write_json(os.path.join(options.output, "summary.json"), summary)
    write_json(os.path.join(options.output, "provenance.json"), build_provenance(
        device=options.device, seed=options.seed,
        dataset={
            "root": options.root,
            "output": options.output,
            "baseline": options.baseline,
            "families": list(options.families),
            "records": len(records),
            "samples": len(summary["samples_with_records"]),
            "dataset": "coco2017-val",
        },
        extra={"method": options.method, "selection_size": options.selection,
               "build_seed": options.build_seed,
               "csvs": {family: path for family, path in csv_paths.items()}},
    ))
    return {"summary": summary, "csvs": csv_paths, "records": records}


def _write_progress(options: RobustnessOptions, done: int, total: int,
                    families: Dict[str, int]) -> None:
    write_json(os.path.join(options.output, "progress.json"), {
        "done": done,
        "total": total,
        "families": families,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def _build_cases(options: RobustnessOptions, selected: Sequence[str]) -> List[Any]:
    """Load every requested family's manifest and keep the selected samples,
    ordered so all variants of one sample run back to back."""
    selected_set = set(selected)
    cases: List[Any] = []
    for family in options.families:
        manifest = os.path.join(options.root, "processed", family, "manifest.json")
        if not os.path.isfile(manifest):
            raise SystemExit(f"manifest not found: {manifest}\n"
                             f"build the dataset first: python tools/build_robustness.py")
        for case in cases_from_manifest(manifest):
            if str(case.meta.get("original_sample_id")) in selected_set:
                cases.append(case)
    order = {sample_id: index for index, sample_id in enumerate(selected)}
    cases.sort(key=lambda case: (
        order.get(str(case.meta.get("original_sample_id")), len(order)),
        FAMILY_ORDER.get(str(case.meta.get("family")), len(FAMILIES)),
        int(case.meta.get("severity") or 0),
        str(case.meta.get("variant") or ""),
    ))
    return cases


def run_robustness(options: RobustnessOptions, pipeline: Any = None,
                   cases: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """Score the robustness dataset and write every artefact. Returns the summary.

    The integrity gate, the per-record resume unit and the CSV contract are the
    same ones Phase 2 proved; the additions here are retention against the clean
    baseline, visible-region scoring for occlusions, and sharding by *sample* so
    two workers never fight over the encoder cache.
    """
    from pipeline.one_shot_segmentation import get_default_pipeline

    set_seed(options.seed)
    if options.report_only:
        result = assemble(options)
        print(f"report: {result['summary']['records']} records, "
              f"{len(result['csvs'])} CSVs in {options.output}")
        return result["summary"]

    if options.samples:
        selected = list(options.samples)
    else:
        selected = select_samples(options.root, options.baseline, options.selection,
                                  reselect=options.reselect)
    if options.shard is not None:
        shard_index, shard_count = options.shard
        selected = selected[shard_index - 1::shard_count]

    report = check_dataset(root=options.root, strict=options.strict_integrity)
    if not report.ok():
        raise SystemExit("dataset integrity check failed, refusing to run:\n"
                         + report.summary(limit=8))
    print(f"integrity: {report.samples_checked} samples, "
          f"{len(report.categories)} categories, no critical findings")

    if cases is None:
        cases = _build_cases(options, selected)
    if options.limit is not None:
        cases = list(cases)[:options.limit]
    if not cases:
        raise SystemExit("no cases selected; run tools/build_robustness.py first")
    clean = load_baseline(options.baseline)
    missing_clean = sorted({
        str(case.meta.get("original_sample_id")) for case in cases
        if str(case.meta.get("original_sample_id")) not in clean})
    if missing_clean:
        raise SystemExit(f"no baseline record for samples {missing_clean}; "
                         f"run tools/run_benchmark.py first")

    print(f"{len(cases)} cases over {len(selected)} samples "
          f"({len(options.families)} families)")
    if pipeline is None:
        print(f"loading models (device={options.device or 'auto'})")
        pipeline = get_default_pipeline(
            device=options.device,
            encoder_kwargs={"model_id": options.encoder_model},
            segmenter_kwargs={"model_id": options.segmenter_model},
        )

    samples_dir = os.path.join(options.output, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    reused = 0
    started = time.time()
    families_done: Dict[str, int] = {family: 0 for family in options.families}

    for position, case in enumerate(cases, start=1):
        record_path = os.path.join(samples_dir, f"{case.name}.json")
        if options.resume and not options.force and os.path.isfile(record_path):
            try:
                with open(record_path, "r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                reused += 1
                families_done[str(cached.get("family"))] = (
                    families_done.get(str(cached.get("family")), 0) + 1)
                if reused % 25 == 0:
                    _write_progress(options, position, len(cases), families_done)
                continue
            except (OSError, ValueError):
                pass

        reset_gpu_memory(getattr(pipeline, "device", None))
        sample_started = time.perf_counter()
        result = pipeline.segment_object(case.reference_image, case.reference_mask,
                                         case.target_image)
        wall = time.perf_counter() - sample_started

        occluder = None
        if case.meta.get("family") == "occlusion":
            path = _occluder_path(options.root, case)
            if path is not None:
                occluder = load_evaluation_mask(path)
        metric_values = _score(result, case, occluder)

        clean_iou, clean_dice = _clean_metrics(clean, case)
        stages = _stage_seconds(result)
        record: Dict[str, Any] = {
            "sample_id": case.name,
            "original_sample_id": case.meta.get("original_sample_id"),
            "family": case.meta.get("family"),
            "condition": case.meta.get("condition"),
            "severity": case.meta.get("severity"),
            "variant": case.meta.get("variant"),
            "transformation": case.meta.get("transformation"),
            "random_seed": case.meta.get("random_seed"),
            "method": options.method,
            "category": case.meta.get("category"),
            "object_id": case.meta.get("object_id"),
            "clean_iou": clean_iou,
            "clean_dice": clean_dice,
            "retention": _retention(metric_values["iou"], clean_iou),
            "retention_dice": _retention(metric_values["dice"], clean_dice),
            "found": bool(result.found),
            "confidence": float(result.confidence),
            "achieved_factor": case.meta.get("achieved_factor"),
            "achieved_fraction": case.meta.get("achieved_fraction"),
            "target_fraction": case.meta.get("target_fraction"),
            "occluder_type": case.meta.get("occluder_type"),
            "visible_object_fraction": case.meta.get("visible_object_fraction"),
            "yaw_deg": case.meta.get("yaw_deg"),
            "wall_seconds": float(wall),
            "stage_seconds": stages,
            "peak_gpu_mb": _peak_memory(pipeline),
            "device": str(getattr(pipeline, "device", "unknown")),
            **metric_values,
        }
        write_json(record_path, record)
        families_done[str(record["family"])] = (
            families_done.get(str(record["family"]), 0) + 1)
        _write_progress(options, position, len(cases), families_done)

        iou = record.get("iou")
        iou_text = "  n/a" if iou is None else f"IoU {iou:.3f}"
        retention = record.get("retention")
        retention_text = "" if retention is None else f" ret {retention:.2f}"
        print(f"[{position:4d}/{len(cases)}] {case.name:44s} {iou_text:12s}"
              f"{retention_text:9s} {wall:6.2f}s")

    elapsed = time.time() - started
    result = assemble(options, elapsed=elapsed, reused=reused)
    summary = result["summary"]
    print(f"\nwrote {len(result['csvs'])} CSVs to {options.output}")
    print(f"scored {summary['scored']}/{summary['records']} records, "
          f"reused {reused} cached, {summary.get('elapsed_seconds')}s")
    if summary["retention"]:
        stats = summary["retention"]
        print(f"retention mean {stats['mean']:.3f} median {stats['median']:.3f} "
              f"over {stats['count']} rows")
    return summary


def _peak_memory(pipeline: Any) -> Optional[float]:
    from evaluation.runtime import gpu_memory_mb

    return gpu_memory_mb(getattr(pipeline, "device", None))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data", help="dataset root (default: data)")
    parser.add_argument("--output", default="results/robustness",
                        help="results directory (default results/robustness)")
    parser.add_argument("--baseline", default="results/baseline",
                        help="Phase 2 results used for retention (default results/baseline)")
    parser.add_argument("--families", nargs="+", default=list(FAMILIES),
                        choices=list(FAMILIES), help="families to score")
    parser.add_argument("--selection", type=int, default=12,
                        help="how many baseline-stratified samples to run")
    parser.add_argument("--samples", nargs="+", default=None,
                        help="explicit sample ids (overrides --selection)")
    parser.add_argument("--reselect", action="store_true",
                        help="rewrite data/processed/selection.json")
    parser.add_argument("--method", default="dinov2-base+sam-base")
    parser.add_argument("--device", default=None)
    parser.add_argument("--encoder-model", default="facebook/dinov2-base")
    parser.add_argument("--segmenter-model", default="facebook/sam-vit-base")
    parser.add_argument("--seed", type=int, default=0, help="runtime seed")
    parser.add_argument("--build-seed", type=int, default=7,
                        help="seed used when the dataset was built (provenance)")
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N cases (smoke runs)")
    parser.add_argument("--shard", default=None, metavar="K/N",
                        help="run shard K of N over the selected samples")
    parser.add_argument("--force", action="store_true", help="re-score existing records")
    parser.add_argument("--report-only", action="store_true",
                        help="reassemble CSVs/summary from records without a pipeline")
    parser.add_argument("--strict-integrity", action="store_true")
    arguments = parser.parse_args(argv)

    shard: Optional[Tuple[int, int]] = None
    if arguments.shard:
        try:
            shard_part, count_part = arguments.shard.split("/")
            shard = (int(shard_part), int(count_part))
        except ValueError:
            parser.error("--shard must look like 1/2")
        if not (1 <= shard[0] <= shard[1]):
            parser.error("--shard K must satisfy 1 <= K <= N")

    options = RobustnessOptions(
        root=arguments.root,
        output=arguments.output,
        baseline=arguments.baseline,
        families=tuple(arguments.families),
        selection=arguments.selection,
        samples=tuple(arguments.samples) if arguments.samples else None,
        reselect=arguments.reselect,
        method=arguments.method,
        device=arguments.device,
        encoder_model=arguments.encoder_model,
        segmenter_model=arguments.segmenter_model,
        seed=arguments.seed,
        build_seed=arguments.build_seed,
        limit=arguments.limit,
        shard=shard,
        force=arguments.force,
        report_only=arguments.report_only,
        strict_integrity=arguments.strict_integrity,
    )
    run_robustness(options)
    return 0


if __name__ == "__main__":
    sys.exit(main())
