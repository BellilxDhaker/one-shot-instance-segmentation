"""Run the pipeline over a benchmark split and save results, figures and provenance.

This is the Phase 2 counterpart to `evaluation.cases.run_cases`, which printed a
line per case. A benchmark has to survive being interrupted, has to keep its
numbers next to the machine that produced them, and has to write figures that
explain the numbers. So each sample is scored into its own JSON file first, and
the tables and figures are assembled from those files afterwards.

That ordering is what makes resume cheap and safe: a completed sample is never
recomputed, and if the process dies halfway the next run picks up from the last
finished sample. Deleting a sample's JSON is how you ask for it to be redone.

    python tools/run_benchmark.py --split test --device cuda --figures 12
    python tools/run_benchmark.py --split test --resume          # fill in what is missing

Outputs under `--output` (default `results/baseline`):

    samples/<sample_id>.json    one record per pair, the unit of resume
    results.csv                 one row per pair, the table of record
    summary.json                aggregate statistics per metric and group
    iou_distribution.png        run-level figures
    iou_by_category.png
    runtime_by_stage.png
    provenance.json             seed, model ids, versions, GPU, dataset, split
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.cases import cases_from_manifest  # noqa: E402
from evaluation.dataset_integrity import check_dataset  # noqa: E402
from evaluation.metrics import aggregate_rows, evaluate_mask, iou_histogram  # noqa: E402
from evaluation.runtime import (  # noqa: E402
    build_provenance,
    describe_environment,
    reset_gpu_memory,
    set_seed,
    summarize_times,
    write_json,
)

CSV_COLUMNS = (
    "sample_id", "object_id", "method", "iou", "dice", "precision", "recall",
    "runtime", "feature_time", "segmentation_time", "found", "confidence",
    "dataset", "category", "identity_level",
    "viewpoint_condition", "illumination_condition", "scale_condition", "occlusion_condition",
    "reference_area_fraction", "target_area_fraction", "peak_gpu_mb", "device",
)


@dataclass
class BenchmarkOptions:
    """Everything a run needs to know, in one place, for the record."""

    root: str = "data"
    split: str = "test"
    output: str = "results/baseline"
    method: str = "dinov2-base+sam-base"
    device: Optional[str] = None
    encoder_model: str = "facebook/dinov2-base"
    segmenter_model: str = "facebook/sam-vit-base"
    seed: int = 0
    limit: Optional[int] = None
    figures: int = 8
    resume: bool = True
    force: bool = False
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
    """Write the results table with a fixed column order."""
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


def run_split(options: BenchmarkOptions, pipeline: Any = None,
              cases: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """Score one split and write every artefact. Returns the summary dict.

    The pipeline is built once and reused, which is what makes the per-sample
    timings comparable: they all pay the same one-off model load.
    """
    from pipeline.one_shot_segmentation import get_default_pipeline

    set_seed(options.seed)
    manifest = os.path.join(options.root, "splits", f"{options.split}.json")
    if options.split == "all":
        manifests = [os.path.join(options.root, "splits", f"{name}.json")
                     for name in ("train", "validation", "test")]
        missing = [path for path in manifests if not os.path.isfile(path)]
        if missing:
            raise SystemExit(f"split manifest(s) not found: {missing}\n"
                             f"build the benchmark first: python tools/build_benchmark.py")
    elif not os.path.isfile(manifest):
        raise SystemExit(f"split manifest not found: {manifest}\n"
                         f"build the benchmark first: python tools/build_benchmark.py")
    report = check_dataset(root=options.root, strict=options.strict_integrity)
    if not report.ok():
        raise SystemExit("dataset integrity check failed, refusing to run:\n"
                         + report.summary(limit=8))
    print(f"integrity: {report.samples_checked} samples, "
          f"{len(report.categories)} categories, no critical findings")

    if cases is None:
        if options.split == "all":
            cases = []
            seen: set = set()
            for path in manifests:
                for case in cases_from_manifest(path):
                    if case.name not in seen:
                        seen.add(case.name)
                        cases.append(case)
            manifest = ", ".join(os.path.basename(path) for path in manifests)
        else:
            cases = cases_from_manifest(manifest)
    if options.limit is not None:
        cases = list(cases)[:options.limit]
    print(f"split {options.split}: {len(cases)} cases from {manifest}")

    if pipeline is None:
        print(f"loading models (device={options.device or 'auto'})")
        pipeline = get_default_pipeline(
            device=options.device,
            encoder_kwargs={"model_id": options.encoder_model},
            segmenter_kwargs={"model_id": options.segmenter_model},
        )

    samples_dir = os.path.join(options.output, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    records: List[Dict[str, Any]] = []
    reused = 0
    started = time.time()

    for position, case in enumerate(cases, start=1):
        record_path = os.path.join(samples_dir, f"{case.name}.json")
        if options.resume and not options.force and os.path.isfile(record_path):
            try:
                import json

                with open(record_path, "r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                records.append(cached)
                reused += 1
                continue
            except (OSError, ValueError):
                pass

        reset_gpu_memory(getattr(pipeline, "device", None))
        sample_started = time.perf_counter()
        result = pipeline.segment_object(case.reference_image, case.reference_mask,
                                         case.target_image)
        wall = time.perf_counter() - sample_started

        if case.scorable:
            metrics = evaluate_mask(result.predicted_mask, case.target_mask)
            metric_values: Dict[str, Any] = metrics.to_dict()
            metric_values["iou"] = float(metrics.iou)
        else:
            metric_values = {name: None for name in ("iou", "dice", "precision", "recall")}

        stages = _stage_seconds(result)
        record: Dict[str, Any] = {
            "sample_id": case.name,
            "object_id": case.meta.get("object_id"),
            "method": options.method,
            "dataset": case.meta.get("dataset"),
            "category": case.meta.get("category"),
            "identity_level": case.meta.get("identity_level"),
            "viewpoint_condition": case.meta.get("viewpoint_condition", "unknown"),
            "illumination_condition": case.meta.get("illumination_condition", "unknown"),
            "scale_condition": case.meta.get("scale_condition", "unknown"),
            "occlusion_condition": case.meta.get("occlusion_condition", "unknown"),
            "reference_area_fraction": case.meta.get("reference_area_fraction"),
            "target_area_fraction": case.meta.get("target_area_fraction"),
            "found": bool(result.found),
            "confidence": float(result.confidence),
            "wall_seconds": float(wall),
            "stage_seconds": stages,
            "peak_gpu_mb": _peak_memory(pipeline),
            "device": str(getattr(pipeline, "device", "unknown")),
            **metric_values,
        }
        write_json(record_path, record)
        records.append(record)

        iou = record.get("iou")
        iou_text = "  n/a" if iou is None else f"IoU {iou:.3f}"
        print(f"[{position:3d}/{len(cases)}] {case.name:14s} {str(case.meta.get('category')):<14s} "
              f"{iou_text:12s} conf {record['confidence']:.3f} {wall:6.2f}s")

    records.sort(key=lambda item: str(item.get("sample_id")))
    scored = [record for record in records if record.get("iou") is not None]
    grouped = aggregate_rows(scored, by=("category", "scale_condition", "identity_level"))
    grouped["iou_histogram"] = iou_histogram([record["iou"] for record in scored])
    grouped["runtime"] = {
        "wall": summarize_times([record["wall_seconds"] for record in records]),
        "per_stage": {
            stage: summarize_times([record["stage_seconds"][stage] for record in records
                                    if stage in (record.get("stage_seconds") or {})])
            for stage in ("features", "correspondence", "localization", "segmentation")
        },
    }
    grouped["found_rate"] = (
        float(np.mean([record["found"] for record in records])) if records else 0.0
    )
    grouped["split"] = options.split
    grouped["method"] = options.method
    grouped["samples"] = len(records)
    grouped["scored"] = len(scored)
    grouped["reused_from_cache"] = reused
    grouped["elapsed_seconds"] = round(time.time() - started, 1)

    csv_path = write_csv(os.path.join(options.output, "results.csv"), [_csv_row(record) for record in records])
    write_json(os.path.join(options.output, "summary.json"), grouped)
    write_json(os.path.join(options.output, "provenance.json"), build_provenance(
        pipeline=pipeline, device=getattr(pipeline, "device", None), seed=options.seed,
        dataset={
            "root": options.root,
            "split": options.split,
            "manifest": manifest,
            "samples": len(records),
            "dataset": records[0].get("dataset") if records else None,
            "identity_level": records[0].get("identity_level") if records else None,
        },
        extra={"method": options.method, "elapsed_seconds": grouped["elapsed_seconds"]},
    ))

    figures = _write_figures(records, options, grouped)
    grouped["artefacts"] = {"csv": csv_path, "figures": figures}
    print(f"\nwrote {csv_path}")
    print(f"scored {len(scored)}/{len(records)} pairs, found rate {grouped['found_rate'] * 100:.0f}%, "
          f"reused {reused} cached, {grouped['elapsed_seconds']}s")
    if scored:
        overall = grouped["overall"]["iou"]
        print(f"IoU mean {overall['mean']:.3f} median {overall['median']:.3f} "
              f"std {overall['std']:.3f} | >=0.5 {overall['success_rate_at_0_5'] * 100:.0f}% "
              f">=0.7 {overall['success_rate_at_0_7'] * 100:.0f}%")
    return grouped


def _peak_memory(pipeline: Any) -> Optional[float]:
    from evaluation.runtime import gpu_memory_mb

    return gpu_memory_mb(getattr(pipeline, "device", None))


def _write_figures(records: Sequence[Dict[str, Any]], options: BenchmarkOptions,
                   grouped: Dict[str, Any]) -> Dict[str, str]:
    """Save the run-level figures. Never allowed to lose the numbers."""
    from visualization.benchmark import plot_benchmark_summary

    directory = os.path.join(options.output, "figures")
    os.makedirs(directory, exist_ok=True)
    written: Dict[str, str] = {}
    try:
        written.update(plot_benchmark_summary(records, directory, method=options.method))
    except Exception as error:  # noqa: BLE001 - figures must not lose the numbers
        print(f"warning: run-level figures failed: {error}")
    return written


def save_pair_figures(options: BenchmarkOptions, pipeline: Any = None,
                      limit: Optional[int] = None, worst: bool = True) -> List[str]:
    """Re-run pairs to draw the seven-panel figure, worst first.

    Separate from the scoring loop on purpose: figures are the expensive optional
    part, and a reader usually wants the failures rather than an arbitrary few.
    """
    import matplotlib

    matplotlib.use("Agg")

    from pipeline.one_shot_segmentation import get_default_pipeline
    from visualization.benchmark import save_benchmark_pair

    names = ("train", "validation", "test") if options.split == "all" else (options.split,)
    cases: List[Any] = []
    seen: set = set()
    for name in names:
        path = os.path.join(options.root, "splits", f"{name}.json")
        if not os.path.isfile(path):
            continue
        for case in cases_from_manifest(path):
            if case.name not in seen:
                seen.add(case.name)
                cases.append(case)
    if not cases:
        raise SystemExit(f"no cases found under {options.root}/splits for {options.split}")
    records = []
    for record_dir in sorted(os.listdir(os.path.join(options.output, "samples"))):
        if not record_dir.endswith(".json"):
            continue
        import json

        with open(os.path.join(options.output, "samples", record_dir), "r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    scored = [record for record in records if record.get("iou") is not None]
    scored.sort(key=lambda item: float(item["iou"]))
    chosen = scored if worst else scored[::-1]
    if limit is not None:
        chosen = chosen[:limit]

    by_name = {case.name: case for case in cases}
    if pipeline is None:
        pipeline = get_default_pipeline(
            device=options.device,
            encoder_kwargs={"model_id": options.encoder_model},
            segmenter_kwargs={"model_id": options.segmenter_model},
        )
    directory = os.path.join(options.output, "figures")
    written: List[str] = []
    for record in chosen:
        case = by_name.get(str(record["sample_id"]))
        if case is None:
            continue
        result = pipeline.segment_object(case.reference_image, case.reference_mask,
                                         case.target_image)
        metrics = evaluate_mask(result.predicted_mask, case.target_mask) if case.scorable else None
        caption = (f"{case.name} | {case.meta.get('category')} | "
                   f"scale: {case.meta.get('scale_condition')} | "
                   f"identity: {case.meta.get('identity_level')}")
        written.append(save_benchmark_pair(
            os.path.join(directory, f"{case.name}.png"),
            reference_image=case.reference_image,
            reference_mask=case.reference_mask,
            target_image=case.target_image,
            similarity_map=result.similarity_map,
            candidate_region=result.candidate_region,
            predicted_mask=result.predicted_mask,
            target_mask=case.target_mask,
            metrics=metrics,
            caption=caption,
            confidence=float(result.confidence),
        ))
        print(f"figure {written[-1]}")
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    import matplotlib

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data", help="dataset root (default: data)")
    parser.add_argument("--split", default="test",
                        choices=("train", "validation", "test", "all"),
                        help="which split to score; 'all' concatenates the three")
    parser.add_argument("--output", default=None, help="results directory (default results/<split>)")
    parser.add_argument("--method", default="dinov2-base+sam-base")
    parser.add_argument("--device", default=None, help="cuda, cpu, or omit for auto")
    parser.add_argument("--encoder-model", default="facebook/dinov2-base")
    parser.add_argument("--segmenter-model", default="facebook/sam-vit-base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N cases")
    parser.add_argument("--figures", type=int, default=8,
                        help="draw per-pair figures for the N worst pairs (0 to skip)")
    parser.add_argument("--no-resume", action="store_true", help="recompute every sample")
    parser.add_argument("--strict-integrity", action="store_true",
                        help="treat integrity warnings as failures")
    arguments = parser.parse_args(argv)

    output = arguments.output or os.path.join("results", arguments.split)
    options = BenchmarkOptions(
        root=arguments.root, split=arguments.split, output=output, method=arguments.method,
        device=arguments.device, encoder_model=arguments.encoder_model,
        segmenter_model=arguments.segmenter_model, seed=arguments.seed, limit=arguments.limit,
        figures=arguments.figures, resume=not arguments.no_resume,
        strict_integrity=arguments.strict_integrity,
    )
    environment = describe_environment(arguments.device)
    print(f"device: {environment.get('device')} | cuda: {environment.get('cuda_available')} "
          f"| git: {str(environment.get('git_revision'))[:8]}")
    run_split(options)
    if options.figures:
        save_pair_figures(options, limit=options.figures, worst=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
