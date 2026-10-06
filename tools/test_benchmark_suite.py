"""Tests for the Phase 2 benchmark: aggregation, provenance, figures, runner.

The benchmark only answers something if the plumbing around the model is right,
and none of it can be checked by looking at a result file:

`TestAggregation` pins the statistics that end up in `summary.json`, because a
grouped mean that silently pools a single pair into `other` is not visible in a
figure. `TestRuntime` checks the provenance helpers that make two runs
comparable. `TestFigures` renders every plot, including the failure panel with
no prediction, which is where matplotlib problems hide. `TestRunner` drives
`run_split` over a synthetic dataset with a stub pipeline, so resume, the
integrity gate, the CSV contract and the artefact layout are all exercised
without downloading a model.

    python tools/test_benchmark_suite.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLBACKEND", "Agg")

from evaluation.metrics import (  # noqa: E402
    SUCCESS_THRESHOLDS,
    aggregate_rows,
    evaluate_mask,
    iou_histogram,
    summarize_values,
)
from evaluation.runtime import (  # noqa: E402
    Stopwatch,
    build_provenance,
    describe_environment,
    git_revision,
    gpu_memory_mb,
    package_version,
    set_seed,
    summarize_times,
)
from tools.build_benchmark import write_splits  # noqa: E402
from tools.run_benchmark import (  # noqa: E402
    CSV_COLUMNS,
    BenchmarkOptions,
    _csv_row,
    run_split,
    write_csv,
)
from tools.test_dataset_integrity import write_sample  # noqa: E402
from visualization.benchmark import (  # noqa: E402
    plot_benchmark_pair,
    plot_benchmark_summary,
    plot_group_means,
    plot_iou_distribution,
    plot_stage_runtime,
    save_benchmark_pair,
)


class TestAggregation(unittest.TestCase):
    """The numbers written into `summary.json`."""

    def test_empty_values_summarise_to_nothing(self) -> None:
        self.assertEqual(summarize_values([]), {})
        self.assertEqual(iou_histogram([])["counts"], [])

    def test_known_distribution(self) -> None:
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        summary = summarize_values(values)
        self.assertEqual(summary["count"], 5)
        self.assertAlmostEqual(summary["mean"], 0.5)
        self.assertAlmostEqual(summary["median"], 0.5)
        self.assertAlmostEqual(summary["min"], 0.0)
        self.assertAlmostEqual(summary["max"], 1.0)
        self.assertAlmostEqual(summary["q25"], 0.25)
        self.assertAlmostEqual(summary["q75"], 0.75)
        self.assertAlmostEqual(summary["std"], float(np.std(values)))
        for threshold in SUCCESS_THRESHOLDS:
            key = f"success_rate_at_{threshold:g}".replace(".", "_")
            expected = float(np.mean(np.asarray(values) >= threshold))
            self.assertAlmostEqual(summary[key], expected, msg=key)

    def test_threshold_counts_match_the_histogram(self) -> None:
        values = [0.1, 0.6, 0.8]
        histogram = iou_histogram(values)
        self.assertEqual(sum(histogram["counts"]), 3)
        self.assertEqual(histogram["count"], 3)
        self.assertEqual(histogram["thresholds"]["0.5"], 2)
        self.assertEqual(histogram["thresholds"]["0.75"], 1)
        self.assertEqual(histogram["thresholds"]["0.9"], 0)
        self.assertEqual(histogram["thresholds"]["0.25"], 2)
        self.assertEqual(len(histogram["bin_edges"]), len(histogram["counts"]) + 1)

    def test_small_groups_are_pooled(self) -> None:
        rows = (
            [{"category": "person", "iou": 1.0} for _ in range(4)]
            + [{"category": "giraffe", "iou": 0.0}]
        )
        payload = aggregate_rows(rows, by=("category",), min_group=2)
        self.assertAlmostEqual(payload["overall"]["iou"]["mean"], 0.8)
        self.assertNotIn("giraffe", payload["groups"]["category"])
        self.assertEqual(payload["groups"]["category"]["other"]["count"], 1)
        self.assertEqual(payload["groups"]["category"]["person"]["count"], 4)
        self.assertEqual(payload["rows"], 5)

    def test_groups_keep_every_value_when_min_group_is_one(self) -> None:
        rows = [{"category": "person", "iou": 0.4}, {"category": "bus", "iou": 0.9}]
        payload = aggregate_rows(rows, by=("category",), min_group=1)
        self.assertEqual(set(payload["groups"]["category"]), {"person", "bus"})


class TestRuntime(unittest.TestCase):
    """Seeds, timings and the provenance block."""

    def test_seed_repeats_the_same_draw(self) -> None:
        set_seed(7)
        first = float(np.random.rand())
        set_seed(7)
        second = float(np.random.rand())
        set_seed(8)
        third = float(np.random.rand())
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_stopwatch_records_a_stage(self) -> None:
        with Stopwatch() as watch:
            with watch.stage("features"):
                time.sleep(0.01)
            with watch.stage("segmentation"):
                time.sleep(0.01)
        record = watch.to_record()
        self.assertGreater(record.stages["features"], 0.0)
        self.assertGreater(record.stages["segmentation"], 0.0)
        self.assertAlmostEqual(record.total_seconds,
                               sum(record.stages.values()), places=9)
        self.assertEqual(record.device, "cpu")

    def test_nested_stages_are_rejected(self) -> None:
        watch = Stopwatch()
        with watch.stage("outer"):
            with self.assertRaises(RuntimeError):
                with watch.stage("inner"):
                    pass

    def test_cpu_memory_measurement_is_none_without_a_gpu(self) -> None:
        self.assertIsNone(gpu_memory_mb("cpu"))

    def test_summarise_times_of_nothing_is_empty(self) -> None:
        self.assertEqual(summarize_times([]), {})
        stats = summarize_times([1.0, 2.0, 3.0])
        self.assertAlmostEqual(stats["mean_seconds"], 2.0)
        self.assertAlmostEqual(stats["median_seconds"], 2.0)
        self.assertEqual(stats["count"], 3)

    def test_environment_records_the_machine(self) -> None:
        environment = describe_environment("cpu")
        self.assertEqual(environment["python"], sys.version.split()[0])
        self.assertIn("platform", environment)
        self.assertIn("numpy", environment["packages"])
        self.assertEqual(environment["packages"]["numpy"], np.__version__)
        self.assertEqual(environment["device"], "cpu")

    def test_provenance_carries_seed_dataset_and_environment(self) -> None:
        provenance = build_provenance(
            device="cpu", seed=0,
            dataset={"dataset": "unit-test", "samples": 2},
            extra={"method": "stub"},
        )
        self.assertEqual(provenance["seed"], 0)
        self.assertEqual(provenance["dataset"]["dataset"], "unit-test")
        self.assertEqual(provenance["method"], "stub")
        self.assertIn("created_utc", provenance)
        self.assertIn("packages", provenance["environment"])

    def test_git_revision_is_a_hash_or_none(self) -> None:
        revision = git_revision()
        self.assertTrue(revision is None or isinstance(revision, str))
        self.assertEqual(package_version("numpy"), np.__version__)
        self.assertIsNone(package_version("a-package-that-is-not-installed"))


class TestFigures(unittest.TestCase):
    """Every plot, including the ones only drawn when something went wrong."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp(prefix="figures-")
        self.image = np.full((40, 48, 3), 120, dtype=np.uint8)
        self.reference_mask = np.zeros((40, 48), dtype=bool)
        self.reference_mask[8:18, 10:20] = True
        self.target_mask = np.zeros((40, 48), dtype=bool)
        self.target_mask[8:18, 10:20] = True
        self.predicted = np.zeros((40, 48), dtype=bool)
        self.predicted[8:18, 13:23] = True

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _pair(self, **overrides: Any) -> Any:
        arguments: Dict[str, Any] = {
            "reference_image": self.image,
            "reference_mask": self.reference_mask,
            "target_image": self.image,
            "similarity_map": np.linspace(0.0, 1.0, 40 * 48).reshape(40, 48),
            "candidate_region": self.reference_mask,
            "predicted_mask": self.predicted,
            "target_mask": self.target_mask,
            "metrics": evaluate_mask(self.predicted, self.target_mask),
            "caption": "sample_0001 | person | scale: similar | identity: category",
            "confidence": 0.75,
        }
        arguments.update(overrides)
        return plot_benchmark_pair(**arguments)

    def test_pair_figure_has_one_axis_per_panel(self) -> None:
        figure = self._pair()
        self.assertEqual(len(figure.axes), 7)
        self.assertIn("person", figure._suptitle.get_text())
        import matplotlib.pyplot as plt

        plt.close(figure)

    def test_pair_figure_without_a_prediction_still_renders(self) -> None:
        figure = self._pair(predicted_mask=None, metrics=None, similarity_map=None,
                            candidate_region=None, confidence=None, caption="")
        self.assertEqual(len(figure.axes), 7)
        self.assertIn("NOT FOUND", figure.axes[5].get_title())
        self.assertIn("no ground truth", figure.axes[6].get_title())
        import matplotlib.pyplot as plt

        plt.close(figure)

    def test_empty_distribution_still_draws(self) -> None:
        figure = plot_iou_distribution([])
        self.assertEqual(len(figure.axes), 1)
        import matplotlib.pyplot as plt

        plt.close(figure)

    def test_group_and_runtime_plots(self) -> None:
        rows: List[Dict[str, Any]] = [
            {"sample_id": "a", "iou": 0.9, "category": "bus", "scale_condition": "similar",
             "stage_seconds": {"features": 1.0, "segmentation": 2.0}},
            {"sample_id": "b", "iou": 0.1, "category": "bus", "scale_condition": "smaller",
             "stage_seconds": {"features": 3.0, "segmentation": 4.0}},
        ]
        for column in ("scale_condition", "category"):
            figure = plot_group_means(rows, column=column)
            self.assertEqual(len(figure.axes), 1)
            import matplotlib.pyplot as plt

            plt.close(figure)
        figure = plot_stage_runtime(rows)
        self.assertEqual(len(figure.axes), 1)
        import matplotlib.pyplot as plt

        plt.close(figure)

    def test_summary_writes_the_three_run_figures(self) -> None:
        rows = [{"iou": 0.4, "category": "bus", "scale_condition": "similar",
                 "stage_seconds": {"features": 1.0, "segmentation": 2.0}}]
        written = plot_benchmark_summary(rows, self.directory, method="stub")
        self.assertEqual(len(written), 3)
        for path in written.values():
            if path:
                self.assertTrue(os.path.isfile(path), path)

    def test_saved_pair_figure_is_written_to_disk(self) -> None:
        path = os.path.join(self.directory, "sample_0001.png")
        saved = save_benchmark_pair(path, reference_image=self.image,
                                    reference_mask=self.reference_mask,
                                    target_image=self.image,
                                    predicted_mask=self.predicted,
                                    target_mask=self.target_mask)
        self.assertEqual(saved, path)
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 0)


class _StubResult:
    def __init__(self, height: int = 40, width: int = 48) -> None:
        predicted = np.zeros((height, width), dtype=bool)
        predicted[8:18, 13:23] = True
        candidate = np.zeros((height, width), dtype=bool)
        candidate[8:18, 10:20] = True
        self.predicted_mask = predicted
        self.candidate_region = candidate
        self.similarity_map = np.linspace(0.0, 1.0, height * width).reshape(height, width)
        self.confidence = 0.75
        self.found = True
        self.runtime = {"features": 0.01, "correspondence": 0.02, "localization": 0.01,
                        "segmentation": 0.03, "total": 0.07}


class _StubPipeline:
    """Stands in for the real pipeline so the runner can be tested offline."""

    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def segment_object(self, reference_image: Any, reference_mask: Any,
                       target_image: Any) -> _StubResult:
        self.calls += 1
        return _StubResult()


class TestRunner(unittest.TestCase):
    """`run_split` over a synthetic dataset, with no model anywhere near it."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="runner-")
        self.output = os.path.join(self.root, "results", "stub")
        entries = []
        for index in (1, 2):
            metadata = write_sample(
                os.path.join(self.root, "samples", f"sample_{index:04d}"),
                f"sample_{index:04d}", category="person", object_id=f"obj-{index}",
                image_ids=(index, index + 10))
            entries.append({key: metadata[key] for key in (
                "sample_id", "object_id", "dataset", "category", "identity_level",
                "reference_image", "reference_mask", "target_image", "target_mask",
                "viewpoint_condition", "illumination_condition", "scale_condition",
                "occlusion_condition")})
        write_splits({"train": entries, "validation": [], "test": []},
                     self.root, seed=0, quiet=True)
        self.pipeline = _StubPipeline()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _options(self, **overrides: Any) -> BenchmarkOptions:
        arguments: Dict[str, Any] = {
            "root": self.root, "split": "train", "output": self.output,
            "device": "cpu", "seed": 0, "figures": 0,
        }
        arguments.update(overrides)
        return BenchmarkOptions(**arguments)

    def test_run_writes_every_artefact(self) -> None:
        summary = run_split(self._options(), pipeline=self.pipeline)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["scored"], 2)
        self.assertEqual(summary["reused_from_cache"], 0)
        self.assertEqual(self.pipeline.calls, 2)

        csv_path = os.path.join(self.output, "results.csv")
        self.assertTrue(os.path.isfile(csv_path))
        with open(csv_path, "r", encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
        self.assertEqual(header, list(CSV_COLUMNS))

        for name in ("summary.json", "provenance.json"):
            self.assertTrue(os.path.isfile(os.path.join(self.output, name)), name)
        records = os.listdir(os.path.join(self.output, "samples"))
        self.assertEqual(sorted(records), ["sample_0001.json", "sample_0002.json"])
        self.assertTrue(os.path.isdir(os.path.join(self.output, "figures")))

        with open(os.path.join(self.output, "provenance.json"), encoding="utf-8") as handle:
            provenance = json.load(handle)
        self.assertEqual(provenance["seed"], 0)
        self.assertEqual(provenance["dataset"]["dataset"], "unit-test")

        with open(os.path.join(self.output, "summary.json"), encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertAlmostEqual(stored["overall"]["iou"]["mean"],
                               summary["overall"]["iou"]["mean"])
        self.assertGreater(stored["overall"]["iou"]["mean"], 0.0)

    def test_records_carry_the_condition_columns(self) -> None:
        run_split(self._options(), pipeline=self.pipeline)
        with open(os.path.join(self.output, "samples", "sample_0001.json"),
                  encoding="utf-8") as handle:
            record = json.load(handle)
        self.assertEqual(record["dataset"], "unit-test")
        self.assertEqual(record["identity_level"], "category")
        self.assertEqual(record["scale_condition"], "similar")
        self.assertEqual(record["viewpoint_condition"], "unknown")
        self.assertEqual(record["method"], "dinov2-base+sam-base")
        row = _csv_row(record)
        for column in ("iou", "runtime", "feature_time", "segmentation_time",
                       "scale_condition", "identity_level"):
            self.assertIn(column, row)

    def test_resume_reuses_cached_samples(self) -> None:
        run_split(self._options(), pipeline=self.pipeline)
        self.assertEqual(self.pipeline.calls, 2)

        summary = run_split(self._options(), pipeline=self.pipeline)
        self.assertEqual(summary["reused_from_cache"], 2)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(self.pipeline.calls, 2, "a cached pair must not be re-run")

        summary = run_split(self._options(force=True), pipeline=self.pipeline)
        self.assertEqual(summary["reused_from_cache"], 0)
        self.assertEqual(self.pipeline.calls, 4, "force must re-run every pair")

    def test_limit_scores_only_the_first_pairs(self) -> None:
        summary = run_split(self._options(limit=1), pipeline=self.pipeline)
        self.assertEqual(summary["samples"], 1)
        self.assertEqual(self.pipeline.calls, 1)

    def test_csv_writer_formats_numbers_and_booleans(self) -> None:
        path = os.path.join(self.root, "table.csv")
        write_csv(path, [_csv_row({"sample_id": "a", "iou": 0.123456789,
                                   "found": True, "confidence": None,
                                   "stage_seconds": {"total": 1.5}})])
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(len(lines), 2)
        values = dict(zip(CSV_COLUMNS, lines[1].split(",")))
        self.assertEqual(values["sample_id"], "a")
        self.assertEqual(values["iou"], "0.123457")
        self.assertEqual(values["runtime"], "1.5")
        self.assertEqual(values["found"], "1")
        self.assertEqual(values["confidence"], "")

    def test_broken_data_stops_the_run(self) -> None:
        os.remove(os.path.join(self.root, "samples", "sample_0001", "target_mask.png"))
        with self.assertRaises(SystemExit):
            run_split(self._options(), pipeline=self.pipeline)
        self.assertFalse(os.path.isfile(os.path.join(self.output, "results.csv")),
                         "nothing may be written when the data is broken")

    def test_missing_manifest_stops_the_run(self) -> None:
        os.remove(os.path.join(self.root, "splits", "test.json"))
        with self.assertRaises(SystemExit):
            run_split(self._options(split="test"), pipeline=self.pipeline)
        self.assertEqual(self.pipeline.calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
