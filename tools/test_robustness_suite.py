"""Tests for the Phase 3 robustness suite: ladders, seeds, runner, tables.

The robustness protocol is only trustworthy if three things hold. The ladders
must be exactly the documented sizes, so a missing variant cannot quietly shrink
a severity level. Every transformation must be reproducible from its recorded
seed, so a rebuild can be verified without re-running it. And the runner's
tables must obey the CSV contract and compute retention against the clean
baseline correctly, because a mislabelled severity column would poison every
figure drawn later.

`TestVariantLadders` and `TestApplyDeterminism` check the augmentation package
on synthetic images. `TestSelection` pins the evenly spaced baseline-rank
selection. `TestRunner` drives `run_robustness` over a freshly built synthetic
dataset with a stub pipeline - no model, no downloads - so resume, sharding,
retention, visible-region scoring and the artefact layout are all exercised.

    python tools/test_robustness_suite.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLBACKEND", "Agg")

from augmentation import registry  # noqa: E402
from augmentation.common import FAMILIES, derive_seed  # noqa: E402
from tools.build_robustness import build  # noqa: E402
from tools.run_robustness import (  # noqa: E402
    CSV_COLUMNS,
    RobustnessOptions,
    _csv_row,
    run_robustness,
    select_samples,
    severity_table,
    write_csv,
)
from tools.test_dataset_integrity import write_sample  # noqa: E402

REQUIRED_CSV_PREFIX = ("sample_id", "condition", "severity", "method", "iou",
                       "dice", "precision", "recall", "runtime")

EXPECTED_COUNTS = {"control": 1, "illumination": 19, "scale": 6, "occlusion": 5,
                   "viewpoint": 4, "combined": 8}


def synthetic_sample(height: int = 96, width: int = 120) -> Dict[str, np.ndarray]:
    """A gradient image with one rectangular object, large enough to survive
    every transformation without padding or clamping surprises."""
    ramp = np.linspace(30, 220, width, dtype=np.uint8)
    image = np.repeat(ramp[None, :, None], height, axis=0)
    image = np.repeat(image, 3, axis=2)
    mask = np.zeros((height, width), dtype=bool)
    mask[24:72, 30:90] = True
    return {"image": image, "mask": mask}


class TestVariantLadders(unittest.TestCase):
    """The documented ladder sizes and their ordering."""

    def test_every_family_has_its_documented_size(self) -> None:
        for family, expected in EXPECTED_COUNTS.items():
            variants = registry.build_variants(family, sample_index=1)
            self.assertEqual(len(variants), expected, family)
        self.assertEqual(len(registry.build_variants("control")), 1)

    def test_a_sample_undergoes_exactly_43_variants_in_protocol_order(self) -> None:
        variants = registry.variants_for_sample("sample_0007")
        self.assertEqual(len(variants), 43)
        families = [variant.family for variant in variants]
        self.assertEqual(families, sorted(families, key=FAMILIES.index))

    def test_severities_increase_within_each_family(self) -> None:
        for family in ("occlusion", "viewpoint"):
            variants = [v for v in registry.build_variants(family)
                        if v.condition != "mirror"]
            severities = [variant.severity for variant in variants]
            self.assertEqual(severities, sorted(severities), family)
            self.assertEqual(severities, list(range(1, len(severities) + 1)))
        scale = registry.build_variants("scale")
        for direction in ("smaller", "larger"):
            severities = [v.severity for v in scale if v.condition == direction]
            self.assertEqual(severities, [1, 2, 3], direction)

    def test_scale_factors_are_ordered_by_direction(self) -> None:
        variants = registry.build_variants("scale")
        smaller = [float(v.parameters["requested_factor"])
                   for v in variants if v.condition == "smaller"]
        larger = [float(v.parameters["requested_factor"])
                  for v in variants if v.condition == "larger"]
        self.assertEqual(smaller, sorted(smaller, reverse=True))
        self.assertEqual(larger, sorted(larger))
        self.assertTrue(all(factor < 1.0 for factor in smaller))
        self.assertTrue(all(factor > 1.0 for factor in larger))

    def test_occlusion_targets_rise_from_10_to_50_percent(self) -> None:
        targets = [float(v.parameters["target_fraction"])
                   for v in registry.build_variants("occlusion")]
        self.assertEqual(targets, [0.10, 0.20, 0.30, 0.40, 0.50])

    def test_occluder_schedule_varies_with_the_sample(self) -> None:
        schedule_0 = [v.condition for v in registry.build_variants("occlusion", 0)]
        schedule_1 = [v.condition for v in registry.build_variants("occlusion", 1)]
        self.assertNotEqual(schedule_0, schedule_1,
                            "the occluder rotation must depend on the sample")
        self.assertEqual(set(schedule_0), {"rect", "ellipse", "polygon", "patch"})

    def test_viewpoint_conditions_are_yaw_and_mirror(self) -> None:
        conditions = {v.condition for v in registry.build_variants("viewpoint")}
        self.assertEqual(conditions, {"yaw", "mirror"})

    def test_combined_is_four_pairs_at_two_severities(self) -> None:
        variants = registry.build_variants("combined", sample_index=3)
        self.assertEqual(len(variants), 8)
        self.assertEqual({v.severity for v in variants}, {2, 3})
        names = sorted(v.variant for v in variants)
        self.assertEqual(names, sorted(
            f"{pair}_{level}"
            for pair in ("vp_scale", "vp_illum", "vp_occl", "illum_occl")
            for level in ("mild", "strong")))

    def test_unknown_family_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            registry.build_variants("weather")


class TestSeedsAndDeterminism(unittest.TestCase):
    """Reproducibility from the recorded seed, on synthetic pixels."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sample = synthetic_sample()
        cls.image = cls.sample["image"]
        cls.mask = cls.sample["mask"]

    def _apply(self, family: str, variant_name: str, seed: int = 7,
               sample_id: str = "sample_0042") -> Any:
        variants = {variant.variant: variant
                    for variant in registry.build_variants(family, registry.sample_index(sample_id))}
        variant = variants[variant_name]
        rng = registry.variant_rng(seed, sample_id, variant)
        return registry.apply_variant(family, self.image, self.mask, variant, rng)

    def test_derive_seed_is_stable_and_bounded(self) -> None:
        first = derive_seed(7, "sample_0001", "illumination", "warm_1")
        self.assertEqual(first, derive_seed(7, "sample_0001", "illumination", "warm_1"))
        self.assertNotEqual(first, derive_seed(7, "sample_0002", "illumination", "warm_1"))
        self.assertNotEqual(first, derive_seed(8, "sample_0001", "illumination", "warm_1"))
        self.assertTrue(0 <= first < 2**32)

    def test_variant_seed_matches_the_derived_scheme(self) -> None:
        variant = registry.build_variants("illumination")[0]
        expected = derive_seed(7, "sample_0001", variant.family, variant.variant)
        self.assertEqual(registry.variant_seed(7, "sample_0001", variant), expected)

    def test_two_builds_describe_identical_variants(self) -> None:
        for family in FAMILIES:
            first = [v.describe() for v in registry.build_variants(family, 5)]
            second = [v.describe() for v in registry.build_variants(family, 5)]
            self.assertEqual(first, second, family)

    def test_control_returns_the_inputs_untouched(self) -> None:
        variant = registry.CONTROL_VARIANT
        image, mask, info = registry.apply_variant(
            "control", self.image, self.mask, variant,
            registry.variant_rng(7, "sample_0001", variant))
        self.assertTrue(np.array_equal(image, self.image))
        self.assertTrue(np.array_equal(mask, self.mask))
        self.assertTrue(info["identity"])

    def test_same_seed_reproduces_the_same_pixels(self) -> None:
        for family, variant_name in (("illumination", "brightness_up_1"),
                                     ("scale", "smaller_2"),
                                     ("occlusion", "occlude_3"),
                                     ("viewpoint", "yaw_2"),
                                     ("combined", "vp_occl_strong")):
            image_a, mask_a, info_a = self._apply(family, variant_name, seed=11)
            image_b, mask_b, info_b = self._apply(family, variant_name, seed=11)
            self.assertTrue(np.array_equal(image_a, image_b), variant_name)
            self.assertTrue(np.array_equal(mask_a, mask_b), variant_name)
            self.assertEqual(json.dumps(info_a, default=str, sort_keys=True),
                             json.dumps(info_b, default=str, sort_keys=True),
                             variant_name)

    def test_illumination_never_touches_the_mask(self) -> None:
        for variant in registry.build_variants("illumination"):
            rng = registry.variant_rng(3, "sample_0001", variant)
            _, mask, _ = registry.apply_variant(
                "illumination", self.image, self.mask, variant, rng)
            self.assertTrue(np.array_equal(mask, self.mask), variant.variant)

    def test_scale_records_the_achieved_factor(self) -> None:
        for variant in registry.build_variants("scale"):
            rng = registry.variant_rng(3, "sample_0001", variant)
            _, mask, info = registry.apply_variant(
                "scale", self.image, self.mask, variant, rng)
            requested = float(variant.parameters["requested_factor"])
            achieved = float(info["achieved_factor"])
            self.assertGreater(achieved, 0.0, variant.variant)
            self.assertLess(abs(achieved / requested - 1.0), 0.15, variant.variant)
            self.assertEqual(mask.shape, self.mask.shape, variant.variant)

    def test_occlusion_reaches_its_target_coverage(self) -> None:
        from augmentation.occlusion import COVERAGE_TOLERANCE

        for variant in registry.build_variants("occlusion"):
            rng = registry.variant_rng(3, "sample_0001", variant)
            _, mask, info = registry.apply_variant(
                "occlusion", self.image, self.mask, variant, rng)
            target = float(variant.parameters["target_fraction"])
            achieved = float(info["achieved_fraction"])
            self.assertLessEqual(abs(achieved - target), COVERAGE_TOLERANCE + 1e-9,
                                 variant.variant)
            self.assertEqual(mask.sum(), self.mask.sum(),
                             "ground truth must stay unchanged under occlusion")
            self.assertIn("artifacts", info)

    def test_viewpoint_records_geometry_and_keeps_the_object(self) -> None:
        for variant in registry.build_variants("viewpoint"):
            rng = registry.variant_rng(3, "sample_0001", variant)
            _, mask, info = registry.apply_variant(
                "viewpoint", self.image, self.mask, variant, rng)
            self.assertGreater(mask.sum(), 0, variant.variant)
            self.assertEqual(mask.shape, self.mask.shape, variant.variant)
            if variant.condition == "yaw":
                self.assertIn("yaw_deg", info, variant.variant)
                self.assertIn("padding_capped", info, variant.variant)
                self.assertIn("visible_object_fraction", info, variant.variant)
                self.assertLessEqual(abs(info["yaw_deg"]), 40.0, variant.variant)
                self.assertGreaterEqual(info["visible_object_fraction"], 0.5,
                                        variant.variant)
            else:
                self.assertEqual(info["axis"], "horizontal", variant.variant)
                self.assertEqual(int(mask.sum()), int(self.mask.sum()),
                                 "a mirror must not change the object")

    def test_viewpoint_refuses_an_empty_mask(self) -> None:
        variant = registry.build_variants("viewpoint")[0]
        empty = np.zeros_like(self.mask)
        with self.assertRaises(ValueError):
            registry.apply_variant(
                "viewpoint", self.image, empty, variant,
                registry.variant_rng(7, "sample_0001", variant))

    def test_crop_zoom_refuses_an_empty_mask(self) -> None:
        variant = registry.build_variants("scale")[0]
        with self.assertRaises(ValueError):
            registry.apply_variant(
                "scale", self.image, np.zeros_like(self.mask), variant,
                registry.variant_rng(7, "sample_0001", variant))


class TestSelection(unittest.TestCase):
    """Evenly spaced baseline-rank selection, written once and reused."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="selection-")
        self.baseline = os.path.join(self.root, "results", "baseline")
        samples = os.path.join(self.baseline, "samples")
        os.makedirs(samples, exist_ok=True)
        for index, iou in enumerate((0.10, 0.20, 0.30, 0.40, 0.50), start=1):
            with open(os.path.join(samples, f"sample_{index:04d}.json"),
                      "w", encoding="utf-8") as handle:
                json.dump({"sample_id": f"sample_{index:04d}", "iou": iou,
                           "dice": iou}, handle)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_picks_evenly_spaced_ranks(self) -> None:
        picked = select_samples(self.root, self.baseline, size=3)
        self.assertEqual(picked, ["sample_0001", "sample_0003", "sample_0005"])
        path = os.path.join(self.root, "processed", "selection.json")
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertIn("rationale", stored)
        self.assertEqual(stored["selection_size"], 3)
        self.assertEqual(stored["samples"][1]["baseline_iou"], 0.30)

    def test_the_choice_is_reused_until_reselected(self) -> None:
        first = select_samples(self.root, self.baseline, size=3)
        with open(os.path.join(self.baseline, "samples", "sample_0001.json"),
                  "w", encoding="utf-8") as handle:
            json.dump({"sample_id": "sample_0001", "iou": 0.99, "dice": 0.99}, handle)
        second = select_samples(self.root, self.baseline, size=3)
        self.assertEqual(first, second, "the recorded selection must be stable")
        third = select_samples(self.root, self.baseline, size=3, reselect=True)
        self.assertNotEqual(first, third, "reselect must re-read the baseline")
        with open(os.path.join(self.root, "processed", "selection.json"),
                  encoding="utf-8") as handle:
            stored = json.load(handle)
        by_id = {entry["sample_id"]: entry for entry in stored["samples"]}
        self.assertAlmostEqual(by_id["sample_0001"]["baseline_iou"], 0.99)
        self.assertEqual(by_id["sample_0001"]["rank"], 4)

    def test_missing_baseline_stops_selection(self) -> None:
        with self.assertRaises(SystemExit):
            select_samples(self.root, os.path.join(self.root, "nope"), size=2)

    def test_more_samples_than_scored_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            select_samples(self.root, self.baseline, size=9)


class _StubResult:
    def __init__(self, height: int = 40, width: int = 48) -> None:
        predicted = np.zeros((height, width), dtype=bool)
        predicted[8:18, 13:23] = True
        self.predicted_mask = predicted
        self.confidence = 0.75
        self.found = True
        self.runtime = {"features": 0.01, "correspondence": 0.02, "localization": 0.01,
                        "segmentation": 0.03, "total": 0.07}


class _StubPipeline:
    """Scores every variant identically; the runner's plumbing is the target."""

    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def segment_object(self, reference_image: Any, reference_mask: Any,
                       target_image: Any) -> _StubResult:
        self.calls += 1
        height, width = np.asarray(target_image).shape[:2]
        return _StubResult(height, width)


class TestRunner(unittest.TestCase):
    """`run_robustness` over a freshly built synthetic dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = tempfile.mkdtemp(prefix="robust-runner-")
        cls.baseline = os.path.join(cls.root, "results", "baseline")
        write_sample(os.path.join(cls.root, "samples", "sample_0001"),
                     "sample_0001", image_ids=(1, 2))
        write_sample(os.path.join(cls.root, "samples", "sample_0002"),
                     "sample_0002", image_ids=(3, 4))
        os.makedirs(os.path.join(cls.baseline, "samples"), exist_ok=True)
        for sample_id, iou in (("sample_0001", 0.40), ("sample_0002", 0.60)):
            with open(os.path.join(cls.baseline, "samples", f"{sample_id}.json"),
                      "w", encoding="utf-8") as handle:
                json.dump({"sample_id": sample_id, "iou": iou, "dice": iou}, handle)
        build(root=cls.root, seed=7, families=FAMILIES, limit=None,
              sample_ids=None, force=False, dry_run=False)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self) -> None:
        self.output = os.path.join(self.root, "results", self._testMethodName)
        self.pipeline = _StubPipeline()

    def _options(self, **overrides: Any) -> RobustnessOptions:
        arguments: Dict[str, Any] = {
            "root": self.root, "output": self.output, "baseline": self.baseline,
            "selection": 2, "seed": 0,
        }
        arguments.update(overrides)
        return RobustnessOptions(**arguments)

    def _records(self) -> List[Dict[str, Any]]:
        directory = os.path.join(self.output, "samples")
        records: List[Dict[str, Any]] = []
        for name in sorted(os.listdir(directory)):
            with open(os.path.join(directory, name), encoding="utf-8") as handle:
                records.append(json.load(handle))
        self.assertTrue(records)
        return records

    def test_run_writes_every_artefact(self) -> None:
        summary = run_robustness(self._options(), pipeline=self.pipeline)
        self.assertEqual(summary["records"], 86)
        self.assertEqual(summary["scored"], 86)
        self.assertEqual(summary["reused_from_cache"], 0)
        self.assertEqual(self.pipeline.calls, 86)
        self.assertEqual(summary["samples_with_records"],
                         ["sample_0001", "sample_0002"])

        csv_path = os.path.join(self.output, "illumination.csv")
        self.assertTrue(os.path.isfile(csv_path))
        with open(csv_path, "r", encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
        self.assertEqual(header, list(CSV_COLUMNS))
        self.assertEqual(tuple(header[:9]), REQUIRED_CSV_PREFIX)

        for name in ("summary.json", "provenance.json", "progress.json"):
            self.assertTrue(os.path.isfile(os.path.join(self.output, name)), name)
        self.assertTrue(os.path.isfile(
            os.path.join(self.root, "processed", "selection.json")))
        for family in FAMILIES:
            self.assertTrue(os.path.isfile(os.path.join(self.output, f"{family}.csv")),
                            family)

        with open(os.path.join(self.output, "provenance.json"), encoding="utf-8") as handle:
            provenance = json.load(handle)
        self.assertEqual(provenance["seed"], 0)
        self.assertEqual(provenance["dataset"]["records"], 86)

        with open(os.path.join(self.output, "summary.json"), encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertEqual(len(stored["severity_table"]), 48)
        self.assertAlmostEqual(stored["overall"]["iou"]["mean"],
                               summary["overall"]["iou"]["mean"])

    def test_retention_is_the_ratio_to_the_clean_baseline(self) -> None:
        run_robustness(self._options(), pipeline=self.pipeline)
        records = {(record["family"], record["original_sample_id"]): record
                   for record in self._records()}
        control = records[("control", "sample_0001")]
        self.assertAlmostEqual(control["clean_iou"], 0.40)
        expected = control["iou"] / 0.40
        self.assertAlmostEqual(control["retention"], expected, places=6)
        scaling = records[("scale", "sample_0002")]
        self.assertAlmostEqual(scaling["retention"], scaling["iou"] / 0.60, places=6)

    def test_only_occlusion_rows_carry_visible_metrics(self) -> None:
        run_robustness(self._options(), pipeline=self.pipeline)
        for record in self._records():
            if record["family"] == "occlusion":
                self.assertIsNotNone(record["iou_visible"], record["sample_id"])
                self.assertIsNotNone(record["achieved_fraction"], record["sample_id"])
                self.assertIsNotNone(record["occluder_type"], record["sample_id"])
            else:
                self.assertIsNone(record["iou_visible"], record["sample_id"])

    def test_records_carry_provenance_columns(self) -> None:
        run_robustness(self._options(), pipeline=self.pipeline)
        record = next(record for record in self._records()
                      if record["family"] == "viewpoint"
                      and record["condition"] == "yaw")
        self.assertEqual(record["method"], "dinov2-base+sam-base")
        self.assertEqual(record["family"], "viewpoint")
        self.assertEqual(record["condition"], "yaw")
        self.assertIsInstance(record["severity"], int)
        self.assertIsInstance(record["random_seed"], int)
        self.assertIn("yaw_deg", record)
        row = _csv_row(record)
        for column in ("iou", "runtime", "feature_time", "retention",
                       "clean_iou", "random_seed", "visible_object_fraction"):
            self.assertIn(column, row)

    def test_resume_reuses_cached_records(self) -> None:
        run_robustness(self._options(), pipeline=self.pipeline)
        self.assertEqual(self.pipeline.calls, 86)

        summary = run_robustness(self._options(), pipeline=self.pipeline)
        self.assertEqual(summary["reused_from_cache"], 86)
        self.assertEqual(self.pipeline.calls, 86, "a cached record must not be re-run")

        summary = run_robustness(self._options(force=True), pipeline=self.pipeline)
        self.assertEqual(summary["reused_from_cache"], 0)
        self.assertEqual(self.pipeline.calls, 172, "force must re-score every record")

    def test_limit_scores_only_the_first_cases(self) -> None:
        summary = run_robustness(self._options(limit=5), pipeline=self.pipeline)
        self.assertEqual(summary["records"], 5)
        self.assertEqual(self.pipeline.calls, 5)

    def test_sharding_splits_the_selected_samples(self) -> None:
        first = run_robustness(self._options(shard=(1, 2)), pipeline=self.pipeline)
        self.assertEqual(first["samples_with_records"], ["sample_0001"])
        self.assertEqual(first["records"], 43)

        second_pipeline = _StubPipeline()
        second = run_robustness(self._options(output=self.output + "-b", shard=(2, 2)),
                                pipeline=second_pipeline)
        self.assertEqual(second["samples_with_records"], ["sample_0002"])
        self.assertEqual(second["records"], 43)

    def test_report_only_reassembles_without_a_pipeline(self) -> None:
        run_robustness(self._options(), pipeline=self.pipeline)
        summary = run_robustness(self._options(report_only=True))
        self.assertEqual(summary["records"], 86)
        self.assertEqual(len(summary["groups"]["family"]), 6)

    def test_missing_manifest_stops_the_run(self) -> None:
        manifest = os.path.join(self.root, "processed", "viewpoint", "manifest.json")
        archived = manifest + ".bak"
        os.replace(manifest, archived)
        try:
            with self.assertRaises(SystemExit):
                run_robustness(self._options(), pipeline=self.pipeline)
            self.assertEqual(self.pipeline.calls, 0)
        finally:
            os.replace(archived, manifest)

    def test_missing_baseline_stops_the_run(self) -> None:
        with self.assertRaises(SystemExit):
            run_robustness(self._options(baseline=os.path.join(self.root, "absent")),
                           pipeline=self.pipeline)
        self.assertEqual(self.pipeline.calls, 0)

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
        self.assertEqual(values["retention"], "")


class TestSeverityTable(unittest.TestCase):
    """The (family, condition, severity) aggregation used by the curves."""

    def test_groups_rows_and_ignores_unscored_ones(self) -> None:
        records = [
            {"family": "scale", "condition": "smaller", "severity": 1,
             "iou": 0.4, "retention": 0.8, "found": True},
            {"family": "scale", "condition": "smaller", "severity": 1,
             "iou": 0.6, "retention": 1.2, "found": False},
            {"family": "scale", "condition": "smaller", "severity": 2,
             "iou": 0.2, "retention": 0.4, "found": True},
            {"family": "control", "condition": "identity", "severity": 0,
             "iou": None, "retention": None, "found": False},
        ]
        rows = severity_table(records)
        self.assertEqual(len(rows), 2)
        first = rows[0]
        self.assertEqual(first["count"], 2)
        self.assertAlmostEqual(first["mean_iou"], 0.5)
        self.assertAlmostEqual(first["mean_retention"], 1.0)
        self.assertAlmostEqual(first["found_rate"], 0.5)
        self.assertEqual(rows[1]["severity"], 2)

    def test_empty_input_is_an_empty_table(self) -> None:
        self.assertEqual(severity_table([]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
