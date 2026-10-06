"""Tests for the benchmark builder's format handling and the integrity checker.

Two independent things are checked here.

The builder turns a dataset's annotation format into pixels, and getting that
wrong would silently corrupt every sample: `TestMaskDecoding` pins the COCO RLE
decoder against a hand-computed example and against a round trip through the
polygon filler, so a future change cannot quietly shift every mask.

The integrity checker is only useful if it fails on damaged data, so
`TestIntegrityDetection` builds a tiny throwaway dataset and then breaks it in
each of the ways the checker promises to catch. Synthetic samples are used on
purpose: the tests stay fast and need no network.

    python tools/test_dataset_integrity.py
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

from evaluation.dataset_integrity import CRITICAL, WARNING, check_dataset  # noqa: E402
from tools.build_benchmark import (  # noqa: E402
    SCALE_RATIO_LARGE,
    annotation_to_mask,
    build_pairs,
    classify_scale,
    decode_compressed_counts,
    decode_rle,
    encode_compressed_counts,
    load_source_objects,
    split_samples,
    write_splits,
)


def make_image(path: str, width: int = 48, height: int = 40, colour: int = 120) -> None:
    from PIL import Image

    array = np.full((height, width, 3), colour, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def make_mask(path: str, box: tuple = (10, 8, 20, 18), size: tuple = (48, 40)) -> None:
    from PIL import Image

    width, height = size
    array = np.zeros((height, width), dtype=np.uint8)
    left, top, right, bottom = box
    array[top:bottom, left:right] = 255
    Image.fromarray(array, mode="L").save(path)


def write_sample(directory: str, sample_id: str, category: str = "person",
                 size: tuple = (48, 40), object_id: str = "obj-1",
                 image_ids: tuple = (1, 2)) -> dict:
    """Write one valid sample folder, with metadata derived from the real pixels."""
    os.makedirs(directory, exist_ok=True)
    box = (10, 8, 20, 18)
    for role in ("reference", "target"):
        make_image(os.path.join(directory, f"{role}.jpg"), width=size[0], height=size[1])
        make_mask(os.path.join(directory, f"{role}_mask.png"), box=box, size=size)
    mask = np.asarray(_read_mask(os.path.join(directory, "reference_mask.png"))) > 127
    area_fraction = float(mask.mean())
    bbox = [10.0, 8.0, 10.0, 10.0]
    metadata = {
        "sample_id": sample_id,
        "object_id": object_id,
        "identity_level": "category",
        "dataset": "unit-test",
        "category": category,
        "reference_image": f"samples/{sample_id}/reference.jpg",
        "reference_mask": f"samples/{sample_id}/reference_mask.png",
        "target_image": f"samples/{sample_id}/target.jpg",
        "target_mask": f"samples/{sample_id}/target_mask.png",
        "viewpoint_condition": "unknown",
        "illumination_condition": "unknown",
        "scale_condition": "similar",
        "occlusion_condition": "unknown",
        "measured": {
            "reference_area_fraction": area_fraction,
            "target_area_fraction": area_fraction,
            "reference_bbox": bbox,
            "target_bbox": bbox,
        },
        "source": {"image_id": list(image_ids), "annotation_id": [11, 12]},
    }
    with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def _read_mask(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path))


class TestMaskDecoding(unittest.TestCase):
    """The COCO decoders must produce exactly the pixels the format describes."""

    def test_uncompressed_rle_is_column_major(self) -> None:
        height, width = 3, 2
        counts = [1, 3, 2, 0]
        mask = decode_rle({"size": [height, width], "counts": counts})
        self.assertEqual(mask.shape, (height, width))
        flat = mask.reshape(-1, order="F").tolist()
        self.assertEqual(flat, [False, True, True, True, False, False])

    def test_rle_round_trip_through_compression(self) -> None:
        expected = np.zeros((5, 4), dtype=bool)
        expected[1:4, 1:3] = True
        expected[0, 3] = True
        flat = expected.reshape(-1, order="F")
        counts = []
        previous = bool(flat[0])
        run = 0
        for value in flat:
            if bool(value) == previous:
                run += 1
            else:
                counts.append(run)
                previous = bool(value)
                run = 1
        counts.append(run)
        decoded = decode_rle({"size": [5, 4], "counts": counts})
        self.assertTrue(np.array_equal(decoded, expected))

    def test_compressed_counts_round_trip(self) -> None:
        for counts in ([0, 5, 3, 0, 2], [1], [10, 0, 7], list(range(0, 40, 3))):
            text = encode_compressed_counts(counts)
            self.assertEqual(len(text) % 5, 0)
            self.assertEqual(decode_compressed_counts(text), list(counts))

    def test_compressed_counts_are_little_endian_in_six_bit_symbols(self) -> None:
        text = encode_compressed_counts([1])
        value = ((ord(text[0]) - 48) & 0x3F) | (((ord(text[1]) - 48) & 0x3F) << 6)
        self.assertEqual(value, 2)
        self.assertEqual(decode_compressed_counts(text), [1])

    def test_negative_runs_round_trip(self) -> None:
        text = encode_compressed_counts([0, -4, 9, -9])
        self.assertEqual(decode_compressed_counts(text), [0, -4, 9, -9])

    def test_list_counts_pass_through(self) -> None:
        self.assertEqual(decode_compressed_counts([3, 4, 5]), [3, 4, 5])

    def test_polygon_fill_covers_the_box(self) -> None:
        polygon = [10, 8, 20, 8, 20, 18, 10, 18]
        mask = annotation_to_mask([polygon], 40, 48)
        self.assertEqual(mask.shape, (40, 48))
        self.assertGreater(mask.mean(), 0.05)
        self.assertLess(mask.mean(), 0.07)
        rows, columns = np.nonzero(mask)
        self.assertAlmostEqual(int(rows.min()), 8, delta=1)
        self.assertAlmostEqual(int(rows.max()), 17, delta=1)
        self.assertAlmostEqual(int(columns.min()), 10, delta=1)
        self.assertAlmostEqual(int(columns.max()), 19, delta=1)

    def test_multiple_polygons_are_unioned(self) -> None:
        mask = annotation_to_mask([[0, 0, 5, 0, 5, 5], [10, 10, 15, 10, 15, 15]], 20, 20)
        self.assertTrue(mask[2, 2])
        self.assertTrue(mask[12, 12])
        self.assertFalse(mask[7, 7])

    def test_rle_annotation_resized_to_image(self) -> None:
        rle = {"size": [2, 2], "counts": [0, 4]}
        mask = annotation_to_mask(rle, 20, 20)
        self.assertEqual(mask.shape, (20, 20))
        self.assertGreater(mask.mean(), 0.9)


def _annotation_file(path: str, entries: list) -> str:
    coco = {
        "images": [{"id": entry["image_id"], "file_name": entry["file_name"],
                    "height": entry["height"], "width": entry["width"]}
                   for entry in entries],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
        "annotations": [{"id": entry["annotation_id"], "image_id": entry["image_id"],
                         "category_id": entry["category_id"], "area": entry["area"],
                         "bbox": entry["bbox"], "iscrowd": entry.get("iscrowd", 0),
                         "segmentation": entry["segmentation"]} for entry in entries],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(coco, handle)
    return path


class TestPairSelection(unittest.TestCase):
    """Pairing must be deterministic, balanced, and never reuse one photograph."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp(prefix="pairs-")
        self.path = _annotation_file(os.path.join(self.directory, "a.json"), [
            {"annotation_id": index, "image_id": index, "category_id": 1 if index < 6 else 2,
             "file_name": f"{index:012d}.jpg", "height": 100, "width": 100,
             "area": 500 + index * 400, "bbox": [10, 10, 50, 50],
             "segmentation": [[10, 10, 60, 10, 60, 60]]}
            for index in range(1, 13)
        ])

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_area_and_crowd_filters(self) -> None:
        objects, _ = load_source_objects(self.path, 0.005, 0.6)
        self.assertEqual(len(objects), 12)
        tight, _ = load_source_objects(self.path, 0.9, 1.0)
        self.assertEqual(tight, [])

    def test_pairs_never_share_an_image(self) -> None:
        objects, _ = load_source_objects(self.path, 0.005, 0.6)
        for pair in build_pairs(objects, 8, seed=0):
            self.assertNotEqual(pair.reference.image_id, pair.target.image_id)
            self.assertEqual(pair.reference.category_id, pair.target.category_id)

    def test_selection_is_reproducible(self) -> None:
        objects, _ = load_source_objects(self.path, 0.005, 0.6)
        first = [(pair.reference.annotation_id, pair.target.annotation_id)
                 for pair in build_pairs(objects, 6, seed=7)]
        second = [(pair.reference.annotation_id, pair.target.annotation_id)
                  for pair in build_pairs(objects, 6, seed=7)]
        other = [(pair.reference.annotation_id, pair.target.annotation_id)
                 for pair in build_pairs(objects, 6, seed=8)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_scale_condition_uses_measured_area(self) -> None:
        objects, _ = load_source_objects(self.path, 0.005, 0.6)
        by_id = {item.annotation_id: item for item in objects}
        reference = by_id[1]
        target = by_id[11]
        ratio_expected = target.area_fraction / reference.area_fraction
        condition, ratio = classify_scale(reference, target)
        self.assertGreater(ratio_expected, SCALE_RATIO_LARGE)
        self.assertEqual(condition, "larger_in_target")
        self.assertAlmostEqual(ratio, ratio_expected, places=6)
        self.assertEqual(classify_scale(reference, reference)[0], "similar")

    def test_splits_are_disjoint_and_complete(self) -> None:
        entries = [{"sample_id": f"sample_{index:04d}"} for index in range(1, 11)]
        buckets = split_samples(entries, seed=0)
        names = [entry["sample_id"] for items in buckets.values() for entry in items]
        self.assertEqual(sorted(names), sorted(entry["sample_id"] for entry in entries))
        self.assertEqual(sum(len(items) for items in buckets.values()), 10)
        again = split_samples(entries, seed=0)
        self.assertEqual(buckets, again)

    def test_written_splits_resolve_from_their_own_directory(self) -> None:
        root = tempfile.mkdtemp(prefix="splits-")
        try:
            entries = []
            for index in range(1, 3):
                sample = f"sample_{index:04d}"
                directory = os.path.join(root, "samples", sample)
                metadata = write_sample(directory, sample)
                entries.append({
                    **{key: metadata[key] for key in metadata if key != "source"},
                    "source": metadata["source"],
                })
            buckets = split_samples(entries, seed=0)
            for name in list(buckets):
                buckets[name] = [
                    {"sample_id": entry["sample_id"], "object_id": entry["object_id"],
                     "dataset": entry["dataset"], "category": entry["category"],
                     "identity_level": entry["identity_level"],
                     "reference_image": entry["reference_image"],
                     "reference_mask": entry["reference_mask"],
                     "target_image": entry["target_image"],
                     "target_mask": entry["target_mask"],
                     "viewpoint_condition": entry["viewpoint_condition"],
                     "illumination_condition": entry["illumination_condition"],
                     "scale_condition": entry["scale_condition"],
                     "occlusion_condition": entry["occlusion_condition"]}
                    for entry in buckets[name]]
            written = write_splits(buckets, root, seed=0, quiet=True)
            self.assertEqual(sorted(written), ["test", "train", "validation"])

            from evaluation.cases import cases_from_manifest

            loaded = 0
            for name, path in written.items():
                cases = cases_from_manifest(path)
                loaded += len(cases)
                for case in cases:
                    self.assertTrue(case.scorable)
                    self.assertEqual(case.reference_mask.shape,
                                     (case.reference_image.size[1], case.reference_image.size[0]))
            self.assertEqual(loaded, 2)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestIntegrityDetection(unittest.TestCase):
    """Damage the dataset in every way the checker claims to catch."""

    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="integrity-")
        entries = []
        for index in (1, 2):
            metadata = write_sample(os.path.join(self.root, "samples", f"sample_{index:04d}"),
                                    f"sample_{index:04d}", object_id=f"obj-{index}",
                                    image_ids=(index, index + 10))
            entries.append({key: metadata[key] for key in (
                "sample_id", "object_id", "dataset", "category", "identity_level",
                "reference_image", "reference_mask", "target_image", "target_mask",
                "viewpoint_condition", "illumination_condition", "scale_condition",
                "occlusion_condition")})
        buckets: Dict[str, List[Dict[str, Any]]] = {"train": entries, "validation": [],
                                                    "test": []}
        write_splits(buckets, self.root, seed=0, quiet=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def codes(self, strict: bool = False) -> set:
        return {finding.code for finding in check_dataset(self.root, strict=strict).findings}

    def test_clean_dataset_passes(self) -> None:
        report = check_dataset(self.root)
        self.assertTrue(report.ok(), report.summary())
        self.assertEqual(report.samples_checked, 2)
        self.assertEqual(report.findings, [])

    def test_missing_sample_directory_is_critical(self) -> None:
        shutil.rmtree(self.root)
        report = check_dataset(self.root)
        self.assertFalse(report.ok())
        self.assertIn("missing_samples_dir", self.codes())

    def test_missing_file_is_critical(self) -> None:
        os.remove(os.path.join(self.root, "samples", "sample_0001", "target.jpg"))
        report = check_dataset(self.root)
        self.assertFalse(report.ok())
        self.assertIn("missing_file", {finding.code for finding in report.critical})

    def test_empty_mask_is_critical(self) -> None:
        make_mask(os.path.join(self.root, "samples", "sample_0002", "reference_mask.png"),
                  box=(0, 0, 0, 0))
        report = check_dataset(self.root)
        self.assertFalse(report.ok())
        self.assertIn("empty_mask", {finding.code for finding in report.critical})

    def test_size_mismatch_is_critical(self) -> None:
        make_mask(os.path.join(self.root, "samples", "sample_0001", "target_mask.png"),
                  size=(30, 30))
        report = check_dataset(self.root)
        self.assertFalse(report.ok())
        self.assertIn("size_mismatch", {finding.code for finding in report.critical})

    def test_corrupt_image_is_critical(self) -> None:
        with open(os.path.join(self.root, "samples", "sample_0002", "reference.jpg"), "wb") as h:
            h.write(b"\xff\xd8\xff\xe0not a real jpeg")
        report = check_dataset(self.root)
        self.assertFalse(report.ok())
        self.assertIn("corrupt_image", {finding.code for finding in report.critical})

    def test_corrupt_metadata_is_critical(self) -> None:
        with open(os.path.join(self.root, "samples", "sample_0001", "metadata.json"), "w") as h:
            h.write("{not json")
        report = check_dataset(self.root)
        self.assertFalse(report.ok())
        self.assertIn("corrupt_metadata", {finding.code for finding in report.critical})

    def test_duplicate_sample_id_is_critical(self) -> None:
        shutil.copytree(os.path.join(self.root, "samples", "sample_0002"),
                        os.path.join(self.root, "samples", "sample_0003"))
        report = check_dataset(self.root)
        self.assertFalse(report.ok())
        self.assertIn("duplicate_sample_id", {finding.code for finding in report.critical})

    def test_unknown_condition_value_is_flagged(self) -> None:
        path = os.path.join(self.root, "samples", "sample_0001", "metadata.json")
        with open(path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["occlusion_condition"] = "mild"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle)
        codes = self.codes()
        self.assertIn("condition_value", codes)
        self.assertTrue(check_dataset(self.root).ok())
        self.assertFalse(check_dataset(self.root, strict=True).ok())

    def test_split_path_that_does_not_resolve_is_critical(self) -> None:
        buckets = {"train": [], "validation": [], "test": []}
        buckets["test"] = [{
            "sample_id": "sample_0001", "object_id": "obj-1", "dataset": "unit-test",
            "category": "person", "identity_level": "category",
            "reference_image": "samples/sample_0001/reference.jpg",
            "reference_mask": "samples/sample_0001/reference_mask.png",
            "target_image": "samples/sample_0009/reference.jpg",
            "target_mask": "samples/sample_0009/reference_mask.png",
            "viewpoint_condition": "unknown", "illumination_condition": "unknown",
            "scale_condition": "similar", "occlusion_condition": "unknown"}]
        write_splits(buckets, self.root, seed=0, quiet=True)
        report = check_dataset(self.root)
        self.assertFalse(report.ok())
        self.assertIn("split_path", {finding.code for finding in report.critical})

    def test_warning_severity_is_reported(self) -> None:
        write_sample(os.path.join(self.root, "samples", "sample_0004"), "sample_0004",
                     object_id="obj-4", image_ids=(1, 11))
        report = check_dataset(self.root)
        self.assertTrue(report.ok())
        self.assertIn("duplicate_pair", {finding.code for finding in report.findings})
        self.assertIn(WARNING, report.counts())

    def test_image_reuse_is_reported_as_info(self) -> None:
        for index in (5, 6, 7):
            write_sample(os.path.join(self.root, "samples", f"sample_{index:04d}"),
                         f"sample_{index:04d}", object_id=f"obj-{index}", image_ids=(99, 100))
        report = check_dataset(self.root)
        codes = {finding.code for finding in report.findings}
        self.assertIn("image_reuse", codes)
        self.assertNotIn(CRITICAL, [finding.severity for finding in report.findings])

    def test_report_is_json_serialisable(self) -> None:
        payload = check_dataset(self.root).to_dict()
        json.dumps(payload)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["samples_checked"], 2)
        self.assertEqual(payload["splits"], {"test": 0, "train": 2, "validation": 0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
