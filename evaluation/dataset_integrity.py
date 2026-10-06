"""Integrity checks for a benchmark laid out on disk.

A benchmark result is worthless if a mask was silently empty, an image was
truncated by a failed download, or two samples share an identifier, so the
checks live here as a plain function that a CLI, a notebook, and a test can all
call. Findings carry a severity so that a report can escalate warnings without
code changes:

`critical`
    The sample cannot be scored as stored: a missing file, an unreadable image, a
    mask that is empty or does not match its image. A benchmark run must stop.
`warning`
    The sample is scorable but suspicious: metadata disagreeing with the pixels, a
    photograph reused across samples, or a pair repeated across splits. Raise
    these to critical with `--strict`.
`info`
    Facts worth recording, such as how the dataset is spread over categories.

The checker never modifies the dataset and never fetches anything.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

CRITICAL = "critical"
WARNING = "warning"
INFO = "info"

REQUIRED_FILES = ("reference.jpg", "reference_mask.png", "target.jpg", "target_mask.png",
                  "metadata.json")
ROLES = ("reference", "target")
CONDITION_VALUES = {
    "viewpoint_condition": {"unknown"},
    "illumination_condition": {"unknown"},
    "occlusion_condition": {"unknown"},
    "scale_condition": {"similar", "smaller_in_target", "larger_in_target"},
}
METADATA_KEYS = ("sample_id", "object_id", "dataset", "category") + tuple(CONDITION_VALUES)
BBOX_TOLERANCE_PX = 3.0


@dataclass
class Finding:
    """One problem or observation about one sample."""

    severity: str
    code: str
    sample_id: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"severity": self.severity, "code": self.code,
                "sample_id": self.sample_id, "message": self.message}


@dataclass
class IntegrityReport:
    """Everything the checker found, plus a few counts."""

    root: str
    findings: List[Finding] = field(default_factory=list)
    samples_checked: int = 0
    categories: Counter = field(default_factory=Counter)
    splits: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, severity: str, code: str, sample_id: str, message: str) -> None:
        self.findings.append(Finding(severity, code, sample_id, message))

    def counts(self) -> Dict[str, int]:
        return dict(Counter(finding.severity for finding in self.findings))

    @property
    def critical(self) -> List[Finding]:
        return [finding for finding in self.findings if finding.severity == CRITICAL]

    def ok(self) -> bool:
        return not self.critical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "samples_checked": self.samples_checked,
            "counts": self.counts(),
            "ok": self.ok(),
            "categories": dict(sorted(self.categories.items())),
            "splits": {name: len(items) for name, items in sorted(self.splits.items())},
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def summary(self, limit: int = 12) -> str:
        """Format the report for a terminal."""
        counts = self.counts()
        lines = [
            f"dataset root: {self.root}",
            f"samples: {self.samples_checked} | categories: {len(self.categories)}",
            "splits: " + ", ".join(f"{name} {len(items)}" for name, items in sorted(self.splits.items())),
            "findings: " + ", ".join(f"{severity} {count}" for severity, count in sorted(counts.items()))
            or "findings: none",
        ]
        for severity in (CRITICAL, WARNING):
            shown = [finding for finding in self.findings if finding.severity == severity][:limit]
            for finding in shown:
                lines.append(f"  [{severity}] {finding.code} {finding.sample_id}: {finding.message}")
            hidden = len([f for f in self.findings if f.severity == severity]) - len(shown)
            if hidden > 0:
                lines.append(f"  [{severity}] ... {hidden} more")
        lines.append("RESULT: " + ("PASS" if self.ok() else "FAIL"))
        return "\n".join(lines)


def _open_image(path: str) -> Any:
    from PIL import Image

    image = Image.open(path)
    image.load()
    return image


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    return (float(columns.min()), float(rows.min()),
            float(columns.max() - columns.min() + 1), float(rows.max() - rows.min() + 1))


def check_sample(directory: str, report: IntegrityReport) -> Optional[Dict[str, Any]]:
    """Validate one sample folder, recording findings against `report`."""
    name = os.path.basename(os.path.normpath(directory))
    for filename in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(directory, filename)):
            report.add(CRITICAL, "missing_file", name, f"{filename} is absent")
    if report.critical and any(finding.sample_id == name for finding in report.critical):
        return None

    try:
        with open(os.path.join(directory, "metadata.json"), "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        report.add(CRITICAL, "corrupt_metadata", name, f"metadata.json unreadable: {error}")
        return None

    sample_id = str(metadata.get("sample_id", name))
    for key in METADATA_KEYS:
        if key not in metadata:
            report.add(CRITICAL, "metadata_field", sample_id, f"metadata lacks {key}")
    for key, allowed in CONDITION_VALUES.items():
        value = metadata.get(key)
        if key in metadata and value not in allowed:
            report.add(WARNING, "condition_value", sample_id,
                       f"{key}={value!r} is not one of {sorted(allowed)}")

    measured = metadata.get("measured", {}) if isinstance(metadata.get("measured"), dict) else {}
    for role in ROLES:
        image_path = os.path.join(directory, f"{role}.jpg")
        mask_path = os.path.join(directory, f"{role}_mask.png")
        try:
            image = _open_image(image_path)
        except Exception as error:  # noqa: BLE001 - any decode failure means the file is unusable
            report.add(CRITICAL, "corrupt_image", sample_id, f"{role}.jpg unreadable: {error}")
            continue
        try:
            mask_image = _open_image(mask_path)
        except Exception as error:  # noqa: BLE001
            report.add(CRITICAL, "corrupt_mask", sample_id, f"{role}_mask.png unreadable: {error}")
            continue

        mask = np.asarray(mask_image)
        if mask.ndim == 3:
            mask = mask[..., 0]
        binary = mask > 127
        if not binary.any():
            report.add(CRITICAL, "empty_mask", sample_id, f"{role} mask has no foreground")
            continue
        if mask_image.size != image.size:
            report.add(CRITICAL, "size_mismatch", sample_id,
                       f"{role} mask is {mask_image.size} but the image is {image.size}")
            continue
        expected_fraction = measured.get(f"{role}_area_fraction")
        if isinstance(expected_fraction, (int, float)):
            if abs(float(binary.mean()) - float(expected_fraction)) > 0.01:
                report.add(WARNING, "area_mismatch", sample_id,
                           f"{role} mask covers {binary.mean():.4f} of the image but metadata "
                           f"says {float(expected_fraction):.4f}")
        expected_bbox = measured.get(f"{role}_bbox")
        if isinstance(expected_bbox, (list, tuple)) and len(expected_bbox) == 4:
            box = _mask_bbox(binary)
            if box is None:
                report.add(CRITICAL, "empty_mask", sample_id, f"{role} mask has no foreground")
            else:
                deviation = max(abs(box[index] - float(expected_bbox[index])) for index in range(4))
                if deviation > BBOX_TOLERANCE_PX:
                    report.add(WARNING, "bbox_mismatch", sample_id,
                               f"{role} mask box deviates {deviation:.1f}px from the source "
                               f"annotation box")
    return metadata


def check_dataset(root: str = "data", strict: bool = False,
                  samples_dir: Optional[str] = None) -> IntegrityReport:
    """Check every sample folder under `root` and cross-check the split manifests.

    Args:
        root: dataset root holding `samples/`, `splits/`, and optionally `raw/`.
        strict: promote warnings to critical, for a pre-publication check.
        samples_dir: override the samples directory, for a root with a different
            layout.

    Returns:
        An `IntegrityReport`. `report.ok()` is false when anything critical was
        found, which is the signal a benchmark runner should stop on.
    """
    report = IntegrityReport(root=root)
    directory = samples_dir or os.path.join(root, "samples")
    if not os.path.isdir(directory):
        report.add(CRITICAL, "missing_samples_dir", "-", f"{directory} does not exist")
        return report

    names = sorted(entry for entry in os.listdir(directory)
                   if os.path.isdir(os.path.join(directory, entry)))
    seen_sample_ids: Dict[str, str] = {}
    seen_pairs: Dict[Tuple[str, str], str] = {}
    image_usage: Dict[str, List[str]] = {}

    for name in names:
        metadata = check_sample(os.path.join(directory, name), report)
        if metadata is None:
            continue
        report.samples_checked += 1
        sample_id = str(metadata.get("sample_id", name))
        category = str(metadata.get("category", "unknown"))
        report.categories[category] += 1

        if sample_id in seen_sample_ids:
            report.add(CRITICAL, "duplicate_sample_id", sample_id,
                       f"also used by {seen_sample_ids[sample_id]}")
        seen_sample_ids[sample_id] = name
        if sample_id != name:
            report.add(WARNING, "id_folder_mismatch", sample_id,
                       f"stored in folder {name}")

        source = metadata.get("source", {}) if isinstance(metadata.get("source"), dict) else {}
        image_ids = source.get("image_id") or []
        if isinstance(image_ids, list) and len(image_ids) == 2:
            key = (str(image_ids[0]), str(image_ids[1]))
            pair_key = tuple(sorted(key))
            if pair_key in seen_pairs:
                report.add(WARNING, "duplicate_pair", sample_id,
                           f"same two photographs as {seen_pairs[pair_key]}")
            seen_pairs[pair_key] = sample_id
            for value in key:
                image_usage.setdefault(value, []).append(sample_id)

    for photograph, owners in sorted(image_usage.items()):
        unique = sorted(set(owners))
        if len(unique) > 2:
            report.add(INFO, "image_reuse", unique[0],
                       f"source photograph {photograph} appears in {len(unique)} samples")

    splits_dir = os.path.join(root, "splits")
    if os.path.isdir(splits_dir):
        split_pairs: Dict[Tuple[str, str], str] = {}
        for filename in sorted(os.listdir(splits_dir)):
            if not filename.endswith(".json"):
                continue
            split = filename[:-5]
            path = os.path.join(splits_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except (OSError, json.JSONDecodeError) as error:
                report.add(CRITICAL, "corrupt_split", split, f"unreadable: {error}")
                continue
            entries = manifest.get("cases", [])
            report.splits[split] = [str(entry.get("name", "?")) for entry in entries]
            on_disk: Set[str] = set()
            for entry in entries:
                case_name = str(entry.get("name", "?"))
                for role in ROLES:
                    relative = entry.get(f"{role}_image")
                    if not isinstance(relative, str):
                        report.add(CRITICAL, "split_field", case_name,
                                   f"manifest entry lacks {role}_image")
                        continue
                    resolved = os.path.normpath(os.path.join(splits_dir, relative))
                    if not os.path.isfile(resolved):
                        report.add(CRITICAL, "split_path", case_name,
                                   f"{role}_image does not resolve: {relative}")
                    else:
                        on_disk.add(os.path.basename(os.path.dirname(resolved)))
            missing = sorted(set(report.splits[split]) - on_disk)
            if missing:
                report.add(WARNING, "split_coverage", split,
                           f"{len(missing)} listed cases were not found on disk "
                           f"(first: {missing[0]})")
            for entry in entries:
                image = entry.get("reference_image")
                target = entry.get("target_image")
                if isinstance(image, str) and isinstance(target, str):
                    key = tuple(sorted((os.path.normpath(image), os.path.normpath(target))))
                    if key in split_pairs and split_pairs[key] != split:
                        report.add(WARNING, "split_leakage", str(entry.get("name", "?")),
                                   f"pair also appears in split {split_pairs[key]}")
                    split_pairs.setdefault(key, split)
    else:
        report.add(WARNING, "missing_splits_dir", "-", f"{splits_dir} does not exist")

    if strict:
        for finding in report.findings:
            if finding.severity == WARNING:
                finding.severity = CRITICAL
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verify benchmark data integrity",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data", help="dataset root (default: data)")
    parser.add_argument("--samples-dir", default=None, help="override the samples directory")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--json", default=None, help="write the full report to this path")
    parser.add_argument("--max-shown", type=int, default=12,
                        help="findings printed per severity (default: 12)")
    arguments = parser.parse_args(argv)

    report = check_dataset(root=arguments.root, strict=arguments.strict,
                           samples_dir=arguments.samples_dir)
    print(report.summary(limit=arguments.max_shown))
    if arguments.json:
        os.makedirs(os.path.dirname(os.path.abspath(arguments.json)), exist_ok=True)
        with open(arguments.json, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
            handle.write("\n")
        print(f"report -> {arguments.json}")
    return 0 if report.ok() else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
