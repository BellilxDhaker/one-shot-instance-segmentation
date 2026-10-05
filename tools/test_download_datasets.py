"""Offline test for the dataset layout logic in `tools/download_datasets.py`.

Fetching 6 GB of COCO to test a path-joining bug is a bad trade, so this builds
tiny archives with the same internal structure as the real ones, serves them over
`file://` URLs, and runs the real `fetch` / `unpack` / `is_ready` / `place` code:

* `train2014.zip` containing a `train2014/` image folder, as COCO ships it;
* `annotations_trainval2014.zip` containing `annotations/instances_*.json`;
* `VOCtrainval_03-May-2010.tar` containing `VOCdevkit/VOC2010/JPEGImages`;
* `trainval.tar.gz` containing `trainval/JPEGImages` and
  `trainval/Annotations_Part_json_merged_part_classes`, which must merge into the
  already existing `Pascal-Part/VOCdevkit/VOC2010` instead of nesting;
* a truncated archive, which must be rejected instead of silently unpacked.

    python tools/test_download_datasets.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.download_datasets import (  # noqa: E402
    Archive,
    DATASETS,
    ensure_layout,
    fetch,
    human,
    is_ready,
    place,
    print_layout_table,
    unpack,
    verify_layout,
)

TINY = 4096


def make_zip(path: str, entries: Dict[str, bytes]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name, payload in entries.items():
            handle.writestr(name, payload)
    return target


def make_tar(path: str, entries: Dict[str, bytes], mode: str = "w:gz") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = tempfile.mkdtemp()
    try:
        for name, payload in entries.items():
            file_path = Path(staging) / name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(payload)
        with tarfile.open(target, mode) as handle:
            for name in entries:
                handle.add(os.path.join(staging, name), arcname=name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def fetch_and_unpack(archive: Archive, root: str, spec: Dict[str, object]) -> None:
    """Run the real download and extraction path for one synthetic archive."""
    assert not is_ready(archive, root, spec), f"{archive.key} should not look ready before fetching"
    downloaded = fetch(archive, root)
    try:
        assert os.path.getsize(downloaded) >= 0
        unpack(archive, downloaded, root)
    finally:
        shutil.rmtree(downloaded, ignore_errors=True)
    assert is_ready(archive, root, spec), f"{archive.key} should be ready after unpacking"


def test_coco_images_and_annotations(root: str) -> None:
    source = Path(root) / "_archives"
    images = make_zip(
        source / "train2014.zip",
        {f"train2014/COCO_train2014_{index:012d}.jpg": bytes(TINY) for index in range(12)},
    )
    annotations = make_zip(
        source / "annotations_trainval2014.zip",
        {
            "annotations/instances_train2014.json": bytes(TINY),
            "annotations/instances_val2014.json": bytes(TINY),
        },
    )
    spec = DATASETS["coco20i"]
    train_archive, _, annotations_archive = DATASETS["coco20i"]["archives"]
    for template, built in ((train_archive, images), (annotations_archive, annotations)):
        sized = Archive(template.key, built.as_uri(), template.filename,
                        size_bytes=built.stat().st_size, moves=template.moves,
                        ready_path=template.ready_path)
        fetch_and_unpack(sized, root, spec)

    extracted = os.path.join(root, "COCO2014", "train2014")
    assert len(os.listdir(extracted)) == 12, os.listdir(extracted)
    for name in ("instances_train2014.json", "instances_val2014.json"):
        assert os.path.exists(os.path.join(root, "COCO2014", "annotations", name)), name
    assert not os.path.exists(os.path.join(root, "downloads", "_unpack")), "staging must be cleaned up"
    print("coco images + annotations ok")


def test_pascal_moves_and_merge(root: str) -> None:
    source = Path(root) / "_archives"
    voc = make_tar(
        source / "voc.tar",
        {f"VOCdevkit/VOC2010/JPEGImages/{name}": bytes(TINY) for name in ("2007_000033.jpg", "2007_000034.jpg")},
        mode="w",
    )
    parts = make_tar(
        source / "parts.tar.gz",
        {
            "trainval/JPEGImages/2008_000123.jpg": bytes(TINY),
            "trainval/Annotations_Part_json_merged_part_classes/2008_000123.json": bytes(TINY),
        },
    )

    spec = DATASETS["pascal"]
    for template, built in ((archive, path) for archive, path in zip(DATASETS["pascal"]["archives"], (voc, parts))):
        sized = Archive(template.key, built.as_uri(), template.filename,
                        size_bytes=built.stat().st_size, kind=template.kind, moves=template.moves,
                        ready_path=template.ready_path)
        fetch_and_unpack(sized, root, spec)

    voc_root = os.path.join(root, "Pascal-Part", "VOCdevkit", "VOC2010")
    images = sorted(os.listdir(os.path.join(voc_root, "JPEGImages")))
    assert images == ["2007_000033.jpg", "2007_000034.jpg", "2008_000123.jpg"], images
    assert os.listdir(os.path.join(voc_root, "Annotations_Part_json_merged_part_classes")) == [
        "2008_000123.json"
    ]
    assert not os.path.isdir(os.path.join(voc_root, "JPEGImages", "JPEGImages")), "must merge, not nest"
    assert not os.path.exists(os.path.join(root, "trainval")), "the moved tree must not stay behind"
    assert not os.path.exists(os.path.join(root, "VOCdevkit")), "the moved tree must not stay behind"
    print("pascal moves + merge ok")


def test_place_merges_directories() -> None:
    with tempfile.TemporaryDirectory() as staging:
        source = os.path.join(staging, "src", "nested")
        destination = os.path.join(staging, "dst")
        os.makedirs(source)
        os.makedirs(destination)
        Path(source, "a.txt").write_text("a", encoding="utf-8")
        Path(destination, "b.txt").write_text("b", encoding="utf-8")
        place(source, os.path.join(destination, "nested"))
        assert os.path.exists(os.path.join(destination, "nested", "a.txt"))
        assert os.path.exists(os.path.join(destination, "b.txt"))
        assert not os.path.exists(source)
    print("place merge ok")


def test_truncated_archive_is_rejected(root: str) -> None:
    truncated = make_zip(Path(root) / "_archives" / "truncated.zip", {"x/y.txt": bytes(1024)})
    archive = Archive("truncated", truncated.as_uri(), "truncated.zip", size_bytes=10 * 1024 * 1024)
    try:
        fetch(archive, root)
    except RuntimeError as error:
        assert "error page" in str(error), error
        print("truncated archive rejected:", str(error)[:64], "...")
    else:
        raise AssertionError("a 1 KiB archive claiming to be 10 MiB must be rejected")
    finally:
        shutil.rmtree(os.path.join(root, "downloads", "truncated.zip"), ignore_errors=True)


def test_layout_report(root: str) -> None:
    rows = verify_layout(root)
    paths: List[str] = [row["path"] for row in rows]
    assert "Pascal-Part/VOCdevkit/VOC2010/JPEGImages" in paths
    assert any(row["path"] == "COCO2014/annotations/train2014" and not row["present"] for row in rows)
    assert any(row["path"] == "COCO2014/splits/val" and row["exists"] for row in rows)
    assert any(row["path"].startswith("Pascal-Part") and row["present"] for row in rows)
    print("layout report ok:", human(1024 * 1024 * 2.5))


def main() -> int:
    root = tempfile.mkdtemp(prefix="one_shot_layout_")
    try:
        ensure_layout(root)
        test_place_merges_directories()
        test_coco_images_and_annotations(root)
        test_pascal_moves_and_merge(root)
        test_truncated_archive_is_rejected(root)
        test_layout_report(root)
        print()
        print_layout_table(root)
        print("\nDOWNLOAD LAYOUT TEST PASSED")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
