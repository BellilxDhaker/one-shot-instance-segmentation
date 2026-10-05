"""Fetch and lay out the few-shot segmentation datasets inside Colab.

Public archives (COCO 2014/2017 images and annotations, VOC 2010, Pascal-Part)
are downloaded over HTTP, size-checked, extracted and moved into the layout
below. The few-shot mask archives live on Google Drive; they are skipped unless
you supply their file ids, and the status table says which ones are still
missing.

    datasets/
    ├── COCO2014/
    │   ├── annotations/{train2014,val2014}/      <- from Google Drive
    │   ├── train2014/
    │   ├── val2014/
    │   └── splits/{trn,val}/
    ├── FSS-1000/
    │   ├── data/<class>/
    │   └── splits/{test.text,trn.txt,val.txt}
    ├── LVIS/
    │   ├── coco/{train2017,val2017}/
    │   ├── lvis_train.pkl
    │   └── lvis_val.pkl
    ├── PACO-Part/
    │   ├── coco/{train2017,val2017}/
    │   └── paco/{paco_part_train.pkl,paco_part_val.pkl}
    └── Pascal-Part/
        └── VOCdevkit/VOC2010/
            ├── Annotations_Part_json_merged_part_classes/
            ├── JPEGImages/
            └── all_obj_part_to_image.json

Usage (Colab, after the project source cell has put the modules on `sys.path`):

    from tools.download_datasets import download_all, print_layout_table
    download_all("datasets", only=["coco20i"], stage="images", val_only=True)
    print_layout_table("datasets")

    python tools/download_datasets.py --only coco20i pascal
    python tools/download_datasets.py --list
    python tools/download_datasets.py --verify
    python tools/download_datasets.py --dry-run

Google Drive ids go in `tools/drive_ids.json`:

    {"train2014": "<file id>", "val2014": "<file id>", "fss1000": "<id>",
     "lvis": "<id>", "paco": "<id>", "pascal": "<id>"}

They can also come from `--drive-ids key=value ...` or from the `GDRIVE_IDS`
environment variable holding the same JSON. `pip install gdown` is required.
Each drive archive is extracted to `datasets/downloads/_drive/<key>/` and its
top-level entries are merged into the directory named in `DRIVE_LAYOUT`.

Disk warning: COCO 2014 train is ~13 GB, val ~6 GB, COCO 2017 train ~18 GB,
val ~1 GB, Pascal-Part ~2 GB. Archives are deleted after extraction unless
`--keep-archives` is passed, but the extracted images stay. In Colab, iterate
with `--val-only` and only fetch the 2017 training split when a split needs it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GIB = 1024**3
MIB = 1024**2


@dataclass
class Archive:
    """One public archive and what to do with it once it is on disk."""

    key: str
    url: str
    filename: str
    size_bytes: Optional[int] = None
    kind: str = "zip"
    moves: Tuple[Tuple[str, str], ...] = ()
    note: str = ""
    optional: bool = False
    ready_path: str = ""


@dataclass
class DriveArchive:
    """One Google Drive archive, skipped until its file id is known."""

    key: str
    filename: str
    description: str
    dataset_root: str
    expects: Tuple[str, ...]


COCO2014: Tuple[Archive, ...] = (
    Archive("cocotr2014", "http://images.cocodataset.org/zips/train2014.zip", "train2014.zip",
            13 * GIB, moves=(("train2014", "COCO2014/train2014"),),
            note="COCO 2014 train images", ready_path="COCO2014/train2014"),
    Archive("cocoval2014", "http://images.cocodataset.org/zips/val2014.zip", "val2014.zip",
            6 * GIB, moves=(("val2014", "COCO2014/val2014"),),
            note="COCO 2014 val images", ready_path="COCO2014/val2014"),
    Archive("cocoann2014", "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
            "annotations_trainval2014.zip", 241 * MIB,
            moves=(("annotations", "COCO2014/annotations"),),
            note="official COCO 2014 instance annotations",
            ready_path="COCO2014/annotations/instances_val2014.json"),
)

COCO2017: Tuple[Archive, ...] = (
    Archive("cocotr2017", "http://images.cocodataset.org/zips/train2017.zip", "train2017.zip",
            18 * GIB, moves=(("train2017", "COCO2017/train2017"),),
            note="COCO 2017 train images, shared by LVIS and PACO",
            ready_path="COCO2017/train2017"),
    Archive("cocoval2017", "http://images.cocodataset.org/zips/val2017.zip", "val2017.zip",
            778 * MIB, moves=(("val2017", "COCO2017/val2017"),),
            note="COCO 2017 val images", ready_path="COCO2017/val2017"),
)

COCO2017_ANNOTATIONS: Tuple[Archive, ...] = (
    Archive("cocoann2017", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "annotations_trainval2017.zip", 241 * MIB, optional=True,
            moves=(("annotations", "COCO2017/annotations"),),
            note="official COCO 2017 annotations; not part of the reference layout, needed only "
                 "to rebuild the LVIS/PACO split pickles yourself",
            ready_path="COCO2017/annotations/instances_val2017.json"),
)

PASCAL: Tuple[Archive, ...] = (
    Archive("voc2010", "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
            "VOCtrainval_03-May-2010.tar", 1 * GIB, kind="tar",
            moves=(("VOCdevkit/VOC2010", "Pascal-Part/VOCdevkit/VOC2010"),),
            note="VOC2010 images and part annotations; the Oxford host is often down, retry later",
            ready_path="Pascal-Part/VOCdevkit/VOC2010/JPEGImages"),
    Archive("pascalparts", "http://roozbehm.info/pascal-parts/trainval.tar.gz", "trainval.tar.gz",
            2 * GIB, kind="tar",
            moves=(("trainval/Annotations_Part_json_merged_part_classes",
                    "Pascal-Part/VOCdevkit/VOC2010/Annotations_Part_json_merged_part_classes"),
                   ("trainval/JPEGImages", "Pascal-Part/VOCdevkit/VOC2010/JPEGImages")),
            note="Pascal-Part part annotations and images",
            ready_path="Pascal-Part/VOCdevkit/VOC2010/Annotations_Part_json_merged_part_classes"),
)

DRIVE_ARCHIVES: Tuple[DriveArchive, ...] = (
    DriveArchive("train2014", "train2014.zip", "COCO-20i train masks (PNG)",
                 "COCO2014/annotations", ("COCO2014/annotations/train2014",)),
    DriveArchive("val2014", "val2014.zip", "COCO-20i val masks (PNG)",
                 "COCO2014/annotations", ("COCO2014/annotations/val2014",)),
    DriveArchive("fss1000", "FSS-1000.zip", "FSS-1000 images, masks and splits",
                 "FSS-1000", ("FSS-1000/data", "FSS-1000/splits")),
    DriveArchive("lvis", "lvis.zip", "LVIS-92i extended masks",
                 "LVIS", ("LVIS/lvis_train.pkl", "LVIS/lvis_val.pkl")),
    DriveArchive("paco", "paco.zip", "PACO-Part extended masks",
                 "PACO-Part/paco", ("PACO-Part/paco/paco_part_train.pkl",
                                    "PACO-Part/paco/paco_part_val.pkl")),
    DriveArchive("pascal", "pascal.zip", "Pascal-Part part masks",
                 "Pascal-Part/VOCdevkit/VOC2010",
                 ("Pascal-Part/VOCdevkit/VOC2010/all_obj_part_to_image.json",)),
)

DATASETS: Dict[str, Dict[str, Any]] = {
    "coco20i": {
        "title": "COCO-20i", "archives": COCO2014, "drive": ("train2014", "val2014"),
        "image_root": "COCO2014",
        "layout": ("COCO2014/train2014", "COCO2014/val2014", "COCO2014/annotations",
                   "COCO2014/splits/trn", "COCO2014/splits/val"),
    },
    "fss1000": {
        "title": "FSS-1000", "archives": (), "drive": ("fss1000",), "image_root": "FSS-1000",
        "layout": ("FSS-1000/data", "FSS-1000/splits"),
    },
    "lvis": {
        "title": "LVIS-92i", "archives": COCO2017, "drive": ("lvis",), "image_root": "LVIS/coco",
        "layout": ("LVIS/coco/train2017", "LVIS/coco/val2017"),
    },
    "paco": {
        "title": "PACO-Part", "archives": COCO2017, "drive": ("paco",), "image_root": "PACO-Part/coco",
        "layout": ("PACO-Part/coco/train2017", "PACO-Part/coco/val2017"),
    },
    "pascal": {
        "title": "Pascal-Part", "archives": PASCAL, "drive": ("pascal",),
        "image_root": "Pascal-Part/VOCdevkit/VOC2010",
        "layout": ("Pascal-Part/VOCdevkit/VOC2010/Annotations_Part_json_merged_part_classes",
                   "Pascal-Part/VOCdevkit/VOC2010/JPEGImages"),
    },
    "coco2017ann": {
        "title": "COCO 2017 official annotations (optional)", "archives": COCO2017_ANNOTATIONS,
        "drive": (), "image_root": "COCO2017", "layout": ("COCO2017/annotations",),
    },
}

DIRECTORIES: Tuple[str, ...] = (
    "COCO2014/annotations/train2014",
    "COCO2014/annotations/val2014",
    "COCO2014/splits/trn",
    "COCO2014/splits/val",
    "FSS-1000/data",
    "FSS-1000/splits",
    "LVIS/coco",
    "PACO-Part/coco",
    "Pascal-Part/VOCdevkit/VOC2010/Annotations_Part_json_merged_part_classes",
    "Pascal-Part/VOCdevkit/VOC2010/JPEGImages",
)


def ensure_layout(root: str = "datasets") -> str:
    """Create every directory of the reference layout, including empty mask folders."""
    for relative in DIRECTORIES:
        os.makedirs(os.path.join(root, relative), exist_ok=True)
    return os.path.abspath(root)


def human(size: Optional[float]) -> str:
    if not size:
        return "unknown"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def _count_entries(path: str) -> int:
    try:
        return sum(1 for _ in os.scandir(path))
    except (FileNotFoundError, NotADirectoryError):
        return 0


def is_ready(archive: Archive, root: str, spec: Dict[str, Any]) -> bool:
    """True when the extracted result of this archive is already in place.

    A directory counts as ready only when it actually holds files, because
    `ensure_layout` pre-creates the mask and image folders.
    """
    if archive.optional and not _include_optional:
        return True
    if not archive.ready_path:
        return False
    target = os.path.join(root, archive.ready_path)
    if os.path.isdir(target):
        return _count_entries(target) > 0
    return os.path.exists(target) and os.path.getsize(target) > 0


def fetch(archive: Archive, root: str, retries: int = 3, resume: bool = True) -> str:
    """Download one archive into `<root>/downloads`, resuming a partial file."""
    downloads = os.path.join(root, "downloads")
    os.makedirs(downloads, exist_ok=True)
    destination = os.path.join(downloads, archive.filename)
    if os.path.exists(destination) and _size_ok(os.path.getsize(destination), archive.size_bytes):
        print(f"  cached  {archive.filename} ({human(os.path.getsize(destination))})")
        return destination

    partial = f"{destination}.part"
    for attempt in range(1, retries + 1):
        existing = os.path.getsize(partial) if resume and os.path.exists(partial) else 0
        headers = {"User-Agent": "one-shot-segmentation/1.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            mode_hint = f"resuming at {human(existing)}" if existing else "fresh"
            print(f"  get     {archive.url}")
            print(f"          -> {archive.filename} ({human(archive.size_bytes)}, {mode_hint})")
            with urlopen(Request(archive.url, headers=headers), timeout=60) as response:
                append = bool(existing) and getattr(response, "status", 200) == 206
                written = existing if append else 0
                total = written + int(response.headers.get("Content-Length") or 0)
                mode = "ab" if append else "wb"
                reported = written
                started = time.time()
                with open(partial, mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)
                        if time.time() - started > 10:
                            rate = (written - reported) / max(time.time() - started, 1e-6)
                            print(f"          {human(written)} / {human(total)} at {human(rate)}/s")
                            reported, started = written, time.time()
            break
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as error:
            if attempt >= retries:
                raise RuntimeError(f"{archive.url} failed after {retries} attempts: {error}")
            wait = 5 * attempt
            print(f"  retry   attempt {attempt}/{retries} failed ({error}); sleeping {wait}s")
            time.sleep(wait)

    actual = os.path.getsize(partial)
    if archive.size_bytes and not _size_ok(actual, archive.size_bytes):
        raise RuntimeError(
            f"{archive.filename} is {human(actual)}, expected about {human(archive.size_bytes)}; "
            "the mirror probably returned an error page instead of the archive"
        )
    os.replace(partial, destination)
    return destination


def _size_ok(actual: int, expected: Optional[int]) -> bool:
    return bool(expected) and actual >= expected * 0.95


def unpack(archive: Archive, path: str, root: str) -> None:
    """Extract an archive into a staging folder, apply its moves, merge into the root."""
    print(f"  unpack  {archive.filename}")
    staging = os.path.join(root, "downloads", "_unpack", archive.key)
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)
    if archive.kind == "tar":
        with tarfile.open(path) as handle:
            handle.extractall(staging)
    else:
        with zipfile.ZipFile(path) as handle:
            handle.extractall(staging)

    for source, destination in archive.moves:
        source_path = os.path.join(staging, source)
        if not os.path.exists(source_path):
            print(f"  warn    {archive.filename} does not contain {source}")
            continue
        place(source_path, os.path.join(root, destination))
    for entry in sorted(os.listdir(staging)):
        entry_path = os.path.join(staging, entry)
        if os.path.isdir(entry_path) and not os.listdir(entry_path):
            continue
        place(entry_path, os.path.join(root, entry))
    shutil.rmtree(staging, ignore_errors=True)
    parent = os.path.dirname(staging)
    if os.path.isdir(parent) and not os.listdir(parent):
        os.rmdir(parent)


def place(source_path: str, destination_path: str) -> None:
    """Move a file or directory, merging into an existing directory if needed."""
    if os.path.isdir(source_path):
        if os.path.isdir(destination_path):
            for child in sorted(os.listdir(source_path)):
                place(os.path.join(source_path, child), os.path.join(destination_path, child))
            shutil.rmtree(source_path, ignore_errors=True)
            return
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.move(source_path, destination_path)
        return
    if os.path.exists(destination_path) and os.path.getsize(destination_path) == os.path.getsize(source_path):
        os.remove(source_path)
        return
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.move(source_path, destination_path)


def share_coco2017(root: str) -> None:
    """LVIS and PACO-Part share one COCO 2017 copy; link it instead of downloading twice."""
    source = os.path.join(root, "COCO2017")
    if not _count_entries(source):
        return
    for holder in ("LVIS/coco", "PACO-Part/coco"):
        destination = os.path.join(root, holder)
        os.makedirs(destination, exist_ok=True)
        for split in ("train2017", "val2017"):
            source_split = os.path.join(source, split)
            destination_split = os.path.join(destination, split)
            if _count_entries(source_split) > 10 and _count_entries(destination_split) <= 10:
                print(f"  link    {holder}/{split} -> ../../COCO2017/{split}")
                try:
                    os.symlink(os.path.relpath(source_split, destination), destination_split,
                               target_is_directory=True)
                except (OSError, NotImplementedError):
                    print(f"  note    no symlink support; link {destination_split} manually")


def fetch_drive(archive: DriveArchive, root: str, file_ids: Dict[str, str]) -> bool:
    """Fetch and place one Google Drive archive. False when no id is known."""
    file_id = file_ids.get(archive.key)
    if not file_id:
        print(f"  skip    {archive.key:9s} {archive.description}")
        print(f"          no Google Drive id; add one to tools/drive_ids.json "
              f"({', '.join(archive.expects)})")
        return False
    try:
        import gdown
    except ImportError:
        print(f"  warn    pip install gdown to fetch {archive.key}")
        return False

    downloads = os.path.join(root, "downloads")
    os.makedirs(downloads, exist_ok=True)
    destination = os.path.join(downloads, archive.filename)
    print(f"  drive   {archive.key:9s} {archive.description}")
    gdown.download(id=file_id, output=destination, quiet=False)

    staging = os.path.join(downloads, "_drive", archive.key)
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)
    if archive.filename.endswith(".zip"):
        with zipfile.ZipFile(destination) as handle:
            handle.extractall(staging)
    else:
        with tarfile.open(destination) as handle:
            handle.extractall(staging)

    entries = sorted(os.listdir(staging))
    if len(entries) == 1 and os.path.isdir(os.path.join(staging, entries[0])):
        staging = os.path.join(staging, entries[0])
        entries = sorted(os.listdir(staging))
    for entry in entries:
        place(os.path.join(staging, entry), os.path.join(root, archive.dataset_root, entry))
    shutil.rmtree(os.path.join(downloads, "_drive"), ignore_errors=True)
    print(f"          merged {len(entries)} entries into {archive.dataset_root}")
    return True


def download_all(
    root: str = "datasets",
    only: Optional[Sequence[str]] = None,
    stage: str = "all",
    keep_archives: bool = False,
    drive_ids: Optional[Dict[str, str]] = None,
    val_only: bool = False,
    dry_run: bool = False,
    include_optional: bool = False,
) -> Dict[str, Any]:
    """Download the selected datasets and build the reference layout.

    Args:
        root: destination for the `datasets/` tree.
        only: subset of `DATASETS` keys; `None` means every dataset.
        stage: `"images"`, `"masks"` or `"all"`.
        keep_archives: keep the downloaded archives instead of deleting them.
        drive_ids: Google Drive file ids keyed by `DRIVE_ARCHIVES` key.
        val_only: skip the 2014/2017 *train* image archives.
        dry_run: print the plan without downloading anything.
        include_optional: also fetch the optional official COCO 2017 annotations.

    Returns:
        A report with what was handled and which Google Drive items are missing.
    """
    global _include_optional
    _include_optional = include_optional
    root = os.path.abspath(root)
    unknown = [name for name in (only or []) if name not in DATASETS]
    if unknown:
        raise ValueError(f"unknown dataset(s) {unknown}; choose from {sorted(DATASETS)}")
    selected = list(DATASETS) if not only else list(only)
    ids = dict(drive_ids or {})

    report: Dict[str, Any] = {"root": root, "stage": stage, "datasets": {}, "missing_drive": [],
                              "skipped": []}
    print(f"layout   {ensure_layout(root)}")

    for name in selected:
        spec = DATASETS[name]
        print(f"\n== {spec['title']} ==")
        for archive in spec["archives"]:
            if val_only and archive.key.startswith("cocotr"):
                print(f"  skip    {archive.filename:32s} --val-only")
                report["skipped"].append(archive.key)
                continue
            if archive.optional and not include_optional:
                print(f"  skip    {archive.filename:32s} optional; {archive.note}")
                report["skipped"].append(archive.key)
                continue
            if stage == "masks":
                continue
            if dry_run:
                print(f"  plan    {archive.filename:32s} {human(archive.size_bytes):>10s}  {archive.url}")
                continue
            if is_ready(archive, root, spec):
                print(f"  have    {archive.note}")
                continue
            path = fetch(archive, root)
            unpack(archive, path, root)
            if not keep_archives:
                os.remove(path)

        if name in ("lvis", "paco") and not dry_run:
            share_coco2017(root)

        if stage in ("masks", "all"):
            for key in spec["drive"]:
                drive = next(item for item in DRIVE_ARCHIVES if item.key == key)
                if dry_run:
                    print(f"  plan    drive:{drive.key:9s} {drive.description} -> {drive.dataset_root}")
                    continue
                try:
                    fetched = fetch_drive(drive, root, ids)
                except RuntimeError as error:
                    print(f"  warn    {error}")
                    fetched = False
                if not fetched:
                    report["missing_drive"].append(
                        {"dataset": name, "key": drive.key, "what": drive.description,
                         "expects": list(drive.expects)}
                    )
        report["datasets"][name] = spec["title"]

    if not dry_run:
        for temporary in ("_unpack", "_drive"):
            shutil.rmtree(os.path.join(root, "downloads", temporary), ignore_errors=True)

    if report["missing_drive"]:
        print("\nstill missing (Google Drive, ids not configured):")
        for entry in report["missing_drive"]:
            print(f"  - {entry['key']:9s} {entry['what']} -> {', '.join(entry['expects'])}")
    return report


def verify_layout(root: str = "datasets") -> List[Dict[str, Any]]:
    """Report the state of every expected path of the reference layout."""
    rows: List[Dict[str, Any]] = []
    for relative in DIRECTORIES:
        path = os.path.join(root, relative)
        present = os.path.isdir(path)
        entries = _count_entries(path)
        rows.append({
            "path": relative,
            "present": bool(present and entries),
            "exists": present,
            "entries": entries,
            "note": "" if entries else ("empty" if present else "missing"),
        })
    return rows


def print_layout_table(root: str = "datasets") -> None:
    """Print the layout status as a table."""
    rows = verify_layout(root)
    width = max(len(row["path"]) for row in rows)
    print(f"{'path'.ljust(width)}  status   entries  note")
    print(f"{'-' * width}  ------  -------  {'-' * 34}")
    for row in rows:
        status = "ok" if row["present"] else ("EMPTY" if row["exists"] else "missing")
        print(f"{row['path'].ljust(width)}  {status:>6}  {row['entries']:>7}  {row['note']}")


def load_drive_ids(path: Optional[str] = None, overrides: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Read Drive ids from a json file, `--drive-ids key=value`, or `GDRIVE_IDS`."""
    file_ids: Dict[str, str] = {}
    candidate = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "drive_ids.json")
    if os.path.exists(candidate):
        with open(candidate, "r", encoding="utf-8") as handle:
            file_ids.update(json.load(handle))
    if os.environ.get("GDRIVE_IDS"):
        file_ids.update(json.loads(os.environ["GDRIVE_IDS"]))
    for override in overrides or ():
        key, _, value = override.partition("=")
        if value:
            file_ids[key] = value
    return file_ids


def list_datasets() -> None:
    """Print every dataset with its public archives and Drive items."""
    for name, spec in DATASETS.items():
        print(f"{name:12s} {spec['title']}")
        for archive in spec["archives"]:
            marker = "optional" if archive.optional else "public "
            print(f"             [{marker}] {archive.filename:34s} {human(archive.size_bytes):>10s}  {archive.url}")
        for key in spec["drive"]:
            drive = next(item for item in DRIVE_ARCHIVES if item.key == key)
            print(f"             [drive  ] {drive.filename:34s} {'unknown':>10s}  {drive.description}")


_include_optional = False


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Download the few-shot segmentation datasets.")
    parser.add_argument("--root", default="datasets", help="destination directory (default: datasets)")
    parser.add_argument("--only", nargs="*", default=None, choices=sorted(DATASETS),
                        help="subset of datasets to fetch")
    parser.add_argument("--stage", default="all", choices=("images", "masks", "all"))
    parser.add_argument("--val-only", action="store_true", help="skip the 2014/2017 train image archives")
    parser.add_argument("--keep-archives", action="store_true", help="keep downloaded archives")
    parser.add_argument("--drive-ids", nargs="*", default=None, help="key=file_id pairs for Google Drive")
    parser.add_argument("--include-optional", action="store_true",
                        help="also fetch the optional official COCO 2017 annotations")
    parser.add_argument("--verify", action="store_true", help="only print the layout status")
    parser.add_argument("--list", action="store_true", help="list datasets and archives")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.list:
        list_datasets()
        return 0
    if arguments.verify:
        print_layout_table(arguments.root)
        return 0

    download_all(
        root=arguments.root,
        only=arguments.only,
        stage=arguments.stage,
        keep_archives=arguments.keep_archives,
        drive_ids=load_drive_ids(overrides=arguments.drive_ids),
        val_only=arguments.val_only,
        dry_run=arguments.dry_run,
        include_optional=arguments.include_optional,
    )
    print()
    print_layout_table(arguments.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
