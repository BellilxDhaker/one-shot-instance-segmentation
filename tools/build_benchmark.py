"""Build the Phase 2 robustness benchmark: reference/target pairs on disk.

The benchmark asks one question with real data instead of synthetic edits: given
one annotated image of an object, how well does the pipeline find that same kind
of object again in a photograph it has never seen? Every pair therefore carries
two different photographs and two different masks, and the target mask is the
only thing used for scoring.

Candidate sources were compared in `docs/dataset_comparison.md`. The conclusion
is that COCO val2017 is the only public instance-mask corpus that is practical
here: its annotation file is small enough to fetch without downloading the whole
241 MB archive, and its images can be pulled one file at a time. The cost is
that COCO does not track object identity across photographs, so a pair relates
two *different instances of the same category*. Every sample therefore records
`identity_level` so no result can be mistaken for same-object matching.
`docs/pairing_protocol.md` explains the choice and the limits.

    python tools/build_benchmark.py --pairs 100 --seed 0

Outputs, all relative to `--root`:

    raw/coco2017-val/*.jpg              cached source photographs
    raw/annotations/instances_val2017.json
    processed/source_index.json         which annotations were usable
    samples/sample_0001/reference.jpg   the annotated object, first view
    samples/sample_0001/reference_mask.png
    samples/sample_0001/target.jpg      another photograph, same category
    samples/sample_0001/target_mask.png
    samples/sample_0001/metadata.json
    splits/{train,validation,test}.json manifests for `evaluation.cases`

The split manifests use paths relative to their own directory, so
`cases_from_manifest("data/splits/test.json")` works from any working directory.

Selecting annotations is deterministic: the same seed and the same annotation
file always yield the same pairs, which is what makes the saved results
reproducible.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

COCO_2017_VAL_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_2017_VAL_ANNOTATION_MEMBER = "annotations/instances_val2017.json"
COCO_2017_VAL_IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"
DATASET_NAME = "coco2017-val"

CONDITION_FIELDS = (
    "viewpoint_condition",
    "illumination_condition",
    "scale_condition",
    "occlusion_condition",
)
METADATA_FIELDS = (
    "sample_id",
    "object_id",
    "dataset",
    "category",
) + CONDITION_FIELDS

SCALE_RATIO_SMALL = 0.5
SCALE_RATIO_LARGE = 2.0


class HttpRangeFile(io.IOBase):
    """Read-only seekable view of a remote file, fetched one block at a time.

    A ZIP archive is readable over HTTP without downloading all of it: the
    central directory sits at the end and is small, so a seekable reader plus
    `Range` requests are enough to pull out a single member. Requests here are
    block-aligned and cached, which keeps the request count near the number of
    blocks touched instead of the number of reads the ZIP parser performs.
    """

    def __init__(self, url: str, block: int = 1 << 20, timeout: int = 60,
                 retries: int = 3) -> None:
        super().__init__()
        self.url = url
        self.block = int(block)
        self.timeout = int(timeout)
        self.retries = int(retries)
        self.size = self._remote_size()
        self._position = 0
        self._buffer = b""
        self._buffer_start = 0

    def _request(self, start: int, end: int) -> Tuple[bytes, Dict[str, str]]:
        request = urllib.request.Request(self.url)
        request.add_header("Range", f"bytes={start}-{end}")
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                    headers = {key.lower(): value for key, value in response.headers.items()}
                    return payload, headers
            except (urllib.error.URLError, OSError, TimeoutError) as error:
                last = error
                time.sleep(1.0 + attempt)
        raise IOError(f"range request {start}-{end} failed for {self.url}: {last}")

    def _remote_size(self) -> int:
        head = urllib.request.Request(self.url, method="HEAD")
        try:
            with urllib.request.urlopen(head, timeout=self.timeout) as response:
                length = response.headers.get("Content-Length")
                if length is not None:
                    return int(length)
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        payload, headers = self._request(0, 0)
        del payload
        content_range = headers.get("content-range", "")
        if "/" not in content_range:
            raise IOError(f"server did not report a size for {self.url}")
        return int(content_range.rsplit("/", 1)[1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"unsupported whence {whence}")
        self._position = max(0, min(int(position), self.size))
        return self._position

    def read(self, size: int = -1) -> bytes:
        if self._position >= self.size:
            return b""
        remaining = self.size - self._position
        wanted = remaining if size is None or size < 0 else min(int(size), remaining)
        chunk = bytearray()
        while len(chunk) < wanted:
            start = self._position
            in_buffer = (self._buffer_start <= start < self._buffer_start + len(self._buffer))
            if in_buffer:
                offset = start - self._buffer_start
                take = min(wanted - len(chunk), len(self._buffer) - offset)
                chunk.extend(self._buffer[offset:offset + take])
                self._position = start + take
                continue
            block_start = start - (start % self.block)
            block_end = min(block_start + self.block, self.size) - 1
            payload, _ = self._request(block_start, block_end)
            self._buffer = payload
            self._buffer_start = block_start
        return bytes(chunk)


def fetch_annotation_file(cache_path: str, url: str = COCO_2017_VAL_ANNOTATIONS_URL,
                         member: str = COCO_2017_VAL_ANNOTATION_MEMBER,
                         quiet: bool = False) -> str:
    """Return a local path to the COCO instance annotations, fetching if needed.

    Only the requested member is transferred. The full archive is roughly 241 MB
    while `instances_val2017.json` is about a fifth of that compressed, so this
    saves the bulk of a slow download and leaves the rest of the archive alone.
    """
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1024:
        return cache_path
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    if not quiet:
        print(f"fetching {member} from {url}")
    remote = HttpRangeFile(url)
    if not quiet:
        print(f"archive {human(remote.size)}; reading central directory")
    with zipfile.ZipFile(remote) as archive:
        info = archive.getinfo(member)
        if not quiet:
            print(f"{member}: {human(info.compress_size)} compressed, "
                  f"{human(info.file_size)} uncompressed")
        temporary = cache_path + ".part"
        started = time.time()
        with archive.open(info) as source, open(temporary, "wb") as handle:
            copied = 0
            while True:
                block = source.read(1 << 18)
                if not block:
                    break
                handle.write(block)
                copied += len(block)
                if not quiet and copied % (4 << 20) < (1 << 18):
                    rate = (copied / (1 << 20)) / max(time.time() - started, 1e-6)
                    print(f"  {human(copied)} at {rate:.2f} MB/s")
    os.replace(temporary, cache_path)
    if not quiet:
        print(f"saved {human(os.path.getsize(cache_path))} to {cache_path}")
    return cache_path


def human(size: Optional[float]) -> str:
    """Format a byte count for progress output."""
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def download_image(url: str, destination: str, quiet: bool = True) -> str:
    """Fetch one image, skipping the network entirely when it is already cached."""
    if os.path.exists(destination) and os.path.getsize(destination) > 512:
        return destination
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "one-shot-benchmark/2.0"})
    last: Optional[Exception] = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            if len(payload) < 512:
                raise IOError(f"short response ({len(payload)} bytes) for {url}")
            temporary = destination + ".part"
            with open(temporary, "wb") as handle:
                handle.write(payload)
            os.replace(temporary, destination)
            if not quiet:
                print(f"    fetched {os.path.basename(destination)} ({human(len(payload))})")
            return destination
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            last = error
            time.sleep(1.0 + attempt)
    raise IOError(f"could not download {url}: {last}")


def decode_compressed_counts(text: str) -> List[int]:
    """Decode COCO's string-encoded RLE counts into a list of run lengths.

    COCO packs five six-bit symbols into one 32-bit integer, least significant
    symbol first, with the sign carried in the lowest bit: a run whose length
    occupies `n << 1` is a foreground run and `n << 1 | 1` is a background run.
    Keeping this here means one dataset format detail does not pull in an extra
    dependency for the notebook.
    """
    if isinstance(text, list):
        return [int(value) for value in text]
    counts: List[int] = []
    for index in range(0, len(text) - 1, 5):
        chunk = text[index:index + 5]
        if len(chunk) < 5:
            break
        value = 0
        for shift in range(5):
            digit = (ord(chunk[shift]) - 48) & 0x3F
            value |= digit << (6 * shift)
        counts.append(-((value - 1) >> 1) if value & 1 else value >> 1)
    return counts


def encode_compressed_counts(counts: Sequence[int]) -> str:
    """Encode run lengths the way COCO stores them, the inverse of the decoder.

    Only needed to prove the decoder against known values, and to write RLE
    annotations back out if a later source uses them as the stored form.
    """
    symbols: List[str] = []
    for count in counts:
        value = int(count)
        encoded = ((-value) << 1 | 1) if value < 0 else (value << 1)
        for shift in range(5):
            symbols.append(chr(48 + ((encoded >> (6 * shift)) & 0x3F)))
    return "".join(symbols)


def decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    """Turn a COCO RLE into a boolean mask at the RLE's own resolution.

    COCO scans columns top to bottom, so the flat mask is filled in `fortran`
    order before being reshaped.
    """
    height, width = int(rle["size"][0]), int(rle["size"][1])
    counts = decode_compressed_counts(rle["counts"])
    flat = np.zeros(height * width, dtype=bool)
    position = 0
    value = False
    for run in counts:
        run = int(run)
        if value and run > 0:
            flat[position:position + run] = True
        position += max(run, 0)
        value = not value
        if position >= flat.size:
            break
    return flat.reshape((height, width), order="F")


def annotation_to_mask(segmentation: Dict[str, Any], height: int, width: int) -> np.ndarray:
    """Rasterise one COCO `segmentation` field at the image's resolution.

    Polygons are filled with Pillow; RLE is decoded directly. Crowd regions are
    never paired, so the union of a mask's parts is the object itself.
    """
    from PIL import Image, ImageDraw

    if isinstance(segmentation, list):
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for polygon in segmentation:
            points = [(float(polygon[i]), float(polygon[i + 1]))
                      for i in range(0, len(polygon) - 1, 2)]
            if len(points) >= 3:
                draw.polygon(points, fill=255)
        mask = np.asarray(canvas) > 127
    else:
        rle_mask = decode_rle(segmentation)
        if rle_mask.shape != (height, width):
            mask = np.asarray(Image.fromarray(rle_mask.astype(np.uint8) * 255)
                              .resize((width, height), Image.NEAREST)) > 127
        else:
            mask = rle_mask
    return mask


@dataclass
class SourceObject:
    """One usable annotated instance from the source corpus."""

    annotation_id: int
    image_id: int
    file_name: str
    category_id: int
    category: str
    area_fraction: float
    bbox: Tuple[float, float, float, float]
    width: int
    height: int
    segmentation: Dict[str, Any] = field(repr=False)


@dataclass
class Pair:
    """A reference/target pair drawn from two different photographs."""

    reference: SourceObject
    target: SourceObject
    scale_condition: str
    area_ratio: float


def load_source_objects(annotation_path: str, min_area_fraction: float, max_area_fraction: float,
                        quiet: bool = False) -> Tuple[List[SourceObject], Dict[int, Dict[str, Any]]]:
    """Read the annotation file and keep the instances that can form a pair.

    An instance qualifies when it is not a crowd region, occupies a workable
    share of its photograph, and is not one of a handful of images that are
    known to be unreadable in COCO val2017.
    """
    with open(annotation_path, "r", encoding="utf-8") as handle:
        coco = json.load(handle)
    images = {int(entry["id"]): entry for entry in coco["images"]}
    categories = {int(entry["id"]): entry for entry in coco["categories"]}
    objects: List[SourceObject] = []
    for annotation in coco["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        image = images.get(int(annotation["image_id"]))
        if image is None or not image.get("file_name"):
            continue
        height, width = int(image["height"]), int(image["width"])
        if height <= 0 or width <= 0:
            continue
        fraction = float(annotation["area"]) / float(height * width)
        if not min_area_fraction <= fraction <= max_area_fraction:
            continue
        segmentation = annotation.get("segmentation")
        if not segmentation:
            continue
        objects.append(
            SourceObject(
                annotation_id=int(annotation["id"]),
                image_id=int(annotation["image_id"]),
                file_name=str(image["file_name"]),
                category_id=int(annotation["category_id"]),
                category=str(categories[int(annotation["category_id"])]["name"]),
                area_fraction=fraction,
                bbox=tuple(float(value) for value in annotation["bbox"]),  # type: ignore[arg-type]
                width=width,
                height=height,
                segmentation=segmentation,
            )
        )
    objects.sort(key=lambda item: (item.category_id, item.image_id, item.annotation_id))
    if not quiet:
        print(f"usable instances: {len(objects)} across "
              f"{len({item.category_id for item in objects})} categories")
    return objects, images


def classify_scale(reference: SourceObject, target: SourceObject) -> Tuple[str, float]:
    """Name the scale relation from measured mask areas, not from annotations."""
    ratio = target.area_fraction / max(reference.area_fraction, 1e-9)
    if ratio < SCALE_RATIO_SMALL:
        return "smaller_in_target", ratio
    if ratio > SCALE_RATIO_LARGE:
        return "larger_in_target", ratio
    return "similar", ratio


def build_pairs(objects: Sequence[SourceObject], count: int, seed: int,
                categories_cap: Optional[int] = None,
                quiet: bool = False) -> List[Pair]:
    """Choose pairs deterministically, spreading them over as many categories as possible.

    Category balance comes first: COCO is dominated by `person`, so a naive sample
    would be almost entirely people. Each category contributes a near-equal share
    before any category is allowed a second pair, and both photographs of a pair
    are always different images.
    """
    by_category: Dict[int, List[SourceObject]] = {}
    for item in objects:
        by_category.setdefault(item.category_id, []).append(item)

    usable: List[Tuple[int, List[SourceObject]]] = []
    for category_id, items in sorted(by_category.items()):
        image_ids = {item.image_id for item in items}
        if len(image_ids) >= 2:
            usable.append((category_id, items))
    usable.sort(key=lambda entry: (-len({item.image_id for item in entry[1]}), entry[0]))
    if categories_cap is not None:
        usable = usable[:categories_cap]
    if not usable:
        raise SystemExit("no category has instances in two different images")

    rng = random.Random(seed)
    queues: List[List[SourceObject]] = []
    for _, items in usable:
        shuffled = list(items)
        rng.shuffle(shuffled)
        queues.append(shuffled)

    pairs: List[Pair] = []
    exhausted: List[int] = []
    while len(pairs) < count and len(exhausted) < len(queues):
        for index in range(len(queues)):
            if index in exhausted:
                continue
            queue = queues[index]
            if len(queue) < 2:
                exhausted.append(index)
                continue
            reference, target = queue.pop(), queue.pop()
            while reference.image_id == target.image_id and queue:
                target = queue.pop()
            if reference.image_id == target.image_id:
                exhausted.append(index)
                continue
            condition, ratio = classify_scale(reference, target)
            pairs.append(Pair(reference=reference, target=target,
                              scale_condition=condition, area_ratio=ratio))
            if len(pairs) >= count:
                break
    if not quiet:
        spread: Dict[str, int] = {}
        for pair in pairs:
            spread[pair.reference.category] = spread.get(pair.reference.category, 0) + 1
        print(f"selected {len(pairs)} pairs over {len(spread)} categories "
              f"(requested {count}, seed {seed})")
    return pairs


def write_sample(pair: Pair, index: int, root: str, images_dir: str,
                 quiet: bool = False) -> Dict[str, Any]:
    """Download both photographs, render both masks, and write one sample folder.

    Image bytes are cached under `raw/` and hard-linked into the sample folder,
    so a benchmark never stores the same photograph twice on disk while each
    sample still stands on its own.
    """
    from PIL import Image

    sample_id = f"sample_{index:04d}"
    directory = os.path.join(root, "samples", sample_id)
    os.makedirs(directory, exist_ok=True)

    paths: Dict[str, str] = {}
    masks: Dict[str, np.ndarray] = {}
    for role, item in (("reference", pair.reference), ("target", pair.target)):
        cached = os.path.join(images_dir, item.file_name)
        download_image(COCO_2017_VAL_IMAGE_URL.format(file_name=item.file_name), cached, quiet)
        image_path = os.path.join(directory, f"{role}.jpg")
        try:
            os.link(cached, image_path)
        except (OSError, AttributeError):
            with open(cached, "rb") as source, open(image_path, "wb") as handle:
                handle.write(source.read())
        paths[f"{role}_image"] = os.path.relpath(image_path, root).replace(os.sep, "/")
        mask = annotation_to_mask(item.segmentation, item.height, item.width)
        if not mask.any():
            raise ValueError(f"empty rendered mask for {role} of {sample_id}")
        masks[role] = mask
        mask_path = os.path.join(directory, f"{role}_mask.png")
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(mask_path)
        paths[f"{role}_mask"] = os.path.relpath(mask_path, root).replace(os.sep, "/")

    with Image.open(os.path.join(directory, "reference.jpg")) as probe:
        reference_size = probe.size
    with Image.open(os.path.join(directory, "target.jpg")) as probe:
        target_size = probe.size

    metadata: Dict[str, Any] = {
        "sample_id": sample_id,
        "object_id": f"coco2017-{pair.reference.annotation_id}-{pair.target.annotation_id}",
        "identity_level": "category",
        "dataset": DATASET_NAME,
        "category": pair.reference.category,
        "category_id": pair.reference.category_id,
        "reference_image": paths["reference_image"],
        "reference_mask": paths["reference_mask"],
        "target_image": paths["target_image"],
        "target_mask": paths["target_mask"],
        "viewpoint_condition": "unknown",
        "illumination_condition": "unknown",
        "scale_condition": pair.scale_condition,
        "occlusion_condition": "unknown",
        "measured": {
            "reference_area_fraction": round(float(masks["reference"].mean()), 6),
            "target_area_fraction": round(float(masks["target"].mean()), 6),
            "area_ratio_target_over_reference": round(float(pair.area_ratio), 4),
            "reference_image_size": list(reference_size),
            "target_image_size": list(target_size),
            "reference_bbox": [round(value, 2) for value in pair.reference.bbox],
            "target_bbox": [round(value, 2) for value in pair.target.bbox],
        },
        "source": {
            "image_id": [pair.reference.image_id, pair.target.image_id],
            "annotation_id": [pair.reference.annotation_id, pair.target.annotation_id],
            "file_name": [pair.reference.file_name, pair.target.file_name],
            "annotation_file": "instances_val2017.json",
        },
    }
    with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def split_samples(entries: Sequence[Dict[str, Any]], seed: int,
                  ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2)) -> Dict[str, List[Dict[str, Any]]]:
    """Partition samples into train/validation/test with a seeded shuffle."""
    ordered = sorted(entries, key=lambda item: item["sample_id"])
    rng = random.Random(seed + 1)
    rng.shuffle(ordered)
    total = len(ordered)
    train_end = int(round(total * ratios[0]))
    validation_end = train_end + int(round(total * ratios[1]))
    buckets = {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }
    return buckets


def write_splits(buckets: Dict[str, List[Dict[str, Any]]], root: str, seed: int,
                 quiet: bool = False) -> Dict[str, str]:
    """Write one manifest per split, in the schema `evaluation.cases` already reads."""
    directory = os.path.join(root, "splits")
    os.makedirs(directory, exist_ok=True)
    written: Dict[str, str] = {}
    for name, entries in buckets.items():
        cases = []
        for metadata in entries:
            measured = metadata.get("measured", {})
            cases.append({
                "name": metadata["sample_id"],
                "reference_image": f"../{metadata['reference_image']}",
                "reference_mask": f"../{metadata['reference_mask']}",
                "target_image": f"../{metadata['target_image']}",
                "target_mask": f"../{metadata['target_mask']}",
                "object_id": metadata["object_id"],
                "dataset": metadata["dataset"],
                "category": metadata["category"],
                "identity_level": metadata["identity_level"],
                "reference_area_fraction": measured.get("reference_area_fraction"),
                "target_area_fraction": measured.get("target_area_fraction"),
                **{field: metadata[field] for field in CONDITION_FIELDS},
            })
        manifest = {
            "dataset": DATASET_NAME,
            "split": name,
            "seed": seed,
            "pairing": "class-level: two different instances of the same category in "
                       "two different photographs",
            "cases": cases,
        }
        path = os.path.join(directory, f"{name}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        written[name] = path
        if not quiet:
            print(f"  {name}: {len(cases)} cases -> {path}")
    return written


def build(root: str, pairs: int, seed: int, min_area_fraction: float, max_area_fraction: float,
          categories_cap: Optional[int], annotation_path: Optional[str], annotation_url: str,
          dry_run: bool) -> Dict[str, Any]:
    """Run the whole build: annotations, pair selection, samples, splits."""
    quiet = False
    annotation_cache = annotation_path or os.path.join(
        root, "raw", "annotations", "instances_val2017.json")
    if not os.path.exists(annotation_cache):
        fetch_annotation_file(annotation_cache, url=annotation_url, quiet=quiet)
    objects, _ = load_source_objects(annotation_cache, min_area_fraction, max_area_fraction, quiet)
    selection = build_pairs(objects, pairs, seed, categories_cap, quiet)

    report: Dict[str, Any] = {
        "dataset": DATASET_NAME,
        "seed": seed,
        "requested_pairs": pairs,
        "annotation_file": os.path.relpath(annotation_cache, root).replace(os.sep, "/"),
        "min_area_fraction": min_area_fraction,
        "max_area_fraction": max_area_fraction,
        "pairs": [],
    }
    if dry_run:
        for pair in selection:
            report["pairs"].append({
                "reference_image": pair.reference.file_name,
                "target_image": pair.target.file_name,
                "category": pair.reference.category,
                "scale_condition": pair.scale_condition,
            })
        print(json.dumps(report, indent=2))
        return report

    images_dir = os.path.join(root, "raw", "coco2017-val")
    entries: List[Dict[str, Any]] = []
    started = time.time()
    for index, pair in enumerate(selection, start=1):
        metadata = write_sample(pair, index, root, images_dir, quiet=False)
        entries.append(metadata)
        report["pairs"].append({
            "sample_id": metadata["sample_id"],
            "category": metadata["category"],
            "scale_condition": metadata["scale_condition"],
        })
        print(f"[{index:3d}/{len(selection)}] {metadata['sample_id']} "
              f"{metadata['category']:<14s} {metadata['scale_condition']:<18s} "
              f"area x{metadata['measured']['area_ratio_target_over_reference']:.2f}")

    buckets = split_samples(entries, seed)
    write_splits(buckets, root, seed, quiet)

    index_path = os.path.join(root, "processed", "source_index.json")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    report["samples"] = len(entries)
    report["categories"] = sorted({metadata["category"] for metadata in entries})
    report["elapsed_seconds"] = round(time.time() - started, 1)
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"built {len(entries)} samples in {report['elapsed_seconds']}s "
          f"over {len(report['categories'])} categories")
    print(f"source index -> {index_path}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data", help="dataset root (default: data)")
    parser.add_argument("--pairs", type=int, default=100, help="number of pairs to build")
    parser.add_argument("--seed", type=int, default=0, help="deterministic selection seed")
    parser.add_argument("--min-area-fraction", type=float, default=0.005,
                        help="drop instances smaller than this share of the image")
    parser.add_argument("--max-area-fraction", type=float, default=0.6,
                        help="drop instances larger than this share of the image")
    parser.add_argument("--categories-cap", type=int, default=None,
                        help="use at most this many categories")
    parser.add_argument("--annotations", default=None,
                        help="use an existing instances_val2017.json instead of fetching")
    parser.add_argument("--annotations-url", default=COCO_2017_VAL_ANNOTATIONS_URL)
    parser.add_argument("--dry-run", action="store_true", help="select pairs and print them only")
    arguments = parser.parse_args(argv)
    if arguments.pairs < 1:
        parser.error("--pairs must be positive")
    build(root=arguments.root, pairs=arguments.pairs, seed=arguments.seed,
          min_area_fraction=arguments.min_area_fraction,
          max_area_fraction=arguments.max_area_fraction,
          categories_cap=arguments.categories_cap,
          annotation_path=arguments.annotations,
          annotation_url=arguments.annotations_url,
          dry_run=arguments.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
