#!/usr/bin/env python3
"""Prepare the COMPLETE official ISIC 2016 Task 1 release from four ZIPs/folders.

No downloads, source relabeling, source split changes, or image selection occur.
Run this only in an organizer checkout: test masks/provenance remain private.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import warnings
import zipfile

import numpy as np
from PIL import Image

BENCHMARK_ID = "derma-v1"
SEED = 25
NOISE_SIGMA = 3.0
MAX_SOURCE_BYTES = 8 * 1024**3
MAX_MEMBER_BYTES = 64 * 1024**2
MAX_PIXELS = 64_000_000
CATEGORIES = [{"id": 0, "name": "background"}, {"id": 1, "name": "lesion"}]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(member):
    name = member.filename
    path = PurePosixPath(name)
    mode = member.external_attr >> 16
    if (not name or "\\" in name or ":" in name or "\x00" in member.orig_filename or path.is_absolute()
            or any(part in (".", "..") for part in name.split("/"))
            or stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
        raise ValueError("Unsafe member in source ZIP")
    if member.flag_bits & 1 or member.file_size > MAX_MEMBER_BYTES:
        raise ValueError("Encrypted or oversized source ZIP member")


class Source:
    """Read one original source archive/directory without extracting ZIP paths."""

    def __init__(self, path, masks=False):
        self.path = Path(path)
        self.zip = None
        self.entries = {}
        pattern = re.compile(r"(ISIC_\d{7})_Segmentation\.png" if masks else r"(ISIC_\d{7})\.jpe?g", re.I)
        if self.path.is_dir():
            entries = []
            for entry in sorted(self.path.rglob("*")):
                if entry.is_symlink():
                    raise ValueError("Symlink in source directory")
                if entry.is_file():
                    if entry.stat().st_size > MAX_MEMBER_BYTES:
                        raise ValueError("Oversized source file")
                    entries.append((entry.relative_to(self.path).as_posix(), entry))
            self.source_hash = None
        else:
            if self.path.stat().st_size > MAX_SOURCE_BYTES:
                raise ValueError("Source archive too large")
            self.source_hash = sha256_file(self.path)
            self.zip = zipfile.ZipFile(self.path)
            entries, names, total = [], set(), 0
            try:
                if len(self.zip.infolist()) > 10000:
                    raise ValueError("Too many source ZIP entries")
                for member in self.zip.infolist():
                    _safe_member(member)
                    if member.filename in names:
                        raise ValueError("Duplicate source ZIP member")
                    names.add(member.filename)
                    total += member.file_size
                    if total > MAX_SOURCE_BYTES:
                        raise ValueError("Source ZIP exceeds uncompressed limit")
                    if not member.is_dir():
                        entries.append((member.filename, member))
            except Exception:
                self.zip.close()
                raise
        for name, entry in entries:
            # macOS metadata is not a dataset image; retain every actual image.
            if "__MACOSX" in PurePosixPath(name).parts or PurePosixPath(name).name.startswith("._"):
                continue
            match = pattern.fullmatch(PurePosixPath(name).name)
            if match:
                key = match.group(1).upper()
                if key in self.entries:
                    self.close()
                    raise ValueError("Duplicate original image ID")
                self.entries[key] = (name, entry)
            elif PurePosixPath(name).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                self.close()
                raise ValueError(f"Unexpected dataset image filename: {name}")
        if not self.entries:
            self.close()
            raise ValueError("No official ISIC images found in source")

    def read(self, key):
        _, entry = self.entries[key]
        if self.zip is not None:
            with self.zip.open(entry) as stream:
                data = stream.read(MAX_MEMBER_BYTES + 1)
        else:
            data = entry.read_bytes()
        if len(data) > MAX_MEMBER_BYTES:
            raise ValueError("Source file exceeds size limit")
        return data

    def close(self):
        if self.zip is not None:
            self.zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load_pair(image_bytes, mask_bytes):
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format != "JPEG" or image.width * image.height > MAX_PIXELS:
                raise ValueError("Source images must be bounded JPEG files")
            # Do not EXIF-transpose: source coordinates match the supplied masks.
            rgb = np.asarray(image.convert("RGB"))
            size = image.size
        with Image.open(io.BytesIO(mask_bytes)) as mask:
            if mask.format != "PNG" or mask.mode != "L" or mask.size != size:
                raise ValueError("Source mask must be grayscale PNG matching image dimensions")
            values = np.asarray(mask)
            if not np.all((values == 0) | (values == 255)):
                raise ValueError("Source mask contains values outside {0, 255}")
            target = values.copy()
    return rgb, target


def add_noise(rgb, split, original_id):
    """Per-image seed is stable across filesystems, ordering, and processes."""
    key = f"{BENCHMARK_ID}|{SEED}|{split}|{original_id}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    # Generate one row at a time to bound temporary memory at native resolution.
    noisy = np.empty_like(rgb)
    for y in range(rgb.shape[0]):
        values = rgb[y].astype(np.float64) + rng.normal(0.0, NOISE_SIGMA, rgb[y].shape)
        noisy[y] = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return noisy


def _json_write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _zip_write(archive, name, payload):
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    archive.writestr(info, payload)


def prepare(train_images, train_masks, test_images, test_masks, output_root, *, expected_counts=(900, 379)):
    """Internal expected_counts hook is for unit fixtures; CLI always uses 900/379."""
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    public_target = output_root / "data/public"
    private_target = output_root / "evaluation-private"
    if public_target.exists() and any(public_target.iterdir()):
        raise ValueError("data/public is not empty; use a fresh organizer checkout/output root")
    for name in ("ground_truth.zip", "provenance.json"):
        if (private_target / name).exists():
            raise ValueError(f"Refusing to replace existing private artifact: {name}")
    with ExitStack() as stack:
        sources = {
            "train": (stack.enter_context(Source(train_images)), stack.enter_context(Source(train_masks, masks=True))),
            "test": (stack.enter_context(Source(test_images)), stack.enter_context(Source(test_masks, masks=True))),
        }
        for split, expected in zip(("train", "test"), expected_counts):
            images, masks = sources[split]
            if len(images.entries) != expected or set(images.entries) != set(masks.entries):
                raise ValueError(f"{split} must contain exactly {expected} matching image/mask pairs; found {len(images.entries)} images, {len(masks.entries)} masks")
        if set(sources["train"][0].entries) & set(sources["test"][0].entries):
            raise ValueError("An original image ID appears in both official partitions")
        with tempfile.TemporaryDirectory(prefix=".isic-prepare-", dir=output_root) as temp:
            stage = Path(temp)
            public, private = stage / "public", stage / "private"
            private.mkdir(mode=0o700)
            manifests, provenance = {}, {}
            truth_path = private / "ground_truth.zip"
            with zipfile.ZipFile(truth_path, "w", compression=zipfile.ZIP_DEFLATED) as truth:
                for split in ("train", "test"):
                    image_source, mask_source = sources[split]
                    image_dir = public / split / "images"
                    image_dir.mkdir(parents=True)
                    if split == "train":
                        (public / "train/masks").mkdir()
                    manifest = {"benchmark_id": BENCHMARK_ID, "images": [], "categories": CATEGORIES}
                    for index, original_id in enumerate(sorted(image_source.entries), 1):
                        image_id = f"{split}_{index:06d}"
                        raw_image, raw_mask = image_source.read(original_id), mask_source.read(original_id)
                        rgb, mask = load_pair(raw_image, raw_mask)
                        noisy = add_noise(rgb, split, original_id)
                        relative_image = f"{split}/images/{image_id}.png"
                        image_path = public / relative_image
                        Image.fromarray(noisy).save(image_path, format="PNG", compress_level=6)
                        record = {"image_id": image_id, "file_name": relative_image,
                                  "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                                  "sha256": sha256_file(image_path)}
                        if split == "train":
                            mask_relative = f"train/masks/{image_id}.png"
                            # Byte-for-byte source masks: labels never modified.
                            (public / mask_relative).write_bytes(raw_mask)
                            record["mask_file_name"] = mask_relative
                        else:
                            _zip_write(truth, f"masks/{image_id}.png", raw_mask)
                        manifest["images"].append(record)
                        provenance[image_id] = {"original_id": original_id,
                            "source_image": image_source.entries[original_id][0],
                            "source_mask": mask_source.entries[original_id][0],
                            "source_image_sha256": hashlib.sha256(raw_image).hexdigest(),
                            "source_mask_sha256": hashlib.sha256(raw_mask).hexdigest(),
                            "output_image_sha256": record["sha256"]}
                    manifests[split] = manifest
                    _json_write(public / f"{split}_manifest.json", manifest)
                payload_manifest = {"benchmark_id": BENCHMARK_ID,
                                    "image_ids": [image["image_id"] for image in manifests["test"]["images"]]}
                _zip_write(truth, "manifest.json", json.dumps(payload_manifest, sort_keys=True).encode())
            os.chmod(truth_path, 0o600)
            _json_write(private / "provenance.json", {"benchmark_id": BENCHMARK_ID, "images": provenance,
                "sources": {split: {"images_sha256": image_source.source_hash, "masks_sha256": mask_source.source_hash}
                            for split, (image_source, mask_source) in sources.items()}})
            os.chmod(private / "provenance.json", 0o600)
            metadata = {"benchmark_id": BENCHMARK_ID, "dataset": "ISIC 2016 Task 1 (complete official release)",
                "dataset_license": "CC0-1.0", "source": "https://challenge.isic-archive.com/data/",
                "split": "original official partitions, unchanged", "counts": {key: len(value["images"]) for key, value in manifests.items()},
                "modification": {"name": "additive_gaussian_rgb", "sigma_uint8": NOISE_SIGMA,
                    "seed": SEED, "per_image_seed": "first 8 SHA256 bytes, big endian, of benchmark_id|seed|split|original_id",
                    "generator": "NumPy PCG64, float64 row-major normal, rint then clip 0..255",
                    "images": "native dimensions, lossless PNG", "masks": "unchanged source PNG bytes"},
                "preparation_numpy_version": np.__version__, "training_manifest": "train_manifest.json",
                "test_manifest": "test_manifest.json", "cpu_pilot_status": "not_measured"}
            _json_write(public / "dataset.json", metadata)
            (output_root / "data").mkdir(exist_ok=True)
            if public_target.exists():
                public_target.rmdir()
            public.rename(public_target)
            private_target.mkdir(mode=0o700, exist_ok=True)
            for source in private.iterdir():
                shutil.move(str(source), str(private_target / source.name))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images", required=True, type=Path)
    parser.add_argument("--train-masks", required=True, type=Path)
    parser.add_argument("--test-images", required=True, type=Path)
    parser.add_argument("--test-masks", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("."), type=Path)
    args = parser.parse_args()
    try:
        metadata = prepare(args.train_images, args.train_masks, args.test_images, args.test_masks, args.output_root)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print("Keep evaluation-private/ private. Seal ground_truth.zip before configuring automated scoring.")


if __name__ == "__main__":
    main()
