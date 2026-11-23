

"""Strict, bounded mask archive validation and foreground F1 scoring.

Untrusted ZIP members are read in memory individually; nothing is extracted.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import warnings
import zipfile

import numpy as np
from PIL import Image

BENCHMARK_ID = "derma-v1"
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_PIXELS = 64_000_000
MAX_MEMBERS = 5000


def _json_bytes(raw: bytes) -> dict:
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_hook,
                            parse_constant=invalid_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid UTF-8 JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("JSON must contain an object")
    return result


def load_manifest(path: str | Path) -> dict:
    """Read a trusted organizer manifest, validating its essential schema."""
    path = Path(path)
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Manifest is too large")
    manifest = _json_bytes(path.read_bytes())
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Wrong benchmark_id")
    images = manifest.get("images")
    if not isinstance(images, list) or not images or len(images) > 2000:
        raise ValueError("Manifest requires a nonempty images list")
    ids = set()
    for image in images:
        if not isinstance(image, dict):
            raise ValueError("Invalid image record")
        image_id = image.get("image_id")
        if not isinstance(image_id, str) or not re.fullmatch(r"(?:train|test)_\d{6}", image_id):
            raise ValueError("Invalid image_id")
        if image_id in ids:
            raise ValueError("Duplicate manifest image_id")
        ids.add(image_id)
        width, height = image.get("width"), image.get("height")
        if type(width) is not int or type(height) is not int or min(width, height) < 1:
            raise ValueError("Invalid image dimensions")
        if width * height > MAX_PIXELS:
            raise ValueError("Image exceeds pixel limit")
        file_name = image.get("file_name")
        if not isinstance(file_name, str):
            raise ValueError("Missing file_name")
        _safe_name(file_name)
    return manifest


def _safe_name(name: str) -> None:
    if not name or "\\" in name or "\x00" in name or ":" in name:
        raise ValueError("Unsafe ZIP member name")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in (".", "..") for part in name.split("/") if part):
        raise ValueError("Unsafe ZIP member path")
    if "//" in name:
        raise ValueError("Noncanonical ZIP member path")


class MaskArchive:
    """Bounded archive reader shared by prediction and organizer truth archives."""

    def __init__(self, path: str | Path, manifest: dict):
        self.path = Path(path)
        self.zip = None
        if self.path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Archive exceeds compressed-size limit")
        self.manifest = manifest
        self.images = {image["image_id"]: image for image in manifest["images"]}
        try:
            self.zip = zipfile.ZipFile(self.path)
            self._validate_members()
        except zipfile.BadZipFile as exc:
            if self.zip is not None:
                self.zip.close()
            raise ValueError("Invalid ZIP archive") from exc
        except Exception:
            if self.zip is not None:
                self.zip.close()
            raise

    def _validate_members(self):
        infos = self.zip.infolist()
        if len(infos) > MAX_MEMBERS:
            raise ValueError("Too many ZIP members")
        names, total = set(), 0
        expected = {"manifest.json"} | {f"masks/{key}.png" for key in self.images}
        actual_files = set()
        for member in infos:
            _safe_name(member.orig_filename)
            _safe_name(member.filename)
            if member.filename in names:
                raise ValueError("Duplicate ZIP member")
            names.add(member.filename)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError("Nonregular ZIP member")
            if member.flag_bits & 1:
                raise ValueError("Inner ZIP members must not be encrypted")
            if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("Unsupported ZIP compression")
            if member.is_dir():
                if member.filename != "masks/":
                    raise ValueError("Unexpected directory in archive")
                continue
            limit = MAX_MANIFEST_BYTES if member.filename == "manifest.json" else MAX_MEMBER_BYTES
            if member.file_size > limit:
                raise ValueError("ZIP member exceeds size limit")
            total += member.file_size
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("Archive exceeds uncompressed-size limit")
            actual_files.add(member.filename)
        if actual_files != expected:
            raise ValueError("Archive must contain manifest.json and exactly one PNG per image_id")
        metadata = _json_bytes(self.read("manifest.json", MAX_MANIFEST_BYTES))
        if set(metadata) != {"benchmark_id", "image_ids"}:
            raise ValueError("Submission manifest requires only benchmark_id and image_ids")
        if metadata["benchmark_id"] != self.manifest["benchmark_id"]:
            raise ValueError("Submission benchmark_id mismatch")
        image_ids = metadata["image_ids"]
        if not isinstance(image_ids, list) or not all(isinstance(key, str) for key in image_ids):
            raise ValueError("image_ids must be a list of strings")
        if len(image_ids) != len(set(image_ids)) or set(image_ids) != set(self.images):
            raise ValueError("Submission image_ids must match all manifest images exactly")

    def read(self, name: str, limit: int = MAX_MEMBER_BYTES) -> bytes:
        try:
            with self.zip.open(name) as stream:
                raw = stream.read(limit + 1)
                if len(raw) > limit:
                    raise ValueError("ZIP member exceeds size limit")
                return raw
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
            raise ValueError("Invalid or unsupported ZIP member") from exc

    def mask(self, image_id: str) -> np.ndarray:
        expected = self.images[image_id]
        raw = self.read(f"masks/{image_id}.png")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format != "PNG" or image.mode != "L":
                        raise ValueError("Masks must be single-channel 8-bit PNG (mode L)")
                    if image.size != (expected["width"], expected["height"]):
                        raise ValueError(f"Wrong mask dimensions for {image_id}")
                    if image.width * image.height > MAX_PIXELS or getattr(image, "n_frames", 1) != 1:
                        raise ValueError("Mask exceeds pixel limit or has multiple frames")
                    pixels = np.asarray(image, dtype=np.uint8)
                    if not np.all((pixels == 0) | (pixels == 255)):
                        raise ValueError("Mask values must be exactly 0 or 255")
                    return pixels == 255
            except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                raise ValueError(f"Invalid PNG for {image_id}") from exc

    def close(self):
        self.zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def validate_submission(path: str | Path, manifest_path: str | Path) -> dict:
    manifest = load_manifest(manifest_path)
    with MaskArchive(path, manifest) as archive:
        for image in manifest["images"]:
            archive.mask(image["image_id"])
    return {"valid": True, "benchmark_id": BENCHMARK_ID, "num_images": len(manifest["images"])}


def evaluate(predictions_path: str | Path, truth_path: str | Path, manifest_path: str | Path) -> dict:
    """Unweighted mean per-image foreground F1 (Dice)."""
    manifest = load_manifest(manifest_path)
    ious, dices = [], []
    with MaskArchive(predictions_path, manifest) as predictions, MaskArchive(truth_path, manifest) as truth:
        for image in manifest["images"]:
            image_id = image["image_id"]
            pred, target = predictions.mask(image_id), truth.mask(image_id)
            intersection = int(np.count_nonzero(pred & target))
            union = int(np.count_nonzero(pred | target))
            size_sum = int(np.count_nonzero(pred)) + int(np.count_nonzero(target))
            ious.append(intersection / union if union else 1.0)
            dices.append(2 * intersection / size_sum if size_sum else 1.0)
    mean_f1 = math.fsum(dices) / len(dices)
    return {"score": mean_f1, "primary_metric": "mean_foreground_f1",
            "metrics": {"f1_score": mean_f1}}
