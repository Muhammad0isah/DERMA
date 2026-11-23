"""Synthetic tests exercise archive contracts and metrics, not model quality."""
from contextlib import contextmanager
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np
from PIL import Image

from competition import task
from scripts.prepare_data import Source, add_noise, prepare


def png(pixels, mode="L"):
    result = io.BytesIO()
    Image.fromarray(np.array(pixels, dtype=np.uint8), mode=mode).save(result, "PNG")
    return result.getvalue()


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "test_manifest.json"
        self.manifest.write_text(json.dumps({"benchmark_id": task.BENCHMARK_ID,
            "images": [{"image_id": "test_000001", "file_name": "test/images/test_000001.png",
                        "width": 2, "height": 2}],
            "categories": [{"id": 0, "name": "background"}, {"id": 1, "name": "lesion"}]}))

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, name="pred.zip", pixels=((255, 0), (0, 0)), *, mask_bytes=None, manifest=None, extra=None):
        path = self.root / name
        metadata = manifest or {"benchmark_id": task.BENCHMARK_ID, "image_ids": ["test_000001"]}
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(metadata))
            archive.writestr("masks/test_000001.png", mask_bytes or png(pixels))
            if extra:
                archive.writestr(*extra)
        return path

    def test_perfect_and_all_background(self):
        truth = self.archive("truth.zip")
        self.assertEqual(task.evaluate(truth, truth, self.manifest)["score"], 1)
        background = self.archive("background.zip", ((0, 0), (0, 0)))
        result = task.evaluate(background, truth, self.manifest)
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["metrics"]["f1_score"], 0)
        self.assertTrue(task.validate_submission(truth, self.manifest)["valid"])

    def test_empty_both_scores_one_and_false_positive_scores_zero(self):
        empty = self.archive("empty.zip", ((0, 0), (0, 0)))
        self.assertEqual(task.evaluate(empty, empty, self.manifest)["score"], 1)
        positive = self.archive("positive.zip")
        self.assertEqual(task.evaluate(positive, empty, self.manifest)["score"], 0)

    def test_per_image_average_and_dice(self):
        target = self.archive("truth.zip", ((255, 255), (0, 0)))
        pred = self.archive("pred.zip", ((255, 0), (255, 0)))
        result = task.evaluate(pred, target, self.manifest)
        self.assertAlmostEqual(result["score"], 0.5)
        self.assertAlmostEqual(result["metrics"]["f1_score"], 0.5)
        data = json.loads(self.manifest.read_text())
        data["images"].append({"image_id": "test_000002", "file_name": "test/images/test_000002.png", "width": 1, "height": 1})
        self.manifest.write_text(json.dumps(data))
        metadata = {"benchmark_id": task.BENCHMARK_ID, "image_ids": ["test_000001", "test_000002"]}
        for path, original in ((target, ((255, 255), (0, 0))), (pred, ((255, 0), (255, 0)))):
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("manifest.json", json.dumps(metadata))
                archive.writestr("masks/test_000001.png", png(original))
                archive.writestr("masks/test_000002.png", png(((0,),)))
        self.assertAlmostEqual(task.evaluate(pred, target, self.manifest)["score"], (0.5 + 1) / 2)

    def test_reject_bad_values_dimensions_rgb_and_fake_png(self):
        cases = [png(((1, 0), (0, 0))), png(((255, 0, 0),)),
                 png(np.zeros((2, 2, 3), dtype=np.uint8), "RGB"), b"not an image"]
        for index, payload in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                task.validate_submission(self.archive(f"bad{index}.zip", mask_bytes=payload), self.manifest)

    def test_reject_traversal_absolute_backslash_and_extra_member(self):
        for index, name in enumerate(("../escape", "/escape", "masks\\escape", "C:/escape", "unexpected.txt")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                task.validate_submission(self.archive(f"bad{index}.zip", extra=(name, b"x")), self.manifest)

    def test_reject_symlink_duplicate_and_oversized_members(self):
        symlink = zipfile.ZipInfo("link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(ValueError):
            task.validate_submission(self.archive("link.zip", extra=(symlink, b"/tmp/target")), self.manifest)
        path = self.archive("duplicate.zip")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("masks/test_000001.png", png(((0, 0), (0, 0))))
        with self.assertRaises(ValueError):
            task.validate_submission(path, self.manifest)
        with patch.object(task, "MAX_MEMBER_BYTES", 8), self.assertRaises(ValueError):
            task.validate_submission(self.archive("large.zip"), self.manifest)

    def test_reject_wrong_benchmark_missing_ids_and_duplicate_json_keys(self):
        cases = [{"benchmark_id": "other", "image_ids": ["test_000001"]},
                 {"benchmark_id": task.BENCHMARK_ID, "image_ids": []},
                 {"benchmark_id": task.BENCHMARK_ID, "image_ids": ["test_000001", "test_000001"]}]
        for index, metadata in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                task.validate_submission(self.archive(f"wrong{index}.zip", manifest=metadata), self.manifest)
        path = self.root / "dupe-key.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", '{"benchmark_id":"derma-v1","image_ids":[],"image_ids":["test_000001"]}')
            archive.writestr("masks/test_000001.png", png(((255, 0), (0, 0))))
        with self.assertRaises(ValueError):
            task.validate_submission(path, self.manifest)

    def test_invalid_zip_is_validation_error(self):
        path = self.root / "bad.zip"
        path.write_bytes(b"not zip")
        with self.assertRaises(ValueError):
            task.validate_submission(path, self.manifest)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = []
        self.mask_bytes = png(np.array([[0, 255] * 4] * 6, dtype=np.uint8))
        for name, ids, masks in (("train_images", (1, 2), False), ("train_masks", (1, 2), True),
                                 ("test_images", (3,), False), ("test_masks", (3,), True)):
            path = self.root / f"{name}.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for image_id in ids:
                    original = f"ISIC_{image_id:07d}"
                    if masks:
                        filename, raw = f"{original}_Segmentation.png", self.mask_bytes
                    else:
                        filename = original + ".jpg"
                        buffer = io.BytesIO()
                        Image.new("RGB", (8, 6), (120, 140, 100)).save(buffer, "JPEG")
                        raw = buffer.getvalue()
                    archive.writestr(f"original/{filename}", raw)
            self.sources.append(path)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_release_enforcement(self):
        with self.assertRaisesRegex(ValueError, "exactly 900"):
            prepare(*self.sources, self.root / "official")

    def test_tiny_fixture_retains_all_labels_deterministically(self):
        first, second = self.root / "first", self.root / "second"
        a = prepare(*self.sources, first, expected_counts=(2, 1))
        b = prepare(*self.sources, second, expected_counts=(2, 1))
        self.assertEqual(a["counts"], {"train": 2, "test": 1})
        self.assertEqual(a, b)
        for image in (first / "data/public").rglob("*.png"):
            relative = image.relative_to(first)
            self.assertEqual(image.read_bytes(), (second / relative).read_bytes())
        self.assertEqual((first / "data/public/train/masks/train_000001.png").read_bytes(), self.mask_bytes)
        self.assertEqual((first / "evaluation-private/ground_truth.zip").read_bytes(), (second / "evaluation-private/ground_truth.zip").read_bytes())
        self.assertEqual(len(list((first / "data/public/test/images").glob("*.png"))), 1)
        self.assertFalse((first / "data/public/test/masks").exists())
        truth = first / "evaluation-private/ground_truth.zip"
        manifest = first / "data/public/test_manifest.json"
        self.assertEqual(task.evaluate(truth, truth, manifest)["score"], 1)
        with zipfile.ZipFile(truth) as archive:
            self.assertEqual(archive.read("masks/test_000001.png"), self.mask_bytes)

    def test_noise_is_deterministic_and_source_unchanged(self):
        original = np.full((10, 10, 3), 120, dtype=np.uint8)
        a = add_noise(original, "train", "ISIC_0000001")
        self.assertTrue(np.array_equal(a, add_noise(original, "train", "ISIC_0000001")))
        self.assertFalse(np.array_equal(a, original))
        self.assertTrue(np.all(original == 120))

    def test_source_zip_traversal_rejected(self):
        path = self.root / "unsafe.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("../ISIC_0000001.jpg", b"invalid")
        with self.assertRaises(ValueError):
            Source(path)


if __name__ == "__main__":
    unittest.main()
