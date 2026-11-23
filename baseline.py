#!/usr/bin/env python3
"""Train a GridGNN-UNet from scratch on CPU and write a validated mask ZIP.

The complete official training partition is used. No test labels, external
weights, training subsets, or network access are used by this program.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import os
from pathlib import Path
import platform
import random
import time
import zipfile

import numpy as np
from PIL import Image

from competition.task import BENCHMARK_ID, load_manifest, validate_submission


def letterbox(image, size=256, *, mask=False):
    """Aspect-preserving resize; categorical targets use nearest interpolation."""
    width, height = image.size
    scale = min(size / width, size / height)
    new_width = max(1, min(size, round(width * scale)))
    new_height = max(1, min(size, round(height * scale)))
    interpolation = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    resized = image.resize((new_width, new_height), interpolation)
    canvas = Image.new("L" if mask else "RGB", (size, size), 0)
    left, top = (size - new_width) // 2, (size - new_height) // 2
    canvas.paste(resized, (left, top))
    return np.asarray(canvas).copy(), (left, top, new_width, new_height)


def build_grid_gnn_unet(torch):
    nn = torch.nn
    functional = torch.nn.functional

    def block(in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                             nn.GroupNorm(4, out_channels), nn.ReLU(inplace=True),
                             nn.Conv2d(out_channels, out_channels, 3, padding=1),
                             nn.GroupNorm(4, out_channels), nn.ReLU(inplace=True))

    class GridGraphLayer(nn.Module):
        """Message passing on the four-neighbour graph formed by a feature map.

        Each spatial location is a graph node. Edges join up, down, left, and
        right neighbours; no wraparound edges are introduced at image borders.
        """
        def __init__(self, channels):
            super().__init__()
            self.self_projection = nn.Conv2d(channels, channels, 1, bias=False)
            self.neighbour_projection = nn.Conv2d(channels, channels, 1, bias=False)
            self.norm = nn.GroupNorm(4, channels)

        def forward(self, features):
            neighbours = (
                functional.pad(features[:, :, :-1, :], (0, 0, 1, 0)) +
                functional.pad(features[:, :, 1:, :], (0, 0, 0, 1)) +
                functional.pad(features[:, :, :, :-1], (1, 0, 0, 0)) +
                functional.pad(features[:, :, :, 1:], (0, 1, 0, 0))
            )
            height, width = features.shape[-2:]
            degree = features.new_full((1, 1, height, width), 4.)
            degree[:, :, 0, :] -= 1
            degree[:, :, -1, :] -= 1
            degree[:, :, :, 0] -= 1
            degree[:, :, :, -1] -= 1
            return functional.relu(self.norm(self.self_projection(features) +
                                             self.neighbour_projection(neighbours / degree)), inplace=True)

    class GridGNNUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1, self.enc2 = block(3, 8), block(8, 16)
            self.enc3, self.center = block(16, 32), block(32, 64)
            self.graph1, self.graph2 = GridGraphLayer(64), GridGraphLayer(64)
            self.dec3, self.dec2, self.dec1 = block(96, 32), block(48, 16), block(24, 8)
            self.out = nn.Conv2d(8, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            a = self.enc1(x)
            b = self.enc2(self.pool(a))
            c = self.enc3(self.pool(b))
            d = self.center(self.pool(c))
            d = self.graph2(self.graph1(d))
            d = self.dec3(torch.cat([nn.functional.interpolate(d, size=c.shape[-2:], mode="bilinear", align_corners=False), c], dim=1))
            d = self.dec2(torch.cat([nn.functional.interpolate(d, size=b.shape[-2:], mode="bilinear", align_corners=False), b], dim=1))
            d = self.dec1(torch.cat([nn.functional.interpolate(d, size=a.shape[-2:], mode="bilinear", align_corners=False), a], dim=1))
            return self.out(d)

    return GridGNNUNet()


def _resolve_data(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest path escapes public data directory")
    return path


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/public"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.zip"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--max-seconds", type=int, default=10800)
    args = parser.parse_args()
    if not (1 <= args.threads <= 4 and args.epochs >= 1 and args.batch_size >= 1
            and 64 <= args.input_size <= 512 and args.input_size % 8 == 0
            and 1 <= args.max_seconds <= 10800):
        parser.error("Use threads1..4, epochs/batch-size>=1, input-size64..512 divisible by8, max-seconds1..10800")
    if args.output.exists():
        parser.error("Output exists; choose a fresh path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime_path = args.output.with_suffix(".runtime.json")
    if runtime_path.exists():
        parser.error("Runtime log exists; choose a fresh output path")
    # Fix CPU parallelism before importing PyTorch.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    report = {"benchmark_id": BENCHMARK_ID, "seed": 25, "status": "started",
              "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                           "logical_cpu_count": os.cpu_count()},
              "threads": args.threads, "device": "cpu", "input_size": args.input_size,
              "batch_size": args.batch_size, "max_seconds": args.max_seconds,
              "requested_epochs": args.epochs, "completed_epochs": 0, "epochs": [],
              "initialization": "random, no external pretrained weights"}
    temporary_output = args.output.with_suffix(".partial.zip")
    try:
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(25)
        torch.use_deterministic_algorithms(True)
        np.random.seed(25)
        random.seed(25)
        report["versions"] = {"torch": torch.__version__, "numpy": np.__version__}
        deadline = started + args.max_seconds
        # Reserve 20% for complete native-resolution prediction, archive writing,
        # and validation. An over-budget run exits unsuccessfully.
        training_deadline = deadline - max(1, args.max_seconds * 0.20)

        def check_deadline():
            if time.monotonic() >= deadline:
                raise TimeoutError("CPU wall-time budget exhausted; no successful prediction archive produced")

        training = load_manifest(args.data / "train_manifest.json")
        test = load_manifest(args.data / "test_manifest.json")
        if len(training["images"]) != 900 or len(test["images"]) != 379:
            raise ValueError("Baseline requires the complete official 900/379 release")
        if any(not row["image_id"].startswith("train_") for row in training["images"]):
            raise ValueError("Training manifest contains nontraining IDs")
        report["training_images"] = len(training["images"])
        report["test_images"] = len(test["images"])
        train_x, train_y = [], []
        for row in training["images"]:
            check_deadline()
            with Image.open(_resolve_data(args.data, row["file_name"])) as image:
                x, geometry = letterbox(image.convert("RGB"), args.input_size)
            with Image.open(_resolve_data(args.data, row["mask_file_name"])) as mask:
                y, mask_geometry = letterbox(mask, args.input_size, mask=True)
            if geometry != mask_geometry:
                raise ValueError("Training mask dimensions mismatch")
            train_x.append(x)
            train_y.append(y)
        # Keep only compact resized uint8 caches; cast one batch at a time.
        xs = torch.from_numpy(np.stack(train_x).transpose(0, 3, 1, 2))
        ys = torch.from_numpy(np.stack(train_y)[:, None])
        del train_x, train_y
        test_x, test_geometries = [], []
        for row in test["images"]:
            check_deadline()
            with Image.open(_resolve_data(args.data, row["file_name"])) as image:
                x, geometry = letterbox(image.convert("RGB"), args.input_size)
            test_x.append(x)
            test_geometries.append(geometry)
        test_x = torch.from_numpy(np.stack(test_x).transpose(0, 3, 1, 2))
        report["preprocessing_seconds"] = time.monotonic() - started
        model = build_grid_gnn_unet(torch).cpu()
        report["model"] = "GridGNN-UNet (two learned four-neighbour message-passing layers)"
        report["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        bce = torch.nn.BCEWithLogitsLoss()
        last_epoch_seconds = None
        for epoch in range(args.epochs):
            check_deadline()
            if last_epoch_seconds and time.monotonic() + last_epoch_seconds * 1.25 >= training_deadline:
                break
            if time.monotonic() >= training_deadline:
                break
            previous_state = copy.deepcopy(model.state_dict())
            epoch_started = time.monotonic()
            model.train()
            loss_sum, seen, finished = 0.0, 0, True
            order = torch.randperm(len(xs))
            for offset in range(0, len(xs), args.batch_size):
                if time.monotonic() >= training_deadline:
                    finished = False
                    break
                indices = order[offset:offset + args.batch_size]
                x = xs[indices].float().div_(255.0)
                y = ys[indices].float().div_(255.0)
                # Apply only paired training augmentation; test masks are unused.
                if torch.rand(()).item() < 0.5:
                    x, y = x.flip(-1), y.flip(-1)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                probabilities = logits.sigmoid()
                dims = (1, 2, 3)
                dice_loss = 1 - ((2 * (probabilities * y).sum(dims) + 1)
                                 / (probabilities.sum(dims) + y.sum(dims) + 1)).mean()
                loss = bce(logits, y) + dice_loss
                if not torch.isfinite(loss).item():
                    raise ValueError("Non-finite training loss; baseline run rejected")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(indices)
                seen += len(indices)
            if not finished:
                # Do not keep a partially trained final epoch.
                model.load_state_dict(previous_state)
                break
            last_epoch_seconds = time.monotonic() - epoch_started
            report["completed_epochs"] += 1
            record = {"epoch": epoch + 1, "images_seen": seen,
                      "mean_training_loss": loss_sum / seen, "seconds": last_epoch_seconds}
            report["epochs"].append(record)
            print(json.dumps(record), flush=True)
        if not report["completed_epochs"]:
            raise TimeoutError("No complete epoch across all900 training images fit within the CPU budget")
        prediction_started = time.monotonic()
        model.eval()
        if temporary_output.exists():
            raise ValueError("Partial output exists; choose a fresh output path")
        with zipfile.ZipFile(temporary_output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            metadata = {"benchmark_id": BENCHMARK_ID, "image_ids": [row["image_id"] for row in test["images"]]}
            archive.writestr("manifest.json", json.dumps(metadata))
            with torch.inference_mode():
                for offset in range(0, len(test_x), args.batch_size):
                    check_deadline()
                    output = (model(test_x[offset:offset + args.batch_size].float().div(255.0))[:, 0] >= 0).numpy()
                    for batch_index, pixels in enumerate(output):
                        check_deadline()
                        index = offset + batch_index
                        row = test["images"][index]
                        left, top, width, height = test_geometries[index]
                        cropped = pixels[top:top + height, left:left + width].astype(np.uint8) * 255
                        native = Image.fromarray(cropped).resize((row["width"], row["height"]), Image.Resampling.NEAREST)
                        buffer = io.BytesIO()
                        native.save(buffer, format="PNG")
                        archive.writestr(f"masks/{row['image_id']}.png", buffer.getvalue())
        check_deadline()
        validate_submission(temporary_output, args.data / "test_manifest.json")
        check_deadline()
        temporary_output.rename(args.output)
        report["prediction_and_validation_seconds"] = time.monotonic() - prediction_started
        report["status"] = "complete"
        print(f"Validated predictions: {args.output}", flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["total_wall_seconds"] = time.monotonic() - started
        runtime_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
