#!/usr/bin/env python3
"""List official source URLs, or download all four archives with --download.

Test annotation archives are organizer-only. This script never extracts them.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile

BASE = "https://isic-archive.s3.amazonaws.com/challenges/2016/"
ARCHIVES = [
    "ISBI2016_ISIC_Part1_Training_Data.zip",
    "ISBI2016_ISIC_Part1_Training_GroundTruth.zip",
    "ISBI2016_ISIC_Part1_Test_Data.zip",
    "ISBI2016_ISIC_Part1_Test_GroundTruth.zip",
]
MAX_BYTES = 2 * 1024**3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Explicitly enable approximately 842 MB transfer")
    parser.add_argument("--output", type=Path, default=Path("evaluation-private/source"))
    args = parser.parse_args()
    if not args.download:
        print(json.dumps({"source_page": "https://challenge.isic-archive.com/data/",
                          "urls": [BASE + name for name in ARCHIVES],
                          "downloaded": False}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    report = []
    for name in ARCHIVES:
        destination = args.output / name
        temporary = args.output / (name + ".partial")
        if destination.exists() or temporary.exists():
            parser.error(f"Refusing to overwrite existing file: {destination} or {temporary}")
        digest, total = hashlib.sha256(), 0
        try:
            request = urllib.request.Request(BASE + name, headers={"User-Agent": "DERMA-organizer/1"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as stream:
                if response.geturl().split(":", 1)[0] != "https":
                    raise ValueError("Refusing a non-HTTPS redirect")
                os.chmod(temporary, 0o600)
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > MAX_BYTES:
                        raise ValueError("Download exceeds safety limit")
                    stream.write(block)
                    digest.update(block)
            with zipfile.ZipFile(temporary) as archive:
                if archive.testzip() is not None:
                    raise ValueError("Downloaded ZIP failed CRC check")
            temporary.rename(destination)
        except Exception:
            # Leave partial bytes for organizer inspection; never treat as a valid ZIP.
            raise
        report.append({"url": BASE + name, "file": name, "bytes": total, "sha256": digest.hexdigest()})
        print(f"Downloaded and CRC-checked {name}: {total} bytes", flush=True)
    (args.output / "download_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("All four archives remain in the private organizer source directory.")


if __name__ == "__main__":
    main()
