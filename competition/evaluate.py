"""Score a plaintext prediction ZIP; private ground truth is organizer-only."""
import argparse
import json
from .task import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions")
    parser.add_argument("truth")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.predictions, args.truth, args.manifest)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
