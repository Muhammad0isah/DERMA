"""Validate a plaintext ZIP before encrypting it."""
import argparse
import json
from .task import validate_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    try:
        result = validate_submission(args.predictions, args.manifest)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Validation failed: {exc}\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
