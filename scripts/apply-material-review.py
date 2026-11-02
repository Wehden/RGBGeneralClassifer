#!/usr/bin/env python3
"""Apply a provenance-bound exclusion/flag review to a new weak-label document."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgb.material_review import write_reviewed_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_reviewed_labels(args.dataset, args.labels, args.review, args.output), indent=2))
