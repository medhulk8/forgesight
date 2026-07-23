"""Recompute + print the dataset manifest from an on-disk processed dataset
(§13 step 5). Reads the HF splits written by build_dataset and emits stats for the
README. Standalone from a build run.

Usage:  python scripts/make_manifest.py [--out data/processed]
"""

from __future__ import annotations

import argparse
import json
import os

from forgesight.data import build_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    split_records = {}
    for sp in ("train", "val", "test"):
        path = os.path.join(args.out, f"{sp}.jsonl")
        if os.path.isfile(path):
            with open(path) as f:
                split_records[sp] = [json.loads(line) for line in f if line.strip()]

    manifest = build_dataset.compute_manifest(split_records)
    build_dataset.write_manifest(args.out, manifest)
    build_dataset.print_manifest(manifest)


if __name__ == "__main__":
    main()
