"""Build the ForgeSight dataset: forgery pipeline → doc-level-split, class-balanced
train/val/test on disk (§5, §13 step 5).

Per base doc: 1 clean + up to N distinct-op tampered variants (all sharing the
doc_id). Docs are split by doc_id (70/15/15) so every variant of a doc lands in
ONE split — no leakage. TRAIN is balanced to 50/50 clean:tampered by oversampling
clean rows; val/test keep their natural distribution (duplicating eval rows would
bias ForgeBench). Splice donors are harvested per (split, source) to avoid a
cross-split pixel leak. All images saved lossless PNG.

Writes:
    <out>/images/<split>/<doc_id>__<variant>.png
    <out>/<split>/                         (HF dataset, save_to_disk)
    <out>/manifest.json                    (stats, via make_manifest)

Run:  python -m forgesight.data.build_dataset [--config configs/data.yaml] [--out ...]
"""

from __future__ import annotations

import argparse
import os
import random

from .. import schema
from ..forgery import pipeline
from . import ingest


# --------------------------------------------------------------------------- #
# pure helpers (offline-testable)
# --------------------------------------------------------------------------- #
def assign_splits(doc_ids, ratios, seed=0):
    """Deterministically map each unique doc_id → 'train'|'val'|'test'."""
    ids = sorted(set(doc_ids))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["val"]))
    out = {}
    for i, did in enumerate(ids):
        out[did] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out


def balance_5050(records, seed=0):
    """Oversample the minority class (clean vs tampered) with replacement until the
    two match. Returns a new shuffled list. No-op if either class is empty."""
    clean = [r for r in records if not r["tampered"]]
    tamp = [r for r in records if r["tampered"]]
    if not clean or not tamp:
        return list(records)
    rng = random.Random(seed)
    if len(clean) < len(tamp):
        clean = clean + [rng.choice(clean) for _ in range(len(tamp) - len(clean))]
    elif len(tamp) < len(clean):
        tamp = tamp + [rng.choice(tamp) for _ in range(len(clean) - len(tamp))]
    out = clean + tamp
    rng.shuffle(out)
    return out


def _variant_tag(rec):
    return "clean" if not rec["tampered"] else rec["tamper_type"]


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(config: dict):
    out_dir = config["out_dir"]
    seed = config["seed"]
    img_dir_root = os.path.join(out_dir, "images")
    os.makedirs(img_dir_root, exist_ok=True)

    docs = list(ingest.load_source_docs(
        sources=tuple(config["sources"]),
        limit_per_source=config.get("limit_per_source")))
    split_of = assign_splits([d["doc_id"] for d in docs],
                             config["split_ratios"], seed=seed)

    # group docs by split, build per-(split,source) donor pool
    by_split: dict[str, list] = {"train": [], "val": [], "test": []}
    for d in docs:
        by_split[split_of[d["doc_id"]]].append(d)
    donor_pools = {sp: pipeline.harvest_donor_crops(ds) for sp, ds in by_split.items()}

    split_records: dict[str, list] = {}
    for sp, ds in by_split.items():
        img_dir = os.path.join(img_dir_root, sp)
        os.makedirs(img_dir, exist_ok=True)
        records = []
        for doc in ds:
            rng = random.Random(pipeline._doc_seed(doc["doc_id"], seed))
            for image, rec in pipeline.forge_variants(
                    doc, donor_pools[sp], rng, n_tampered=config["n_tampered_variants"]):
                fname = f"{doc['doc_id']}__{_variant_tag(rec)}.png"
                rel = os.path.join("images", sp, fname)
                image.save(os.path.join(img_dir, fname), format="PNG")
                rec["image_path"] = rel
                schema.validate_record(rec)   # fail fast on any malformed target
                records.append(rec)
        split_records[sp] = records

    # balance TRAIN only
    if config.get("balance_train_5050", True):
        split_records["train"] = balance_5050(split_records["train"], seed=seed)

    # leakage assertion: no doc_id in more than one split
    _assert_no_leakage(split_records)

    _write_hf_datasets(out_dir, split_records)
    return split_records


def _assert_no_leakage(split_records):
    seen = {}
    for sp, recs in split_records.items():
        for r in recs:
            prev = seen.get(r["doc_id"])
            assert prev in (None, sp), (
                f"LEAKAGE: doc_id {r['doc_id']} in both {prev!r} and {sp!r}")
            seen[r["doc_id"]] = sp


def _write_hf_datasets(out_dir, split_records):
    from datasets import Dataset

    for sp, recs in split_records.items():
        Dataset.from_list(recs).save_to_disk(os.path.join(out_dir, sp))


# --------------------------------------------------------------------------- #
# manifest (stats for README + gates)
# --------------------------------------------------------------------------- #
def compute_manifest(split_records: dict) -> dict:
    import collections

    out = {"splits": {}, "totals": collections.Counter()}
    for sp, recs in split_records.items():
        n = len(recs)
        clean = sum(1 for r in recs if not r["tampered"])
        tamp = n - clean
        by_type = collections.Counter(r["tamper_type"] for r in recs if r["tampered"])
        by_source = collections.Counter(r["source"] for r in recs)
        docs = len({r["doc_id"] for r in recs})
        out["splits"][sp] = {
            "examples": n, "docs": docs, "clean": clean, "tampered": tamp,
            "clean_frac": round(clean / n, 3) if n else 0.0,
            "by_tamper_type": dict(by_type), "by_source": dict(by_source),
        }
        out["totals"]["examples"] += n
        out["totals"]["clean"] += clean
        out["totals"]["tampered"] += tamp
    out["totals"] = dict(out["totals"])
    return out


def write_manifest(out_dir, manifest):
    import json

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def print_manifest(manifest):
    import json

    print(json.dumps(manifest, indent=2))


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def load_config(path):
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out", default=None, help="override out_dir")
    ap.add_argument("--limit-per-source", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.out:
        cfg["out_dir"] = args.out
    if args.limit_per_source is not None:
        cfg["limit_per_source"] = args.limit_per_source

    split_records = build(cfg)
    manifest = compute_manifest(split_records)
    write_manifest(cfg["out_dir"], manifest)
    print_manifest(manifest)


if __name__ == "__main__":
    main()
