"""Visualize forgeries with their GT box drawn — the §13 step-4 eyeball gate.

Forces each op on real source docs, overlays the GT tamper box (green), and saves
PNGs so bad boxes / unrealistic forgeries are caught before generating the full
dataset (bad GT = poisoned training + meaningless IoU).

Usage:
    python scripts/viz_sample.py --out <dir> [--n 6] [--sources sroie funsd]
                                 [--ops digit_swap copy_move splice recompress_ghost]
                                 [--limit-per-source 120] [--seed 0]
"""

from __future__ import annotations

import argparse
import os
import random

from PIL import ImageDraw

from forgesight.data import ingest
from forgesight.forgery import pipeline


def draw_gt(image, gt):
    img = image.copy()
    d = ImageDraw.Draw(img)
    for pad in range(2):  # thicker outline
        d.rectangle([gt[0] - pad, gt[1] - pad, gt[2] + pad, gt[3] + pad],
                    outline=(0, 200, 0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6, help="samples per op per source")
    ap.add_argument("--sources", nargs="+", default=["sroie", "funsd"])
    ap.add_argument("--ops", nargs="+", default=pipeline.OP_NAMES)
    ap.add_argument("--limit-per-source", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    made = {}
    for source in args.sources:
        docs = list(ingest.load_source_docs(sources=(source,),
                                            limit_per_source=args.limit_per_source))
        donor_pool = pipeline.harvest_donor_crops(docs)
        for op_name in args.ops:
            count = 0
            for i, doc in enumerate(docs):
                if count >= args.n:
                    break
                ops = pipeline.build_ops(doc["source"], donor_pool)
                op = ops[op_name]
                if not op.applicable(doc):
                    continue
                rng = random.Random(pipeline._doc_seed(doc["doc_id"], args.seed))
                try:
                    img, gt, meta = op.apply(doc["image"], doc["ocr_boxes"], rng)
                except ValueError:
                    continue
                out = os.path.join(args.out, f"{source}_{op_name}_{count}.png")
                draw_gt(img, gt).save(out)
                count += 1
            made[f"{source}:{op_name}"] = count
    print("samples written:", made, "->", args.out)


if __name__ == "__main__":
    main()
