"""Forgery pipeline — orchestrate ops over source-doc records (§5, §13 step 4).

Per source-doc: with probability `p_clean` emit a clean hard-negative (D5), else
sample an applicable op and apply it. Yields `(image, record)` where record is the
§4 training shape (image_path left None — `data.build_dataset` fills + persists it
at step 5). Class balance / final split are step 5's job; this module just forges.

Determinism: a per-doc RNG is seeded from the doc_id (stable across runs) so the
same doc always produces the same forgery for a given base seed.
"""

from __future__ import annotations

import random
import zlib

from .copy_move import CopyMoveOp
from .digit_swap import DigitSwapOp
from .recompress_ghost import RecompressGhostOp
from .splice import SpliceOp

# canonical op order (D-order, §5)
OP_NAMES = ["digit_swap", "copy_move", "splice", "recompress_ghost"]

# labels worth harvesting as splice donors (field values, not headers/noise)
_DONOR_LABELS = {"total", "line_total", "date", "company", "address", "answer"}


def _doc_seed(doc_id: str, base_seed: int) -> int:
    return (zlib.crc32(doc_id.encode("utf-8")) ^ (base_seed & 0xFFFFFFFF)) & 0xFFFFFFFF


def harvest_donor_crops(docs, max_per_source=300, min_w=12, min_h=8):
    """Collect PIL crops of field values, keyed by source, for splice donors."""
    pool: dict[str, list] = {}
    for doc in docs:
        src = doc["source"]
        bucket = pool.setdefault(src, [])
        if len(bucket) >= max_per_source:
            continue
        img = doc["image"]
        for b in doc["ocr_boxes"]:
            if b["label"] not in _DONOR_LABELS:
                continue
            x1, y1, x2, y2 = b["box_pixel"]
            if x2 - x1 < min_w or y2 - y1 < min_h:
                continue
            bucket.append(img.crop((x1, y1, x2, y2)))
            if len(bucket) >= max_per_source:
                break
    return pool


def build_ops(source: str, donor_pool: dict) -> dict:
    """Instantiate the four ops for a given source (splice gets that source's donors)."""
    return {
        "digit_swap": DigitSwapOp(),
        "copy_move": CopyMoveOp(),
        "splice": SpliceOp(donor_pool.get(source, [])),
        "recompress_ghost": RecompressGhostOp(),
    }


def _clean_record(doc):
    return {
        "image_path": None,
        "source": doc["source"], "doc_id": doc["doc_id"],
        "width": doc["width"], "height": doc["height"],
        "tampered": False, "field": None, "tamper_type": None,
        "box_pixel": None, "reason": "No inconsistencies detected.",
    }


def forge_one(doc, donor_pool, rng: random.Random, p_clean=0.5, op_names=None):
    """Return (image, record). Clean with prob p_clean, else a sampled tamper.
    Falls back through applicable ops if one fails; clean if none apply."""
    if rng.random() < p_clean:
        return doc["image"].convert("RGB").copy(), _clean_record(doc)

    ops = build_ops(doc["source"], donor_pool)
    order = list(op_names or OP_NAMES)
    rng.shuffle(order)
    order = [n for n in order if ops[n].applicable(doc)]

    for name in order:
        try:
            image, gt, meta = ops[name].apply(doc["image"], doc["ocr_boxes"], rng)
        except ValueError:
            continue
        rec = {
            "image_path": None,
            "source": doc["source"], "doc_id": doc["doc_id"],
            "width": doc["width"], "height": doc["height"],
            "tampered": True, "field": meta["field"], "tamper_type": name,
            "box_pixel": gt, "reason": meta["reason"],
        }
        return image, rec

    # nothing applied -> honest clean
    return doc["image"].convert("RGB").copy(), _clean_record(doc)


def forge_variants(doc, donor_pool, rng: random.Random, n_tampered=2):
    """Return [(image, record), ...] = 1 clean + up to n_tampered DISTINCT-op
    tampered variants for a single base doc (§13 step 5, Gemini multi-variant).

    All variants share the doc_id → the caller keeps them in the same split.
    Ops that fail to apply (e.g. no empty region) are skipped, not retried."""
    out = [(doc["image"].convert("RGB").copy(), _clean_record(doc))]

    ops = build_ops(doc["source"], donor_pool)
    order = [n for n in OP_NAMES if ops[n].applicable(doc)]
    rng.shuffle(order)

    for name in order:
        if len(out) - 1 >= n_tampered:
            break
        try:
            image, gt, meta = ops[name].apply(doc["image"], doc["ocr_boxes"], rng)
        except ValueError:
            continue
        rec = {
            "image_path": None,
            "source": doc["source"], "doc_id": doc["doc_id"],
            "width": doc["width"], "height": doc["height"],
            "tampered": True, "field": meta["field"], "tamper_type": name,
            "box_pixel": gt, "reason": meta["reason"],
        }
        out.append((image, rec))
    return out


def generate(docs, seed=0, p_clean=0.5):
    """Yield (image, record) for each source-doc. `docs` must be a list (reused to
    build the splice donor pool first)."""
    docs = list(docs)
    donor_pool = harvest_donor_crops(docs)
    for doc in docs:
        rng = random.Random(_doc_seed(doc["doc_id"], seed))
        yield forge_one(doc, donor_pool, rng, p_clean=p_clean)
