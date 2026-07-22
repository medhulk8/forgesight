"""Ingest base datasets → normalized source-doc records (§13 step 3, §5).

Loads SROIE + FUNSD and normalizes each into a common **source-doc** record
carrying the full OCR box list. This is NOT the §4 training record — it is the
input to the forgery pipeline (step 4/5), which picks a tamper site, applies an
op, and only then emits the §4 record (`{image_path,width,height,tampered,field,
tamper_type,box_pixel,reason}`). Keeping the two separate keeps `schema.validate_record`
about training targets only.

Source-doc record shape (dict):
    {
      "source":    "sroie" | "funsd",
      "doc_id":    str,              # stable id for doc-level splitting (step 5)
      "width":     int, "height": int,   # ORIGINAL pixel dims
      "image":     PIL.Image (RGB),
      "ocr_boxes": [ {"text": str,
                      "box_pixel": [x1,y1,x2,y2],   # axis-aligned, ORIGINAL pixels
                      "label": str,                 # semantic field / role
                      "is_numeric": bool} , ... ],
    }

Verified-live coordinate conventions (do NOT assume — checked against the real data):
  - SROIE (`arvindrajan92/sroie_document_understanding`): `ocr[i].box` is a 4-point
    quad in ORIGINAL PIXELS. Normalize quad → axis-aligned bbox (min/max of pts).
  - FUNSD (`nielsr/funsd`): `bboxes[i]` is already [x1,y1,x2,y2] but on the LayoutLM
    0..1000 NORMALIZED scale (bbox maxes exceed image width). Convert → pixels via
    coords.norm_to_pixel.

`darentang/sroie` (Gemini's suggestion) is a SCRIPT dataset → unusable on datasets>=4
(script loading removed). Switched to the parquet `arvindrajan92/...` mirror.
"""

from __future__ import annotations

import re

from .. import coords

FUNSD_ID = "nielsr/funsd"
SROIE_ID = "arvindrajan92/sroie_document_understanding"

_DIGIT_RE = re.compile(r"\d")


# --------------------------------------------------------------------------- #
# pure normalization helpers (network-free → unit-tested offline)
# --------------------------------------------------------------------------- #
def is_numeric(text: str) -> bool:
    """True if the token contains at least one digit (digit_swap candidate)."""
    return bool(_DIGIT_RE.search(text or ""))


def quad_to_bbox(quad) -> list[int]:
    """4-point quad [[x,y]*4] → axis-aligned [x1,y1,x2,y2] ints (SROIE)."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [int(round(min(xs))), int(round(min(ys))),
            int(round(max(xs))), int(round(max(ys)))]


def funsd_box_to_pixel(box, w: int, h: int) -> list[int]:
    """FUNSD 0..1000-normalized [x1,y1,x2,y2] → original-pixel bbox."""
    return coords.norm_to_pixel(box, w, h)


def strip_bio(tag_name: str) -> str:
    """'B-QUESTION'/'I-ANSWER'/'O' → 'question'/'answer'/'other' (FUNSD label)."""
    if not tag_name or tag_name == "O":
        return "other"
    core = tag_name.split("-", 1)[-1]
    return core.lower()


def _valid_bbox(b) -> bool:
    return b[2] > b[0] and b[3] > b[1]


# --------------------------------------------------------------------------- #
# per-source normalization (record-level, still network-free given a raw example)
# --------------------------------------------------------------------------- #
def normalize_sroie_example(ex, doc_id: str) -> dict:
    img = ex["image"].convert("RGB")
    w, h = img.size
    ocr_boxes = []
    for e in ex["ocr"]:
        bbox = quad_to_bbox(e["box"])
        if not _valid_bbox(bbox):
            continue
        text = e.get("text", "")
        ocr_boxes.append({
            "text": text,
            "box_pixel": bbox,
            "label": e.get("label", "other"),
            "is_numeric": is_numeric(text),
        })
    return {"source": "sroie", "doc_id": doc_id, "width": w, "height": h,
            "image": img, "ocr_boxes": ocr_boxes}


def normalize_funsd_example(ex, doc_id: str, tag_names) -> dict:
    img = ex["image"].convert("RGB")
    w, h = img.size
    ocr_boxes = []
    for word, box, tag in zip(ex["words"], ex["bboxes"], ex["ner_tags"]):
        bbox = funsd_box_to_pixel(box, w, h)
        if not _valid_bbox(bbox):
            continue
        label = strip_bio(tag_names[tag]) if tag_names else str(tag)
        ocr_boxes.append({
            "text": word,
            "box_pixel": bbox,
            "label": label,
            "is_numeric": is_numeric(word),
        })
    return {"source": "funsd", "doc_id": doc_id, "width": w, "height": h,
            "image": img, "ocr_boxes": ocr_boxes}


# --------------------------------------------------------------------------- #
# IO loaders
# --------------------------------------------------------------------------- #
def load_funsd(split: str = "train", limit: int | None = None):
    """Yield normalized FUNSD source-doc records."""
    from datasets import load_dataset

    ds = load_dataset(FUNSD_ID, split=split)
    tag_names = None
    feat = ds.features.get("ner_tags")
    if feat is not None and hasattr(feat, "feature") and hasattr(feat.feature, "names"):
        tag_names = feat.feature.names
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield normalize_funsd_example(ex, doc_id=f"funsd-{split}-{i}", tag_names=tag_names)


def load_sroie(split: str = "train", limit: int | None = None):
    """Yield normalized SROIE source-doc records."""
    from datasets import load_dataset

    ds = load_dataset(SROIE_ID, split=split)
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield normalize_sroie_example(ex, doc_id=f"sroie-{split}-{i}")


_LOADERS = {"funsd": load_funsd, "sroie": load_sroie}


def load_source_docs(sources=("sroie", "funsd"), split: str = "train",
                     limit_per_source: int | None = None):
    """Yield normalized source-doc records across the requested base datasets."""
    for src in sources:
        if src not in _LOADERS:
            raise ValueError(f"unknown source {src!r}; known: {sorted(_LOADERS)}")
        yield from _LOADERS[src](split=split, limit=limit_per_source)


if __name__ == "__main__":  # quick gate: print record stats
    import collections

    n = 0
    per_src = collections.Counter()
    per_label = collections.Counter()
    numeric = 0
    boxes = 0
    for doc in load_source_docs(limit_per_source=20):
        n += 1
        per_src[doc["source"]] += 1
        for b in doc["ocr_boxes"]:
            boxes += 1
            per_label[f"{doc['source']}:{b['label']}"] += 1
            numeric += b["is_numeric"]
    print(f"docs={n} per_source={dict(per_src)} total_boxes={boxes} numeric_boxes={numeric}")
    print("labels:", dict(per_label))
