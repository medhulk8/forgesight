"""Offline unit tests for forgery ops + pipeline (synthetic images, no network)."""

import random

import pytest
from PIL import Image, ImageDraw

from forgesight.forgery import base, pipeline
from forgesight.forgery.copy_move import CopyMoveOp
from forgesight.forgery.digit_swap import DigitSwapOp
from forgesight.forgery.recompress_ghost import RecompressGhostOp
from forgesight.forgery.splice import SpliceOp


def _synthetic_doc(doc_id="sroie-train-0", source="sroie"):
    """White page with a few filled 'text' boxes; two numeric, spaced out."""
    w, h = 400, 600
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    ocr = [
        {"text": "COMPANY LTD", "box_pixel": [40, 30, 300, 60], "label": "company"},
        {"text": "123.45", "box_pixel": [220, 120, 320, 150], "label": "total"},
        {"text": "2018-12-25", "box_pixel": [40, 120, 190, 150], "label": "date"},
        {"text": "ADDRESS ROAD", "box_pixel": [40, 200, 260, 230], "label": "address"},
    ]
    # draw distinct content so pastes actually change pixels
    for i, b in enumerate(ocr):
        x1, y1, x2, y2 = b["box_pixel"]
        d.rectangle([x1, y1, x2, y2], fill=(20 + i * 30, 20, 20))
        d.text((x1 + 2, y1 + 2), b["text"], fill="white")
        b["is_numeric"] = any(c.isdigit() for c in b["text"])
    return {"source": source, "doc_id": doc_id, "width": w, "height": h,
            "image": img, "ocr_boxes": ocr}


def _valid_gt(gt, w, h):
    return (0 <= gt[0] < gt[2] <= w) and (0 <= gt[1] < gt[3] <= h)


def _changed(a, b):
    return a.tobytes() != b.tobytes()


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def test_clamp_box():
    assert base.clamp_box([-5, -5, 500, 700], 400, 600) == [0, 0, 400, 600]


def test_boxes_overlap():
    assert base.boxes_overlap([0, 0, 10, 10], [5, 5, 15, 15])
    assert not base.boxes_overlap([0, 0, 10, 10], [10, 0, 20, 10])  # edge touch


def test_sample_empty_box_finds_gap():
    doc = _synthetic_doc()
    rng = random.Random(1)
    box = base.sample_empty_box(doc["ocr_boxes"], doc["width"], doc["height"],
                                40, 20, rng)
    assert box is not None
    assert not any(base.boxes_overlap(box, b["box_pixel"]) for b in doc["ocr_boxes"])


def test_sample_empty_box_none_when_full():
    # one box covering the whole page -> no free spot
    ocr = [{"text": "x", "box_pixel": [0, 0, 100, 100], "label": "o", "is_numeric": False}]
    assert base.sample_empty_box(ocr, 100, 100, 50, 50, random.Random(0)) is None


# --------------------------------------------------------------------------- #
# digit_swap
# --------------------------------------------------------------------------- #
def test_digit_swap_applicable():
    assert DigitSwapOp().applicable(_synthetic_doc())
    one_numeric = _synthetic_doc()
    one_numeric["ocr_boxes"] = one_numeric["ocr_boxes"][:2]  # company + one numeric
    assert not DigitSwapOp().applicable(one_numeric)


def test_digit_swap_apply():
    doc = _synthetic_doc()
    orig = doc["image"].copy()
    img, gt, meta = DigitSwapOp().apply(doc["image"], doc["ocr_boxes"], random.Random(3))
    assert img.size == (doc["width"], doc["height"])
    assert _valid_gt(gt, doc["width"], doc["height"])
    # GT must equal one of the numeric line boxes (line-level, per design)
    numeric_boxes = [b["box_pixel"] for b in doc["ocr_boxes"] if b["is_numeric"]]
    assert gt in [base.clamp_box(b, doc["width"], doc["height"]) for b in numeric_boxes]
    assert meta["field"] in {"total", "date"} and "altered" in meta["reason"]
    assert _changed(orig, img)


# --------------------------------------------------------------------------- #
# copy_move
# --------------------------------------------------------------------------- #
def test_copy_move_apply():
    doc = _synthetic_doc()
    orig = doc["image"].copy()
    img, gt, meta = CopyMoveOp().apply(doc["image"], doc["ocr_boxes"], random.Random(5))
    assert img.size == (doc["width"], doc["height"])
    assert _valid_gt(gt, doc["width"], doc["height"])
    # paste target is whitespace -> gt should not overlap existing text boxes
    assert not any(base.boxes_overlap(gt, b["box_pixel"]) for b in doc["ocr_boxes"])
    assert "copy-move" in meta["reason"]
    assert _changed(orig, img)


# --------------------------------------------------------------------------- #
# splice
# --------------------------------------------------------------------------- #
def test_splice_applicable_requires_donors():
    doc = _synthetic_doc()
    assert not SpliceOp([]).applicable(doc)
    assert SpliceOp([Image.new("RGB", (30, 20), "black")]).applicable(doc)


def test_splice_apply():
    doc = _synthetic_doc()
    orig = doc["image"].copy()
    donor = Image.new("RGB", (60, 25), (0, 200, 0))
    img, gt, meta = SpliceOp([donor]).apply(doc["image"], doc["ocr_boxes"], random.Random(7))
    assert img.size == (doc["width"], doc["height"])
    assert _valid_gt(gt, doc["width"], doc["height"])
    # GT equals a field box (spliced over it)
    assert gt in [base.clamp_box(b["box_pixel"], doc["width"], doc["height"])
                  for b in doc["ocr_boxes"]]
    assert "spliced" in meta["reason"]
    assert _changed(orig, img)


# --------------------------------------------------------------------------- #
# recompress_ghost
# --------------------------------------------------------------------------- #
def test_recompress_ghost_apply():
    doc = _synthetic_doc()
    img, gt, meta = RecompressGhostOp().apply(doc["image"], doc["ocr_boxes"],
                                              random.Random(9))
    assert img.size == (doc["width"], doc["height"])
    assert _valid_gt(gt, doc["width"], doc["height"])
    assert "ghosting" in meta["reason"] or "double-compression" in meta["reason"]


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def test_forge_one_clean_when_p_clean_1():
    doc = _synthetic_doc()
    img, rec = pipeline.forge_one(doc, {}, random.Random(0), p_clean=1.0)
    assert rec["tampered"] is False
    assert rec["field"] is None and rec["box_pixel"] is None and rec["tamper_type"] is None


def test_forge_one_tampered_when_p_clean_0():
    doc = _synthetic_doc()
    pool = pipeline.harvest_donor_crops([doc, _synthetic_doc("sroie-train-1")])
    img, rec = pipeline.forge_one(doc, pool, random.Random(0), p_clean=0.0)
    assert rec["tampered"] is True
    assert rec["tamper_type"] in pipeline.OP_NAMES
    assert rec["box_pixel"] is not None and _valid_gt(rec["box_pixel"], 400, 600)
    assert rec["field"] and rec["reason"]


def test_forge_one_deterministic():
    doc = _synthetic_doc()
    pool = pipeline.harvest_donor_crops([doc])
    i1, r1 = pipeline.forge_one(doc, pool, random.Random(42), p_clean=0.0)
    i2, r2 = pipeline.forge_one(doc, pool, random.Random(42), p_clean=0.0)
    assert r1 == r2
    assert i1.tobytes() == i2.tobytes()


def test_generate_yields_records_and_is_stable():
    docs = [_synthetic_doc(f"sroie-train-{i}") for i in range(6)]
    out1 = [rec for _, rec in pipeline.generate(docs, seed=1, p_clean=0.5)]
    out2 = [rec for _, rec in pipeline.generate(docs, seed=1, p_clean=0.5)]
    assert out1 == out2                       # deterministic per seed
    assert len(out1) == 6
    for rec in out1:
        assert set(["tampered", "field", "tamper_type", "box_pixel", "reason",
                    "width", "height", "doc_id", "source"]).issubset(rec)
