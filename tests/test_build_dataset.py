"""Offline tests for build_dataset pure helpers + pipeline.forge_variants."""

import random

from PIL import Image, ImageDraw

from forgesight.data import build_dataset as bd
from forgesight.forgery import pipeline

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _synthetic_doc(doc_id, source="sroie"):
    w, h = 400, 600
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    ocr = [
        {"text": "COMPANY", "box_pixel": [40, 30, 300, 60], "label": "company"},
        {"text": "123.45", "box_pixel": [220, 120, 320, 150], "label": "total"},
        {"text": "2018-12-25", "box_pixel": [40, 120, 190, 150], "label": "date"},
    ]
    for i, b in enumerate(ocr):
        d.rectangle(b["box_pixel"], fill=(20 + i * 30, 20, 20))
        b["is_numeric"] = any(c.isdigit() for c in b["text"])
    return {"source": source, "doc_id": doc_id, "width": w, "height": h,
            "image": img, "ocr_boxes": ocr}


# --------------------------------------------------------------------------- #
# assign_splits
# --------------------------------------------------------------------------- #
def test_assign_splits_partitions_all_and_is_deterministic():
    ids = [f"d{i}" for i in range(100)]
    a = bd.assign_splits(ids, RATIOS, seed=0)
    b = bd.assign_splits(ids, RATIOS, seed=0)
    assert a == b                                  # deterministic
    assert set(a) == set(ids)                      # every doc placed
    counts = {"train": 0, "val": 0, "test": 0}
    for v in a.values():
        counts[v] += 1
    assert counts["train"] == 70 and counts["val"] == 15 and counts["test"] == 15


def test_assign_splits_no_doc_in_two_splits():
    ids = [f"d{i}" for i in range(50)]
    mapping = bd.assign_splits(ids, RATIOS, seed=3)
    # each id maps to exactly one split by construction
    assert all(mapping[i] in ("train", "val", "test") for i in ids)


# --------------------------------------------------------------------------- #
# balance_5050
# --------------------------------------------------------------------------- #
def test_balance_5050_oversamples_minority():
    recs = ([{"tampered": False, "doc_id": f"c{i}"} for i in range(3)]
            + [{"tampered": True, "doc_id": f"t{i}"} for i in range(9)])
    out = bd.balance_5050(recs, seed=0)
    clean = sum(1 for r in out if not r["tampered"])
    tamp = sum(1 for r in out if r["tampered"])
    assert clean == tamp == 9                       # minority oversampled to match
    assert len(out) == 18


def test_balance_5050_noop_when_single_class():
    recs = [{"tampered": True} for _ in range(4)]
    assert len(bd.balance_5050(recs, seed=0)) == 4


# --------------------------------------------------------------------------- #
# forge_variants
# --------------------------------------------------------------------------- #
def test_forge_variants_one_clean_plus_distinct_tampered():
    doc = _synthetic_doc("sroie-train-0")
    pool = pipeline.harvest_donor_crops([doc, _synthetic_doc("sroie-train-1")])
    variants = pipeline.forge_variants(doc, pool, random.Random(0), n_tampered=2)
    recs = [r for _, r in variants]
    assert recs[0]["tampered"] is False            # first is always clean
    tamp = [r for r in recs if r["tampered"]]
    assert 1 <= len(tamp) <= 2
    # distinct ops
    assert len({r["tamper_type"] for r in tamp}) == len(tamp)
    # all variants share the doc_id (→ same split downstream)
    assert {r["doc_id"] for r in recs} == {"sroie-train-0"}


def test_forge_variants_deterministic():
    doc = _synthetic_doc("sroie-train-0")
    pool = pipeline.harvest_donor_crops([doc])
    v1 = [r for _, r in pipeline.forge_variants(doc, pool, random.Random(1), 2)]
    v2 = [r for _, r in pipeline.forge_variants(doc, pool, random.Random(1), 2)]
    assert v1 == v2


# --------------------------------------------------------------------------- #
# leakage assertion
# --------------------------------------------------------------------------- #
def test_assert_no_leakage_passes_when_clean():
    sr = {"train": [{"doc_id": "a"}, {"doc_id": "a"}], "test": [{"doc_id": "b"}]}
    bd._assert_no_leakage(sr)  # no raise


def test_assert_no_leakage_raises_on_straddle():
    sr = {"train": [{"doc_id": "a"}], "test": [{"doc_id": "a"}]}
    try:
        bd._assert_no_leakage(sr)
    except AssertionError as e:
        assert "LEAKAGE" in str(e)
    else:
        raise AssertionError("expected leakage assertion to fire")


# --------------------------------------------------------------------------- #
# compute_manifest
# --------------------------------------------------------------------------- #
def test_compute_manifest_counts():
    sr = {
        "train": [
            {"tampered": False, "tamper_type": None, "source": "sroie", "doc_id": "a"},
            {"tampered": True, "tamper_type": "digit_swap", "source": "sroie", "doc_id": "a"},
            {"tampered": True, "tamper_type": "splice", "source": "funsd", "doc_id": "b"},
        ],
        "test": [
            {"tampered": False, "tamper_type": None, "source": "funsd", "doc_id": "c"},
        ],
    }
    m = bd.compute_manifest(sr)
    assert m["splits"]["train"]["examples"] == 3
    assert m["splits"]["train"]["clean"] == 1 and m["splits"]["train"]["tampered"] == 2
    assert m["splits"]["train"]["docs"] == 2
    assert m["splits"]["train"]["by_tamper_type"] == {"digit_swap": 1, "splice": 1}
    assert m["totals"]["examples"] == 4
