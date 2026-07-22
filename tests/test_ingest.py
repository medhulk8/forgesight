"""Offline unit tests for data/ingest.py pure normalizers (no network).

Live dataset loading is exercised by scripts/check_ingest (the step-3 visual gate),
not here — the test suite stays network-free per the M3 dev rule.
"""

import pytest

from forgesight.data import ingest


# --------------------------------------------------------------------------- #
# is_numeric
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("$45.99", True), ("2018-12-25", True), ("789417-W", True),
    ("TAN WOON YANN", False), ("TOTAL", False), ("", False), (None, False),
])
def test_is_numeric(text, expected):
    assert ingest.is_numeric(text) is expected


# --------------------------------------------------------------------------- #
# quad_to_bbox (SROIE)
# --------------------------------------------------------------------------- #
def test_quad_to_bbox_axis_aligned():
    quad = [[72.0, 25.0], [326.0, 25.0], [326.0, 64.0], [72.0, 64.0]]
    assert ingest.quad_to_bbox(quad) == [72, 25, 326, 64]


def test_quad_to_bbox_skewed_takes_extremes():
    # slightly rotated quad → tight axis-aligned envelope via min/max.
    quad = [[10, 12], [50, 8], [52, 40], [8, 44]]
    assert ingest.quad_to_bbox(quad) == [8, 8, 52, 44]


# --------------------------------------------------------------------------- #
# funsd_box_to_pixel (0..1000 normalized → pixels)
# --------------------------------------------------------------------------- #
def test_funsd_box_to_pixel_scales_down():
    # x2=841 on a 762-wide image is only sane if boxes are 0..1000 normalized:
    # 841/1000*762 ≈ 641. Confirms the normalized-not-pixel convention.
    box = [383, 91, 841, 926]
    px = ingest.funsd_box_to_pixel(box, 762, 1000)
    assert px == [292, 91, 641, 926]
    assert px[2] < 762  # lands inside the image, unlike the raw normalized value


# --------------------------------------------------------------------------- #
# strip_bio (FUNSD labels)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tag,expected", [
    ("O", "other"), ("B-HEADER", "header"), ("I-HEADER", "header"),
    ("B-QUESTION", "question"), ("I-ANSWER", "answer"), ("", "other"), (None, "other"),
])
def test_strip_bio(tag, expected):
    assert ingest.strip_bio(tag) == expected


# --------------------------------------------------------------------------- #
# record normalization with fake raw examples (no dataset download)
# --------------------------------------------------------------------------- #
class _FakeImg:
    def __init__(self, size):
        self.size = size

    def convert(self, mode):
        return self  # size preserved; mode irrelevant for the box math


def test_normalize_sroie_example():
    ex = {
        "image": _FakeImg((463, 1013)),
        "ocr": [
            {"box": [[50, 82], [440, 82], [440, 121], [50, 121]],
             "label": "company", "text": "BOOK TA"},
            {"box": [[0, 0], [0, 0], [0, 0], [0, 0]],  # degenerate → dropped
             "label": "other", "text": "x"},
            {"box": [[10, 10], [30, 10], [30, 20], [10, 20]],
             "label": "total", "text": "9.00"},
        ],
    }
    doc = ingest.normalize_sroie_example(ex, "sroie-train-0")
    assert doc["source"] == "sroie" and doc["width"] == 463 and doc["height"] == 1013
    assert len(doc["ocr_boxes"]) == 2  # degenerate dropped
    assert doc["ocr_boxes"][0]["box_pixel"] == [50, 82, 440, 121]
    assert doc["ocr_boxes"][0]["is_numeric"] is False
    assert doc["ocr_boxes"][1]["label"] == "total"
    assert doc["ocr_boxes"][1]["is_numeric"] is True


def test_normalize_funsd_example():
    ex = {
        "image": _FakeImg((762, 1000)),
        "words": ["R&D", "597005708"],
        "bboxes": [[383, 91, 493, 175], [500, 200, 600, 250]],  # 0..1000 normalized
        "ner_tags": [1, 5],
    }
    tag_names = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION",
                 "B-ANSWER", "I-ANSWER"]
    doc = ingest.normalize_funsd_example(ex, "funsd-train-0", tag_names)
    assert doc["source"] == "funsd"
    b0 = doc["ocr_boxes"][0]
    assert b0["label"] == "header"
    assert b0["box_pixel"] == ingest.funsd_box_to_pixel([383, 91, 493, 175], 762, 1000)
    assert doc["ocr_boxes"][1]["label"] == "answer"
    assert doc["ocr_boxes"][1]["is_numeric"] is True


def test_load_source_docs_unknown_source_raises():
    with pytest.raises(ValueError):
        list(ingest.load_source_docs(sources=("bogus",)))
