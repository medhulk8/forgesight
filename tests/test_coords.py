"""Unit tests for coords.py (§4): pixel<->norm round-trip, IoU on known boxes."""

import pytest

from forgesight import coords


# --------------------------------------------------------------------------- #
# pixel <-> norm round-trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "box,w,h",
    [
        ([0, 0, 1654, 2339], 1654, 2339),      # full frame
        ([827, 1170, 1240, 1755], 1654, 2339),  # interior box
        ([100, 200, 300, 400], 1000, 1000),     # square, exact multiples
        ([0, 0, 500, 500], 1000, 1000),
    ],
)
def test_pixel_norm_roundtrip(box, w, h):
    norm = coords.pixel_to_norm(box, w, h)
    assert all(0 <= v <= coords.NORM_MAX for v in norm)
    back = coords.norm_to_pixel(norm, w, h)
    # round-trip through the integer 0..1000 grid loses at most ~1 pixel per 1/1000.
    tol_x = w / coords.NORM_MAX + 1
    tol_y = h / coords.NORM_MAX + 1
    assert abs(back[0] - box[0]) <= tol_x
    assert abs(back[2] - box[2]) <= tol_x
    assert abs(back[1] - box[1]) <= tol_y
    assert abs(back[3] - box[3]) <= tol_y


def test_pixel_to_norm_exact():
    # 1000-wide image => pixel value maps 1:1 to the 0..1000 scale.
    assert coords.pixel_to_norm([250, 500, 750, 900], 1000, 1000) == [250, 500, 750, 900]


def test_norm_to_pixel_exact():
    assert coords.norm_to_pixel([250, 500, 750, 900], 1000, 1000) == [250, 500, 750, 900]


def test_pixel_to_norm_clamps():
    # out-of-frame coords clamp into [0, 1000], never negative or >1000.
    norm = coords.pixel_to_norm([-50, -10, 2000, 3000], 1654, 2339)
    assert norm[0] == 0 and norm[1] == 0
    assert norm[2] == coords.NORM_MAX and norm[3] == coords.NORM_MAX


def test_bad_dims_raise():
    with pytest.raises(ValueError):
        coords.pixel_to_norm([0, 0, 10, 10], 0, 100)
    with pytest.raises(ValueError):
        coords.norm_to_pixel([0, 0, 10, 10], 100, 0)


# --------------------------------------------------------------------------- #
# IoU on known boxes
# --------------------------------------------------------------------------- #
def test_iou_identical():
    assert coords.iou([10, 10, 50, 50], [10, 10, 50, 50]) == pytest.approx(1.0)


def test_iou_zero_overlap():
    assert coords.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_touching_edges_is_zero():
    # boxes sharing only an edge have zero intersection area.
    assert coords.iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0


def test_iou_half_overlap():
    # A=[0,0,10,10] area 100; B=[5,0,15,10] area 100; inter=[5,0,10,10]=50;
    # union=150 => 1/3.
    assert coords.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_iou_contained():
    # inner fully inside outer: inter = inner area = 100; union = outer = 400.
    assert coords.iou([0, 0, 20, 20], [5, 5, 15, 15]) == pytest.approx(100 / 400)


def test_iou_degenerate_zero_area():
    assert coords.iou([10, 10, 10, 50], [10, 10, 50, 50]) == 0.0


def test_iou_unsorted_corners():
    # corners given in reversed order still yield a correct IoU (self-normalized).
    assert coords.iou([50, 50, 10, 10], [10, 10, 50, 50]) == pytest.approx(1.0)
