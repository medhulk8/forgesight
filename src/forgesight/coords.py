"""Coordinate conversions and IoU (§4).

Two box spaces exist in ForgeSight:
  - pixel space:      original image pixels, [x1, y1, x2, y2].
  - normalized space: Qwen2-VL's 0..1000 integer grounding scale, [x1, y1, x2, y2].

Rule (§4): IoU is always computed in a single consistent space. At eval we convert
a predicted 0..1000 box back to pixels using the known test-image dims, then IoU in
pixel space.
"""

from __future__ import annotations

NORM_MAX = 1000


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def pixel_to_norm(box, w: int, h: int) -> list[int]:
    """Pixel [x1,y1,x2,y2] -> 0..1000 int box, clamped to the valid range.

    x scales by width, y by height. Result is rounded to ints (the native scale
    is integer). Raises on non-positive dims.
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"pixel_to_norm: non-positive dims w={w} h={h}")
    x1, y1, x2, y2 = box
    return [
        int(round(_clamp(x1 / w * NORM_MAX, 0, NORM_MAX))),
        int(round(_clamp(y1 / h * NORM_MAX, 0, NORM_MAX))),
        int(round(_clamp(x2 / w * NORM_MAX, 0, NORM_MAX))),
        int(round(_clamp(y2 / h * NORM_MAX, 0, NORM_MAX))),
    ]


def norm_to_pixel(box, w: int, h: int) -> list[int]:
    """0..1000 int box -> pixel [x1,y1,x2,y2] for an image of size (w, h)."""
    if w <= 0 or h <= 0:
        raise ValueError(f"norm_to_pixel: non-positive dims w={w} h={h}")
    x1, y1, x2, y2 = box
    return [
        int(round(_clamp(x1 / NORM_MAX * w, 0, w))),
        int(round(_clamp(y1 / NORM_MAX * h, 0, h))),
        int(round(_clamp(x2 / NORM_MAX * w, 0, w))),
        int(round(_clamp(y2 / NORM_MAX * h, 0, h))),
    ]


def iou(box_a, box_b) -> float:
    """Intersection-over-union of two [x1,y1,x2,y2] boxes in a shared space.

    Returns 0.0 for non-overlapping or degenerate (zero-area) boxes, 1.0 for
    identical non-degenerate boxes.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # normalize corner order so callers need not pre-sort
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union
