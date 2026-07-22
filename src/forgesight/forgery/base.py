"""ForgeryOp interface + shared geometry helpers (§5, §13 step 4).

Each op takes a source-doc image + its OCR boxes (from `data.ingest`) and returns
a tampered image, the ground-truth tamper box in ORIGINAL pixels, and meta
(field name + templated reason). The pipeline (`pipeline.py`) turns that into a
§4 training record.

    op.apply(image, ocr_boxes, rng) -> (tampered_image: PIL.Image,
                                        gt_box_pixel: [x1,y1,x2,y2],
                                        meta: {"field": str, "reason": str})

Design choices (Gemini-reviewed, step 4):
  - digit_swap GT box = the whole numeric LINE box (no OpenCV glyph extraction).
  - digit alteration = copy a digit region from ELSEWHERE in the same doc (real
    texture/DPI), never a rendered TTF glyph (a VLM would trivially spot clean
    antialiasing on a noisy scan).
  - splice donors are INTRA-SOURCE only (cross-source clashes teach dataset-id
    detection, not splice-edge detection).
  - recompress_ghost GT box = the tampered patch (double-compression ghosting is
    local to the re-pasted patch even though the final save is global).
"""

from __future__ import annotations

import random


class ForgeryOp:
    name: str = "base"

    def applicable(self, record) -> bool:
        """True if this op can act on the given source-doc record."""
        raise NotImplementedError

    def apply(self, image, ocr_boxes, rng: random.Random):
        """Return (tampered_image, gt_box_pixel, meta{field, reason})."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def clamp_box(box, w: int, h: int):
    """Clamp [x1,y1,x2,y2] to image bounds, ints."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), w))
    y1 = max(0, min(int(round(y1)), h))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    return [x1, y1, x2, y2]


def box_wh(box):
    return box[2] - box[0], box[3] - box[1]


def boxes_overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def label_to_field(label: str) -> str:
    """Map an OCR label to the §4 `field` name. Labels are already meaningful
    for SROIE (total/date/...); FUNSD roles (question/answer) pass through."""
    return label or "field"


def pick_box(ocr_boxes, rng, predicate=None, min_w=6, min_h=6):
    """Pick a random OCR box satisfying predicate and a minimum size, else None."""
    cands = [
        b for b in ocr_boxes
        if (predicate is None or predicate(b))
        and (b["box_pixel"][2] - b["box_pixel"][0]) >= min_w
        and (b["box_pixel"][3] - b["box_pixel"][1]) >= min_h
    ]
    if not cands:
        return None
    return cands[rng.randrange(len(cands))]


def sample_empty_box(ocr_boxes, w, h, patch_w, patch_h, rng, tries=60, margin=4):
    """Find a [x1,y1,x2,y2] of size (patch_w, patch_h) that does not overlap any
    OCR box (so copy_move pastes into whitespace, not over existing text). None if
    no free spot found within `tries`."""
    if patch_w >= w or patch_h >= h:
        return None
    occ = [b["box_pixel"] for b in ocr_boxes]
    for _ in range(tries):
        x1 = rng.randint(margin, max(margin, w - patch_w - margin))
        y1 = rng.randint(margin, max(margin, h - patch_h - margin))
        cand = [x1, y1, x1 + patch_w, y1 + patch_h]
        pad = [cand[0] - margin, cand[1] - margin, cand[2] + margin, cand[3] + margin]
        if not any(boxes_overlap(pad, o) for o in occ):
            return cand
    return None
