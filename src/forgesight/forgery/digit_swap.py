"""digit_swap — alter a numeric field by pasting a digit region copied from
elsewhere in the SAME document (real texture/DPI, no rendered font). GT box = the
whole numeric line box. Most on-domain forgery for fraud (altered amounts). §5 op 1.
"""

from __future__ import annotations

import random

from PIL import Image

from . import base

# fields that read as amounts/dates/ids — preferred tamper targets
_PREFERRED = {"total", "line_total", "date", "answer"}


def _is_numeric(b):
    return b.get("is_numeric")


class DigitSwapOp(base.ForgeryOp):
    name = "digit_swap"

    def applicable(self, record) -> bool:
        # need at least two numeric boxes: one target + one donor.
        return sum(1 for b in record["ocr_boxes"] if _is_numeric(b)) >= 2

    def apply(self, image, ocr_boxes, rng: random.Random):
        image = image.convert("RGB").copy()
        w, h = image.size

        numeric = [b for b in ocr_boxes if _is_numeric(b)]
        # prefer amount/date fields as the visible target
        preferred = [b for b in numeric if b["label"] in _PREFERRED]
        target = base.pick_box(preferred or numeric, rng, _is_numeric)
        if target is None:
            raise ValueError("digit_swap: no numeric target")
        tb = target["box_pixel"]

        # donor = a DIFFERENT numeric box to source a digit patch from
        donors = [b for b in numeric if b is not target]
        donor = base.pick_box(donors, rng, _is_numeric)
        if donor is None:
            raise ValueError("digit_swap: no donor numeric box")
        db = donor["box_pixel"]

        line_h = tb[3] - tb[1]
        # a ~one-glyph-wide slice of the donor line
        glyph_w = max(6, min(line_h, db[2] - db[0]))
        dx = rng.randint(db[0], max(db[0], db[2] - glyph_w))
        patch = image.crop((dx, db[1], dx + glyph_w, db[3]))
        # match the target line height
        if patch.height != line_h and patch.height > 0:
            new_w = max(4, int(round(patch.width * line_h / patch.height)))
            patch = patch.resize((new_w, line_h), Image.BILINEAR)
        # subtle warp: tiny random rotation to avoid a pixel-perfect paste
        if rng.random() < 0.7:
            patch = patch.rotate(rng.uniform(-4, 4), expand=False,
                                 resample=Image.BILINEAR, fillcolor=None)

        # paste over a digit slot inside the target line (right half → amounts)
        max_px = max(tb[0], tb[2] - patch.width)
        px = rng.randint((tb[0] + tb[2]) // 2 if max_px > (tb[0] + tb[2]) // 2 else tb[0], max_px) \
            if max_px > tb[0] else tb[0]
        py = tb[1] + rng.randint(-1, 1)
        image.paste(patch, (px, py))

        gt = base.clamp_box(tb, w, h)
        field = base.label_to_field(target["label"])
        reason = (f"The {field} field appears altered: a digit was overwritten and its "
                  f"stroke weight and edges differ from the surrounding characters.")
        return image, gt, {"field": field, "reason": reason}
