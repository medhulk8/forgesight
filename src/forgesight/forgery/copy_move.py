"""copy_move — duplicate a field-value patch from one region into empty
whitespace elsewhere in the same document, with a slight blend. GT box = the
paste location. §5 op 2.
"""

from __future__ import annotations

import random

from PIL import Image

from . import base


class CopyMoveOp(base.ForgeryOp):
    name = "copy_move"

    def applicable(self, record) -> bool:
        return len(record["ocr_boxes"]) >= 1

    def apply(self, image, ocr_boxes, rng: random.Random):
        image = image.convert("RGB").copy()
        w, h = image.size

        src = base.pick_box(ocr_boxes, rng)
        if src is None:
            raise ValueError("copy_move: no source box")
        sb = src["box_pixel"]
        patch = image.crop(tuple(sb))
        pw, ph = patch.size

        dest = base.sample_empty_box(ocr_boxes, w, h, pw, ph, rng)
        if dest is None:
            raise ValueError("copy_move: no empty destination region")

        # slight blend so the paste is not a hard-edged clone
        if rng.random() < 0.6:
            patch = Image.blend(
                patch, image.crop((dest[0], dest[1], dest[0] + pw, dest[1] + ph)),
                alpha=rng.uniform(0.1, 0.25))
        image.paste(patch, (dest[0], dest[1]))

        gt = base.clamp_box(dest, w, h)
        field = base.label_to_field(src["label"])
        reason = (f"The {field} region shows copy-move tampering; a duplicated patch "
                  f"with matching texture was pasted into otherwise empty space.")
        return image, gt, {"field": field, "reason": reason}
