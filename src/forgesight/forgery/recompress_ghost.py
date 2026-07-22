"""recompress_ghost — tamper a region then JPEG-recompress the whole image at a
different quality, so the re-pasted patch carries double-compression (misaligned
8x8 DCT grid) artifacts the rest of the image lacks. GT box = the tampered patch.
A signal-level forgery: the model must catch it from texture, not text. §5 op 4.
"""

from __future__ import annotations

import io
import random

from PIL import Image

from . import base


def _jpeg_roundtrip(img, quality):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class RecompressGhostOp(base.ForgeryOp):
    name = "recompress_ghost"

    def applicable(self, record) -> bool:
        return len(record["ocr_boxes"]) >= 1

    def apply(self, image, ocr_boxes, rng: random.Random):
        image = image.convert("RGB").copy()
        w, h = image.size

        target = base.pick_box(ocr_boxes, rng, min_w=10, min_h=10)
        if target is None:
            raise ValueError("recompress_ghost: no region large enough")
        tb = target["box_pixel"]

        # 1) recompress ONLY the patch at high quality (its own DCT grid, origin 0,0)
        patch = image.crop(tuple(tb))
        patch_q = _jpeg_roundtrip(patch, quality=rng.choice([90, 95]))
        # shift the paste by 1-4px so the patch grid is misaligned vs the final save
        off = rng.randint(1, 4)
        px = min(tb[0] + off, w - patch_q.width)
        py = min(tb[1] + off, h - patch_q.height)
        image.paste(patch_q, (max(0, px), max(0, py)))

        # 2) recompress the WHOLE image at a lower quality -> double-compression ghost
        image = _jpeg_roundtrip(image, quality=rng.choice([70, 75, 80]))

        gt = base.clamp_box(tb, w, h)
        field = base.label_to_field(target["label"])
        reason = (f"The {field} region shows double-compression (JPEG ghosting) "
                  f"artifacts inconsistent with the rest of the image.")
        return image, gt, {"field": field, "reason": reason}
