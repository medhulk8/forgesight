"""splice — paste a crop taken from a DIFFERENT document (intra-source) over a
field region, creating font/lighting mismatch at the edges. GT box = the spliced
region. §5 op 3.

Donors are supplied at construction as a pool of PIL crops harvested from OTHER
same-source docs (the pipeline builds this). Cross-source donors are deliberately
avoided (Gemini review): the resolution/background clash would teach the model to
spot the dataset, not the splice.
"""

from __future__ import annotations

import random

from PIL import Image

from . import base


class SpliceOp(base.ForgeryOp):
    name = "splice"

    def __init__(self, donor_crops=None):
        # list of PIL.Image crops from other same-source documents
        self.donor_crops = list(donor_crops) if donor_crops else []

    def applicable(self, record) -> bool:
        return len(self.donor_crops) > 0 and len(record["ocr_boxes"]) >= 1

    def apply(self, image, ocr_boxes, rng: random.Random):
        image = image.convert("RGB").copy()
        w, h = image.size

        target = base.pick_box(ocr_boxes, rng)
        if target is None:
            raise ValueError("splice: no target field box")
        tb = target["box_pixel"]
        tw, th = base.box_wh(tb)

        donor = self.donor_crops[rng.randrange(len(self.donor_crops))].convert("RGB")
        # fit donor to the target field box (its own font/lighting comes with it)
        donor = donor.resize((max(4, tw), max(4, th)), Image.BILINEAR)
        image.paste(donor, (tb[0], tb[1]))

        gt = base.clamp_box(tb, w, h)
        field = base.label_to_field(target["label"])
        reason = (f"The {field} field was spliced from another document; font and "
                  f"lighting differ from the surrounding text at the region edges.")
        return image, gt, {"field": field, "reason": reason}
