"""Collator acceptance gate (§7.4) — runs on M3 CPU, no GPU.

Hand-made records: 2 tampered + 2 clean (varied lengths) + 1 deliberately
over-long. Asserts (the crux of the whole M3 phase):
  * input_ids / attention_mask / labels share a shape; pixel_values + image_grid_thw present.
  * process_vision_info runs EXACTLY N times for N records (single-pass, O(N) — monkeypatched).
  * every normal row has >=1 learnable label AND the decoded unmasked span EXACTLY
    equals the assistant target JSON (proves loss is on the answer only and the native
    <|box_start|>... tokens survive round-trip).
  * zero image-placeholder tokens are learnable; no label at a padding position is learnable.
  * the over-long row trips the truncation warning; normal rows emit none.

Exit non-zero if any assertion fails. Do NOT proceed to Kaggle until this is green.
"""

from __future__ import annotations

import os
import tempfile
import warnings

from PIL import Image, ImageDraw

import forgesight.collator as collmod
from forgesight import schema
from forgesight.collator import ForgeSightCollator
from forgesight.data import conversation

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def _make_image(path, w, h, seed):
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for k in range(6):
        x = (seed * 37 + k * 53) % max(1, w - 40)
        y = (seed * 29 + k * 41) % max(1, h - 20)
        d.rectangle([x, y, x + 38, y + 16], fill=(30 + k * 20, 40, 60))
    img.save(path, format="PNG")


def _records(img_dir):
    specs = [
        # 2 tampered, varied reason lengths
        dict(name="t0", w=420, h=640, tampered=True, field="total",
             tamper_type="digit_swap", box_pixel=[300, 500, 360, 528],
             reason="Digit altered."),
        dict(name="t1", w=512, h=720, tampered=True, field="date",
             tamper_type="splice", box_pixel=[40, 120, 190, 150],
             reason="The date field was spliced from another document; font and "
                    "lighting differ from the surrounding text at the region edges."),
        # 2 clean, varied
        dict(name="c0", w=400, h=600, tampered=False, field=None, tamper_type=None,
             box_pixel=None, reason="No inconsistencies detected."),
        dict(name="c1", w=600, h=480, tampered=False, field=None, tamper_type=None,
             box_pixel=None, reason="All fields consistent with a genuine document."),
        # 1 over-long: giant reason -> completion overflows max_length -> guard fires
        dict(name="big", w=460, h=680, tampered=True, field="total",
             tamper_type="copy_move", box_pixel=[100, 200, 260, 230],
             reason="tampering evidence " * 4000),
    ]
    recs = []
    for i, s in enumerate(specs):
        fname = f"{s['name']}.png"
        _make_image(os.path.join(img_dir, fname), s["w"], s["h"], seed=i + 1)
        recs.append({
            "image_path": fname, "width": s["w"], "height": s["h"],
            "tampered": s["tampered"], "field": s["field"],
            "tamper_type": s["tamper_type"], "box_pixel": s["box_pixel"],
            "reason": s["reason"],
        })
    return recs


def main():
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
    tok = processor.tokenizer

    img_dir = tempfile.mkdtemp(prefix="forgesight_smoke_")
    records = _records(img_dir)
    n = len(records)

    # monkeypatch process_vision_info to prove single-pass (O(N), not O(2N))
    orig_pvi = collmod.process_vision_info
    calls = {"n": 0}

    def counting_pvi(msgs):
        calls["n"] += 1
        return orig_pvi(msgs)

    collmod.process_vision_info = counting_pvi

    collator = ForgeSightCollator(
        processor,
        build_messages=lambda rec: conversation.build_messages(rec, data_root=img_dir),
        max_length=2048,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = collator(records)

    collmod.process_vision_info = orig_pvi  # restore

    input_ids = batch["input_ids"]
    attn = batch["attention_mask"]
    labels = batch["labels"]
    image_token_id = collator.image_token_id

    # --- shapes / keys ---
    assert input_ids.shape == attn.shape == labels.shape, "shape mismatch"
    assert "pixel_values" in batch and batch["pixel_values"].numel() > 0, "no pixel_values"
    assert "image_grid_thw" in batch, "no image_grid_thw"
    print(f"[ok] shapes {tuple(input_ids.shape)}; pixel_values {tuple(batch['pixel_values'].shape)}")

    # --- single-pass (fix 2) ---
    assert calls["n"] == n, f"process_vision_info ran {calls['n']}x, expected {n} (single-pass!)"
    print(f"[ok] single-pass: process_vision_info ran exactly {calls['n']}x for {n} records")

    # --- pad positions never learnable; image tokens never learnable ---
    assert (labels[attn == 0] == -100).all(), "learnable label at a padding position"
    assert (labels[input_ids == image_token_id] == -100).all(), "learnable image-placeholder token"
    print("[ok] no learnable label at pad positions; zero learnable image tokens")

    # --- per normal row: >=1 learnable AND decoded span == target JSON ---
    normal = [0, 1, 2, 3]
    for i in normal:
        mask = labels[i] != -100
        assert int(mask.sum()) > 0, f"row {i}: no learnable tokens"
        decoded = tok.decode(input_ids[i][mask])
        # completion includes the trailing <|im_end|>(+newline); strip it for the compare
        got = decoded.replace("<|im_end|>", "").strip()
        want = schema.to_target_json(records[i])
        assert got == want, f"row {i}: unmasked span != target\n got: {got!r}\n want: {want!r}"
    print("[ok] every normal row: decoded unmasked span EXACTLY equals target JSON "
          "(native box tokens survive)")

    # --- truncation guard: over-long warns, normal rows do not ---
    msgs = [str(w.message) for w in caught]
    over_long_warned = any(("sample 4" in m) or ("rows" in m and "4" in m)
                           or ("no learnable" in m) or ("truncated" in m and "4" in m)
                           for m in msgs)
    normal_warned = any(f"sample {i}" in m for i in normal for m in msgs)
    assert over_long_warned, f"over-long row did not trip the truncation guard; warnings={msgs}"
    assert not normal_warned, f"a normal row wrongly warned; warnings={msgs}"
    print(f"[ok] truncation guard fired on the over-long row only ({len(msgs)} warning(s))")

    print(f"\nSMOKE PASSED — collator masking verified on CPU (records={n}).")


if __name__ == "__main__":
    main()
