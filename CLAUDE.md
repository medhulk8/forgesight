# ForgeSight — quick reference

QLoRA fine-tune of **Qwen2-VL-2B-Instruct** for document **tamper detection + spatial grounding**: image → JSON `{tampered, field, box, reason}`, box in Qwen2-VL native 0–1000 grounding-token format.

## Stack + responsibility split
- **Python 3** (`src/forgesight/`, src-layout, `pip install -e .`) — all logic, importable + unit-testable.
- **PyTorch + HF transformers 4.47.1** — model/processor (Qwen2VL).
- **TRL 0.12.1 SFTTrainer + PEFT 0.13.2 LoRA + bitsandbytes 0.44.1 (4-bit NF4)** — QLoRA training (Kaggle only).
- **datasets / Pillow / OpenCV / numpy** — synthetic forgery data pipeline.
- **scipy** — eval stats (Wilson CI, McNemar).
- **Dev split (D10):** M3 (MPS/CPU) = data-gen + collator + shape/unit tests, NO GPU/bitsandbytes. Kaggle 2×T4 = all real QLoRA runs.

## The one hard architectural rule
Everything under `src/forgesight/` imports + unit-tests on M3 with no GPU and no bitsandbytes. GPU-only concerns (4-bit load, training) live behind `model.py` / `train_*.py`, exercised only on Kaggle. Boxes use native `<|box_start|>` grounding tokens (D2/D3) — never raw-int arrays. Collator is single-pass, token-search masking (§7.2/7.3) — never double-render.

## Current stage
Stages §13 1–7 **done — M3 phase complete.** Step 7: `collator.py` (single-pass token-search masking §7.2/7.3) + `scripts/smoke_collator.py` (§7.4). **SMOKE PASSED on CPU**: shapes ok, single-pass (process_vision_info N× not 2N, monkeypatched), decoded unmasked span EXACTLY == target JSON (native box tokens survive), zero learnable pad/image tokens, truncation guard fires on over-long only. 108 unit tests green.
**Env change:** dev venv rebuilt on **Python 3.11** (via `uv`, torch 2.13 wheels; aligns with Kaggle) — 3.14 had no torch. Verified Qwen2-VL token ids: im_start=151644 im_end=151645 image_pad=151655 box_start/end=151648/9; `assistant\n`=[77091,198] (n_tail=2); assistant turn ends `<|im_end|>\n` so guard checks im_end *within completion span*, not last token.
Next: §13 step 8 — push to GitHub (done incrementally) + upload `data/processed/` as private Kaggle Dataset; Kaggle notebook clones repo + imports forgesight. **First GPU/Kaggle step.**
Dataset in `data/processed/` (gitignored): 2964 ex (train 2244 / val 360 / test 360), doc-split 561/120/120, train 50/50, 0 leaks.
Build design (locked): 1 clean + up to 2 distinct-op tampered per doc; all variants same split; balance 50/50 TRAIN ONLY via clean-oversampling (val/test untouched — eval integrity); donors per (split,source); all images PNG (ghost artifact baked into pixels).
Op design (locked): digit_swap GT = whole numeric line box, alteration = intra-doc digit copy (no TTF); splice donors intra-source only; recompress_ghost GT = tampered patch. Reasons = parameterized templates.
Coord gotchas (locked): SROIE quads = pixels; FUNSD bboxes = 0–1000 normalized (convert via `norm_to_pixel`). Ingest emits a **source-doc** record (`ocr_boxes`); pipeline emits §4-shape record with `image_path=None` (build_dataset fills + persists at step 5).

## Pointers
Full plan → [plan.md](plan.md). Session history → [SESSIONS.md](SESSIONS.md).

## Standing rules (permanent — survive all sessions)
1. **Self-update docs, unprompted.** After every significant step (stage done, arch decision, dep added, bug fixed, PLAN deviation): update CLAUDE.md if concise facts changed + append to SESSIONS.md. Automatic.
2. **Keep Gemini in the loop, proactively.** Before anything significant, emit a `--- GEMINI PROMPT ---` block: (a) situation, (b) next plan, (c) doubts. Terse. Don't re-explain the project (Gemini has context + memory). Generate these myself.
3. **Gemini is advisory.** Sanity-check every suggestion vs ground truth (live API, docs, code) before acting. Push back + document why in SESSIONS.md if it doesn't hold.
4. **Stage discipline.** PLAN stages in order. Each stage ends runnable + committed. One commit/stage + incremental commits for significant steps.
5. **Verify vs reality.** Check live (real runs/tests/screenshots) over memory/guesses. One-line memory for hard-won facts.
6. **Commits:** private repo, no AI-attribution lines. Gitignore generated/local artifacts. `.env.example` tracked, real `.env` never. Internal docs (CLAUDE/SESSIONS/plan) tracked while building; untrack if repo goes public.
