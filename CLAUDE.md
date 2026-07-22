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
Stages §13 1–4 **done**. Step 4: `forgery/` (base ForgeryOp + digit_swap/copy_move/splice/recompress_ghost + pipeline.py) + `scripts/viz_sample.py`. GT boxes land correctly on all 4 ops × both sources (visually verified). `pipeline.generate()` → ~48% clean, uniform op spread. 94 tests green. **Paused for human/Gemini eyeball before step 5 (mass dataset gen).**
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
