# ForgeSight — session log

_Newest entry at top._

---

## 2026-07-22 — Stage §13 step 4 (`forgery/` ops + pipeline)

**Stage:** §13 step 4 — synthetic forgery ops + orchestration.

**Implemented:**
- `forgery/base.py`: `ForgeryOp` interface (`applicable` / `apply(image, ocr_boxes, rng) -> (image, gt_box_pixel, meta{field,reason})`) + geometry helpers (`clamp_box`, `boxes_overlap`, `sample_empty_box`, `pick_box`, `label_to_field`).
- Four ops (D-order): `digit_swap` (intra-doc digit copy over numeric line; GT = whole line box), `copy_move` (field patch → whitespace via `sample_empty_box`; GT = paste loc), `splice` (intra-source donor crop over a field; GT = field box), `recompress_ghost` (patch JPEG-recompressed + offset, whole image re-saved lower quality; GT = patch).
- `forgery/pipeline.py`: `harvest_donor_crops` (per-source splice pool), `build_ops`, `forge_one` (p_clean → clean D5 hard-neg, else sampled applicable op w/ fallback), `generate` (per-doc RNG seeded from doc_id → reproducible). Emits §4-shape records (+ `doc_id`/`source` for step-5 split), `image_path=None`.
- `scripts/viz_sample.py`: forces each op on real docs, overlays GT box, saves PNGs.
- `tests/test_forgery.py`: 14 offline tests (synthetic images) — helpers, each op (valid GT, size preserved, pixels changed, applicability), pipeline clean/tampered/determinism. Suite 94 green.

**Visual gate (eyeballed real forgeries):** digit_swap on SROIE amount lines, copy_move duplicating field values into whitespace, splice mismatched donor over a field, recompress_ghost subtle patch — GT boxes land correctly on all 4 ops × SROIE + FUNSD. `generate()` on 300 docs → 48% clean, op dist {digit_swap 46, recompress_ghost 39, copy_move 38, splice 32}.

**Decisions (all Gemini answers accepted after review — sound):** line-box GT for digit_swap (no OpenCV); intra-doc digit copy-paste not TTF (avoid clean-antialiasing tell); intra-source splice donors; patch-GT for recompress_ghost; parameterized reason templates w/ field interpolation.

**Open / next:** PAUSED per Gemini request for human eyeball of forgery quality + boxes before mass generation. Next = §13 step 5 `data/build_dataset.py` (doc-level split, balanced train/val/test, persist images + HF dataset, `make_manifest.py` stats, leakage assertion).

---

## 2026-07-22 — Stage §13 step 3 (`data/ingest.py`)

**Stage:** §13 step 3 — ingest SROIE + FUNSD → normalized source-doc records with OCR boxes.

**Implemented:**
- `data/ingest.py`: pure normalizers (`is_numeric`, `quad_to_bbox`, `funsd_box_to_pixel`, `strip_bio`, `normalize_{sroie,funsd}_example`) + IO loaders (`load_funsd`, `load_sroie`, `load_source_docs`). Emits **source-doc** record `{source,doc_id,width,height,image(RGB),ocr_boxes:[{text,box_pixel,label,is_numeric}]}` — deliberately NOT the §4 training record (that comes at step 5 after forgery).
- `tests/test_ingest.py`: 20 offline tests (network-free) on the pure fns + fake-example normalization. Suite now 80 green.
- Visual gate: overlaid OCR boxes on real SROIE + FUNSD docs, viewed PNGs — boxes land tightly on words/lines, numeric flag correct. PASSED.

**Decisions / deviations (from Gemini's Step-3 answers):**
- **SROIE dataset ID changed.** Gemini said `darentang/sroie`; verified live → it's a *script* dataset, unusable on datasets>=4 (script loading removed; we have 5.0.0). Switched to parquet mirror `arvindrajan92/sroie_document_understanding` (image + `ocr:[{box:quad, label, text}]`, 652 train docs, labels company/address/date/total/line_total/line_description/other). FUNSD `nielsr/funsd` confirmed parquet, kept.
- **FUNSD coord gotcha (Gemini didn't flag).** `nielsr/funsd` bboxes are 0–1000 LayoutLM-normalized, NOT pixels (bbox maxes 841–946 > image width 762). Convert via `coords.norm_to_pixel`. SROIE quads ARE pixels → just quad→axis-aligned bbox. Both verified visually.
- Agreed with Gemini: quad→axis-aligned bbox; store full `ocr_boxes` list (not just candidate sites); defer CORD.
- Kept test suite network-free — live loading validated by the visual gate script, not pytest.

**Gemini consultation:** Step-3 approach (SROIE geometry, dataset IDs, ocr_boxes storage, CORD). Resolution above — implemented with 2 corrections after live verification (dataset ID, FUNSD coord scale).

**Env note:** py3.14 + datasets 5.0.0 emits a harmless `AttributeError: 'NoneType' ... ArrowInvalid` on streaming-generator close (teardown only; data reads fine). Non-streaming load avoids it.

**Next session starts with:** §13 step 4 — implement `forgery/` ops in D-order (digit_swap → copy_move → splice → recompress_ghost) + `pipeline.py`. Gate: `scripts/viz_sample.py` shows correct GT box on ~30 samples/op by eye. Issue a Gemini prompt on forgery-op design first.

---

## 2026-07-22 — Stages §13 1–2 + governance setup

**Stage:** §13 steps 1 (repo skeleton) + 2 (schema.py + coords.py); then project governance retrofit.

**Implemented:**
- **Step 1 — skeleton (§2):** full tree (`configs/`, `src/forgesight/` + `forgery/`/`data/`/`eval/`/`serve/` subpkgs, `scripts/`, `notebooks/`, gitignored `data/` + `artifacts/`). Module + script stubs as importable placeholders. `requirements-dev.txt` verbatim §3.1. `pyproject.toml` src-layout editable install. `.gitignore`, `README.md` placeholder. Gate: `import forgesight` OK.
- **Step 2 — `coords.py`:** `pixel_to_norm` (clamped 0–1000), `norm_to_pixel`, `iou` (pixel-space, self-normalizes corner order, 0 on degenerate/touching edges).
- **Step 2 — `schema.py`:** `format_box`, `parse_box` (regex on coord core, token-drop + whitespace tolerant, None on miss), `to_target_json` (stable key order, box=null clean), `parse_prediction` (balanced-brace extract → json.loads → type-check → parse_box→`box_norm`; None on unparseable), `validate_record` (tampered/clean invariants).
- **Tests:** 60 pytest, all green (`test_coords.py`, `test_schema.py`).
- **Governance:** `git init`, private GitHub repo `medhulk8/forgesight`, CLAUDE.md, SESSIONS.md, memory files + MEMORY.md index.

**Decisions / deviations:**
- Added `pyproject.toml` (not in §2) — required for `import forgesight` gate with src-layout. No runtime deps declared there (schema/coords are stdlib-only → keeps M3 install light).
- `requirements-kaggle.txt` NOT created yet — Kaggle-side (§3.2), out of current M3 scope.
- `parse_box` regex relaxed to tolerate whitespace inside/around coord pairs (beyond §4's strict regex) — robustness for dropped/reformatted tokens. One test initially failed on this; fixed regex.
- `parse_prediction` returns envelope with added `box_norm` (parsed 0–1000 ints) while preserving raw `box` string — gives eval the coords it needs without a second parse.

**Gemini consultations:** none yet — first `--- GEMINI PROMPT ---` (confirm stage-3 approach) issued end of this session, awaiting reply.

**Next session starts with:** relay Gemini reply on stage-3 approach, then §13 step 3 — `data/ingest.py`: load SROIE + FUNSD, normalize to §4 records with OCR boxes. Gate: print N records, boxes land on words when visualized.
