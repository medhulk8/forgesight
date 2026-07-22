# ForgeSight — session log

_Newest entry at top._

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
