# ForgeSight — session log

_Newest entry at top._

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
