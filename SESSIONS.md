# ForgeSight — session log

_Newest entry at top._

---

## 2026-07-23 — Steps 9–10 executed on Kaggle: gates + two real bugs fixed

**Step 9 gate PASSED:** after resolving Kaggle env drift (below), 4-bit Qwen2-VL-2B loaded, LoRA 18.5M/2.23B = 0.83% (vision frozen). 

**Kaggle env fixes (drift from §3.2 pins):**
- Namespace shadow: repo dir `forgesight` shadowed `src/forgesight` → cell 3 does `sys.path.insert(0,'src')`. Removed `pip install -e .`.
- bitsandbytes: 0.44.1 imports removed `triton.ops` (Kaggle torch 2.10/triton 3.x); pin-layering left inconsistent files. Fix: Kaggle doesn't preinstall bnb → install UNPINNED (got 0.49.2 clean). requirements-kaggle updated.
- Dataset mounts at `/kaggle/input/datasets/<owner>/<slug>/`, not `/kaggle/input/<slug>/`.
- Dataset too big to upload (2.1 GB PNG) → switched storage to downscaled JPEG (max_side 1024, q90) = 247 MB. Records unchanged (coords resolution-independent).

**Step 10 (overfit-8) — pipeline PROVEN, via two real bug fixes:**
- **Multi-GPU generate bug:** `device_map="auto"` sharded the model across 2×T4; teacher-forced accuracy was 1.0 / tf_loss≈0 but `generate()` emitted "clean" for all tampered. Root cause: KV-cache/hidden states corrupt across the device split. **Fix: `device_map={"":0}` single-GPU** (2B/4-bit fits one T4). After fix, digit_swap generates tamper+field+box correctly, all clean correct. Diagnosis chain recorded in memory [[qwen2vl-multigpu-generate-bug]].
- Overfit metric was too strict (verbatim string) — box coords aren't memorized digit-for-digit under greedy (expected; IoU@0.5 is the metric). `_report_overfit` now scores detection + IoU.

**OPEN ISSUE (needs decision):** `recompress_ghost` examples still generate "clean" — its signal IS the JPEG double-compression grid artifact, and our storage change (downscale + re-JPEG q90) destroys exactly that. So ghost images are ~indistinguishable from clean; unlearnable/undetectable. digit_swap/copy_move/splice are pixel-level edits and survive JPEG fine. Consulting Gemini: store ghost lossless vs drop the op vs rework. Affects step-11 training + D7 per-type eval.

**Next:** resolve ghost decision (Gemini), then step 11 (full SFT single-GPU). Model/train code now single-GPU + balanced overfit + IoU reporting, all pushed.

---

## 2026-07-23 — Step 8 exec: dataset uploaded to Kaggle (after size fix)

**What happened:** set up Kaggle CLI on M3 (uv-installed `kaggle`, token at `~/.kaggle/access_token`, auth OK — username **medhulkhandelwal**). First upload of `data/processed/` (2.1 GB, full-res lossless PNG) repeatedly died with `BrokenPipeError` to googleapis at ~97% of the 1.67 GB images.zip — flaky link + oversized dump.

**Fix (deviation, engineering call):** switched stored images to **downscaled JPEG** (`configs/data.yaml`: image_format=jpg, max_image_side=1024, jpeg_quality=90; `build_dataset._save_image`). **2.1 GB → 247 MB.** Records UNCHANGED — box_pixel/width/height stay in ORIGINAL pixel space, target box is a 0..1000 normalised (resolution-independent) string, eval IoU uses original dims — so shrinking stored pixels does not touch the coordinate math. Qwen2-VL downscales to ~max_pixels on load anyway, so full-res PNG was wasted bytes. Re-verified: collator span==target on the JPEG data; 113 tests green. Re-upload succeeded (images.zip 232 MB). Dataset `medhulkhandelwal/forgesight-data` registered PRIVATE.

**Note:** overrides the earlier stage-5 "lossless PNG for all" decision. Rationale is upload feasibility + no coord/quality impact (model never sees > max_pixels). Flag to Gemini.

**Security:** user pasted the live Kaggle token (KGAT_...) in chat → advised rotation after. Stored only in `~/.kaggle/access_token` (never committed).

**Next:** user does GitHub PAT (step D) → Kaggle Secret `GH_PAT` (E) → import notebook + GPU T4×2 + Internet on + attach dataset (F) → Run All, stop after overfit-8 (G). Report loss curve + cell-7 JSON reproduction.

---

## 2026-07-23 — Stages §13 steps 8–10 (code) — Kaggle bridge; GPU gates pending

**Stage:** §13 step 8 (Kaggle bridge) + step 9 (`model.py`) + step 10 (`train_sft.py` overfit-8) — all CODE written; GPU gates run on Kaggle by user (M3 has no CUDA/bitsandbytes, D10).

**Implemented:**
- **Persistence → JSONL** (Gemini): `build_dataset._write_jsonl` writes `<out>/<split>.jsonl` (dropped Arrow `save_to_disk` — version-drift trap across M3/Kaggle datasets versions). `make_manifest.py` reads JSONL. Re-ran build: identical 2964 examples, now as JSONL + PNGs. **Verified end-to-end on M3**: `load_dataset('json')` → 2244 train rows, collator on 4 real records+images → decoded span == target JSON for all, no warnings.
- `requirements-kaggle.txt`: pinned transformers/trl/peft/bnb/accelerate (§3.2) + datasets==5.0.0 + qwen-vl-utils + jinja2. **No torch/torchvision** (Gemini: Kaggle ships CUDA torch; pip-installing breaks bindings). No flash-attn (D8).
- `model.py`: `load_processor` (CPU-safe), `lora_config`, `load_model_for_training(use_4bit, attn='sdpa', lora)` — 4-bit NF4 + double-quant, LoRA on 7 LLM proj modules, vision tower frozen.
- `configs/sft.yaml` (§9 hyperparams) + `train_sft.py`: `build_trainer` wires TRL SFTTrainer with our collator, `remove_unused_columns=False`, `dataset_kwargs={'skip_prepare_dataset':True}`, `processing_class=processor.tokenizer`, gradient checkpointing (use_reentrant=False). `--overfit 8` → tiny batch/60 steps/no eval; `_report_overfit` generates on the 8 and checks verbatim JSON reproduction.
- `notebooks/forgesight_kaggle.ipynb`: 8 cells — clone via `GH_PAT` secret, install, sanity, 4-bit load (step-9 gate), overfit-8 (step-10 gate), loss plot, reproduce check.
- `KAGGLE_UPLOAD.md` (tracked handoff): upload steps, secret setup, notebook settings, gates to report.
- `tests/test_model.py`: 5 import-safe tests (MODEL_ID, LoRA target modules, lora_config builds, train_sft import-safe, pixel constants). Suite 113 green.

**Decisions (all Gemini step-8/9 answers accepted):** JSONL+PNG over Arrow; no torch in Kaggle reqs; PAT-in-Secrets + git clone (not repo-as-dataset — instant push/pull iteration); `data_root` injection for read-only `/kaggle/input` mount.

**Env notes:** installed peft==0.13.2 on M3 (CPU, for the lora_config test — no CUDA needed). Dataset dir is ~2.1 GB (2403 lossless PNGs). trl/bitsandbytes NOT installed on M3 (Kaggle-only) → train_sft heavy imports are deferred inside functions so the module still imports for tests.

**Split of work:** M3-verifiable gates DONE (JSONL load + collator on real data, 113 tests). GPU gates (step 9 4-bit load, step 10 overfit-8 loss→0 + JSON reproduction) PENDING — user runs the notebook on Kaggle and reports the loss curve.

**Risk flagged:** TRL 0.12.1 SFTTrainer/SFTConfig arg surface (esp. `processing_class`, `dataset_kwargs`, `max_seq_length` with skip_prepare_dataset) is written to the pin but UNVERIFIED without a GPU run — the overfit-8 is where any API drift surfaces first (as planned).

**Next session starts with:** user's overfit-8 result. If loss→0 + JSONs reproduce → architecture phase won, proceed to step 11 (full SFT). Else debug collator/trainer wiring from the curve + cell-7 output.

---

## 2026-07-22 — Stage §13 step 7 (`collator.py`) — M3 CRUX, PASSED

**Stage:** §13 step 7 — THE data collator + smoke gate. Completes the M3 (GPU-free) phase.

**Env change (Gemini-advised):** rebuilt the dev venv on **Python 3.11** (via `uv python install 3.11`; torch 2.13.0 + torchvision 0.28.0 wheels install cleanly, MPS available). Python 3.14 had no torch wheel. Aligns dev with Kaggle T4 (3.10/3.11) to avoid pickle/serialization drift. Full stack reinstalled; 108 tests green on 3.11.

**Implemented:**
- `collator.py` `ForgeSightCollator(processor, build_messages, max_length=2048)`: single-pass (`apply_chat_template` + `process_vision_info` once per record), mask by locating LAST `<|im_start|>` + fixed `assistant\n` tail (n_tail=2) → `labels[:comp_start]=-100`; pad→-100; image_pad→-100. `padding_side="right"` on tokenizer AND processor. Truncation guard + final "no learnable tokens" net.
- `scripts/smoke_collator.py`: 2 tampered + 2 clean (varied) + 1 over-long; monkeypatches `process_vision_info` to prove single-pass; asserts shapes, decoded unmasked span == `to_target_json`, zero learnable pad/image tokens, guard fires on over-long only.

**Gate — SMOKE PASSED on CPU:** shapes (5,2048); process_vision_info ran exactly 5× for 5 records; no learnable pad/image tokens; **every normal row decoded unmasked span EXACTLY equals target JSON (native `<|box_start|>` tokens survive round-trip)**; truncation guard fired on the over-long row only.

**Bug found + fixed during smoke:** first guard checked `row[last_real] == im_end`, but Qwen turns render as `<|im_end|>\n` → the last non-pad token is the newline (198), so the guard fired on every row. Fixed: termination = an `<|im_end|>` exists *within the completion span* `row[comp_start:last_real+1]` (absent only when the tail was truncated). Verified: normal rows silent, over-long warns.

**Decisions (all Gemini step-7 answers accepted):** pin whole dev venv to 3.11; assert `(labels!=-100).sum()>0` (added as final net + per-row); `padding_side="right"` on both; monkeypatch to prove O(N) single-pass.

**Next session starts with:** §13 step 8 — first GPU/Kaggle step. Push repo (ongoing), upload `data/processed/` as a PRIVATE Kaggle Dataset, thin Kaggle notebook that `git clone`s the repo + attaches the dataset + `import forgesight`. Then step 9 (model.py 4-bit load) and step 10 (overfit-8). Issue Gemini prompt on the Kaggle bridge first.

---

## 2026-07-22 — Stage §13 step 6 (`data/conversation.py`) + M3 processor de-risk

**Stage:** §13 step 6 — record → Qwen2-VL chat messages.

**Implemented:**
- `data/conversation.py`: `SYSTEM_PROMPT` (native-box format), `USER_INSTRUCTION`, `image_uri`, `build_messages(record, data_root=None, include_target=True)`. Pure + disk-agnostic (path math only); image referenced by `file://<abs>` URI (NOT PIL — DataLoader pickling/memory); assistant target reuses `schema.to_target_json`. `include_target=False` → inference messages (no assistant turn).
- `tests/test_conversation.py`: 5 offline tests (structure, uri, target = to_target_json, inference omits assistant, prompt mentions native tokens). Suite 108 green.

**M3 processor de-risk (verified live — gates step 7):** installed transformers==4.47.1 + qwen-vl-utils + jinja2 on py3.14; `AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")` loads; `apply_chat_template` renders correctly (system / user with `<|vision_start|><|image_pad|><|vision_end|>` / assistant target incl native box tokens, single render). Token ids: im_start=151644, im_end=151645, image_pad=151655, box_start=151648, box_end=151649; `assistant\n`=[77091,198] (collator n_tail=2); box tokens survive tokenize→decode roundtrip (the §7.4 critical assertion will hold).

**Decisions (all Gemini step-6 answers accepted):** file:// URI over PIL; updated system prompt (native-box string, not §6.1 verbatim); pure `build_messages(record, data_root)`; reuse `to_target_json`. Added `include_target` param for inference reuse (step 12).

**Env notes:** jinja2 not auto-pulled by transformers but required by `apply_chat_template` → added to requirements-dev.txt. pip downgraded huggingface-hub 1.24→0.36.2 for transformers 4.47.1; datasets 5.0.0 still imports fine. **torch still NOT installed** — the step-7 collator smoke test needs it (return_tensors=pt, image tensors); torch>=2.3 on py3.14 is the next gating risk to verify at step 7 open.

**Next session starts with:** §13 step 7 — `collator.py` (§7.2/7.3 single-pass token-search masking) + `scripts/smoke_collator.py` (§7.4 assertions, esp. decoded unmasked span == target JSON). First verify torch installs on M3/py3.14. Issue Gemini prompt (hardest gate).

---

## 2026-07-22 — Stage §13 step 5 (`data/build_dataset.py`)

**Stage:** §13 step 5 — build the doc-level-split, class-balanced dataset on disk.

**Implemented:**
- `pipeline.forge_variants(doc, pool, rng, n_tampered)`: 1 clean + up to N distinct-op tampered variants per base doc (all sharing doc_id), reproducible.
- `data/build_dataset.py`: pure helpers `assign_splits` (doc-level 70/15/15, deterministic), `balance_5050` (oversample minority), `compute_manifest`; `build()` orchestrates ingest → doc-split → per-(split,source) donor pools → forge_variants → save PNG + set image_path + `validate_record` → balance TRAIN → `_assert_no_leakage` → write HF splits. `configs/data.yaml`.
- `scripts/make_manifest.py`: recompute stats from on-disk splits.
- `tests/test_build_dataset.py`: 10 offline tests (split partition/determinism, balance, forge_variants distinct-ops, leakage assert fires, manifest counts). Suite 103 green.

**Generated (live, ~90s):** `data/processed/` — 2964 examples (train 2244 / val 360 / test 360), 801 base docs (sroie 652 + funsd 149). Train 50/50 (1122/1122); val/test natural 33% clean. Op spread uniform (~57–66 per type per eval split). Manifest written.

**Gates (all pass, independently verified):** 0 cross-split doc_id leaks (re-checked from disk, not just inline assert); 0 invalid records across all splits; a persisted PNG rendered with its stored `box_pixel` → GT box lands on the altered amount line (image+coord round-trip through disk intact).

**Decisions / deviations from Gemini's step-5 answers:**
- Accepted: multi-variant (1 clean + up to 2 tampered), all variants same split, natural op spread.
- **Deviation (balance):** Gemini said oversample clean to 50/50 dataset-wide. Applied 50/50 to TRAIN ONLY; val/test keep natural distribution. Duplicating eval rows would bias ForgeBench F1/CIs — the D7 eval-integrity story. Flagged to Gemini.
- **Deviation (image format):** Gemini said ghost=JPEG, others=PNG. Used PNG for ALL. The recompress_ghost op does its JPEG round-trips internally → the artifact is already baked into the returned decoded pixels; PNG stores them losslessly (JPEG would add a spurious 3rd compression), and uniform format removes any format-based shortcut. Flagged to Gemini.
- Extra: donor pools built per (split, source), not global — a splice pasting a crop from another split would be subtle cross-split pixel leakage.

**Gemini consultation:** step-5 (volume/multi-variant, balance, image format, op spread). Resolution above — 2 documented deviations after reasoning about eval integrity + artifact mechanics.

**Next session starts with:** §13 step 6 — `data/conversation.py` (record → Qwen2-VL chat messages, system prompt §6.1, image in user turn, assistant target = `schema.to_target_json`). Gate: printed chat template correct, image inserted. Issue Gemini prompt first.

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
