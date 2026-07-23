# ForgeSight

**Fine-tuning a vision-language model to catch document forgery — and to point at exactly where it happened.**

ForgeSight teaches [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) to look at a scanned receipt or form and answer one question: *has any field been altered, and if so, where?* The model returns a single structured verdict:

```json
{"tampered": true, "field": "total_amount", "box": "<|box_start|>(612,843),(745,878)<|box_end|>", "reason": "Total altered — the digit strokes are heavier than the rest of the row and sit slightly above the baseline."}
```

The box is not a bag of integers bolted onto the JSON. It is written in **Qwen2-VL's own grounding-token format** — the same `<|box_start|>…<|box_end|>` convention the model was pretrained to emit — so fine-tuning *steers* the existing spatial-grounding machinery instead of teaching a language model to relearn coordinate geometry from scratch. That single decision is the spine of the whole project (see [Why native grounding tokens](#why-native-grounding-tokens)).

Everything here is built end to end: a synthetic forgery generator, a QLoRA training pipeline, a purpose-built evaluation benchmark with real statistics — not a wrapper around a hosted API.

> **Quick stats**
> - **Model:** Qwen2-VL-2B-Instruct — QLoRA, 4-bit NF4, ~18.5M / **0.83%** trainable params
> - **Hardware:** single NVIDIA T4 (16 GB) · **Dataset:** 2,964 examples / 801 source docs, zero split leakage
> - **Task:** document forgery detection + spatial grounding, box emitted as native `<|box_start|>` grounding tokens

---

## Contents

- [What it does](#what-it-does)
- [Results](#results)
- [How it works](#how-it-works)
  - [1. Synthetic forgery pipeline](#1-synthetic-forgery-pipeline)
  - [2. The data collator (the hard part)](#2-the-data-collator-the-hard-part)
  - [3. QLoRA fine-tuning](#3-qlora-fine-tuning)
  - [4. ForgeBench evaluation](#4-forgebench-evaluation)
- [Design decisions worth defending](#design-decisions-worth-defending)
- [Reproducing it](#reproducing-it)
- [Project layout](#project-layout)
- [Engineering log: what actually broke](#engineering-log-what-actually-broke)
- [Limitations & honest caveats](#limitations--honest-caveats)
- [License](#license)

---

## What it does

Given a document image, ForgeSight produces a forensic verdict with four parts:

| Field | Type | Meaning |
|-------|------|---------|
| `tampered` | bool | Was anything altered? |
| `field` | string \| null | Which semantic field (e.g. `total_amount`, `date`) |
| `box` | grounding-token string \| null | Where — a normalized 0–1000 bounding box in Qwen2-VL's native format |
| `reason` | string | A short, human-readable justification |

Clean documents return `{"tampered": false, "field": null, "box": null, "reason": "No inconsistencies detected."}`. Teaching the model to *decline* to hallucinate a box on a clean page is as important as localizing a real one — it is enforced with hard-negative clean samples during training (see [D5](#design-decisions-worth-defending)).

Three forgery types are modeled, chosen to span the spectrum from semantic to signal-level:

- **`digit_swap`** — a single digit in a numeric field (a total, a date, an ID) is overwritten with another digit copied from elsewhere *in the same document*, so the font and rendering match perfectly. The hardest to catch and the most on-domain for real fraud (altered amounts).
- **`copy_move`** — a region (a stamp, a signature, a field value) is duplicated and pasted elsewhere with a light blend.
- **`splice`** — a crop from a *different* document of the same source type is pasted into a field, introducing subtle font/lighting mismatch.

---

## Results

> **Status:** the held-out evaluation is produced by [ForgeBench](#4-forgebench-evaluation) against the 360-example test split and a zero-shot baseline. The final training run and evaluation are the last step of the pipeline; the numbers below are populated directly from `artifacts/eval/forgebench_results.json`.

**Detection** — tampered vs. clean, over all test examples:

| Metric | Fine-tuned | Zero-shot baseline | Δ |
|--------|-----------|--------------------|---|
| Precision | _pending_ | _pending_ | — |
| Recall (95% Wilson CI) | _pending_ | _pending_ | — |
| **F1** | _pending_ | _pending_ | — |
| Parse-failure rate | _pending_ | _pending_ | — |

**Localization** — IoU against ground-truth boxes, over true-positive detections only:

| Metric | Fine-tuned | Zero-shot baseline |
|--------|-----------|--------------------|
| **IoU@0.5 hit-rate** | _pending_ | _pending_ |
| Mean IoU | _pending_ | _pending_ |

**Localization by tamper type** (the table that tells you *which* forgeries are visually vs. semantically hard):

| Tamper type | IoU@0.5 | Mean IoU |
|-------------|---------|----------|
| `digit_swap` | _pending_ | _pending_ |
| `copy_move` | _pending_ | _pending_ |
| `splice` | _pending_ | _pending_ |

**Significance:** McNemar's exact test on per-example detection correctness (fine-tuned vs. zero-shot), `p = _pending_`.

Detection quality and localization quality are reported **separately, never blended into one score** — they are two different skills, and averaging them hides exactly the failure mode you most want to see (a model that flags tampering correctly but points at the wrong place, or vice versa). This split is the reason ForgeBench exists rather than a single accuracy number.

---

## How it works

```
SROIE / FUNSD          synthetic forgery         Qwen2-VL chat            QLoRA SFT             ForgeBench
 receipts + forms  ──▶   pipeline (3 ops)   ──▶   messages + native  ──▶  (4-bit NF4,      ──▶  detection F1
 with OCR boxes         (image, GT-box,           grounding-token          LoRA on LLM           + IoU@0.5
                         verdict) triples          JSON target             projections)          + McNemar
```

The architecture obeys one hard rule: **everything under [src/forgesight/](src/forgesight/) imports and unit-tests on a laptop with no GPU and no bitsandbytes.** The GPU-only concerns (4-bit quantized load, training) live behind [model.py](src/forgesight/model.py) and [train_sft.py](src/forgesight/train_sft.py) and are only exercised on Kaggle. This split let ~90% of the risk — data correctness, the collator, masking, the schema round-trip — be de-risked with fast CPU tests before spending a single minute of GPU time.

### 1. Synthetic forgery pipeline

Real forgery datasets with pixel-accurate ground-truth boxes barely exist. So ForgeSight *manufactures* them. The base corpora — [SROIE](https://huggingface.co/datasets/arvindrajan92/sroie_document_understanding) receipts and [FUNSD](https://huggingface.co/datasets/nielsr/funsd) forms — ship with OCR word boxes. Those boxes are both the **candidate tamper sites** and the **source of ground truth for free**: forge a field, and you already know exactly which box changed.

Key correctness decisions, each learned the hard way:

- **`digit_swap` alters using the document's own glyphs.** Rather than rendering a digit with a bundled TTF font (which never matches the scan's rasterization, JPEG history, or anti-aliasing), a digit is copied from elsewhere *in the same document*. The forgery is visually seamless — which is the point; an easy forgery inflates metrics and teaches nothing.
- **Splice donors are intra-source only.** A SROIE crop only ever lands in another SROIE receipt, never in a FUNSD form — otherwise the model learns "different paper texture = tampered," a shortcut that would evaporate in the real world.
- **Coordinate conventions differ per source and are normalized on ingest.** SROIE quads are absolute pixels; FUNSD boxes are already 0–1000 normalized. [ingest.py](src/forgesight/data/ingest.py) converts both to a single pixel-space record shape so nothing downstream has to care.
- **Splitting is by source document, not by example.** A document's clean version and its tampered variants all live in the *same* split. Splitting naively by example would leak the same receipt into both train and test and silently inflate every number. [build_dataset.py](src/forgesight/data/build_dataset.py) enforces this with an explicit leakage assertion that fails the build if any source doc straddles splits.
- **Class balance is applied to the training split only.** The 50/50 tampered/clean balance (via clean-oversampling) is a training-time convenience; validation and test keep their natural distribution so the reported metrics reflect reality.

The resulting dataset: **2,964 examples** (2,244 train / 360 val / 360 test) drawn from **801 source documents** (561 / 120 / 120, zero cross-split leakage). Boxes are stored resolution-independently, so downscaling images for storage never corrupts a label.

### 2. The data collator (the hard part)

This is where VLM fine-tuning quietly goes wrong, so it got the most attention and the strictest test gate. [collator.py](src/forgesight/collator.py) has two jobs that are each easy to get subtly wrong:

**Mask the loss to the answer only.** The model must be trained on the assistant's JSON completion, *not* on the system prompt, the image tokens, or the user turn. Get this wrong and the model either learns to parrot the prompt or learns nothing. The collator sets `labels = -100` everywhere except the assistant completion, located by **searching `input_ids` for the last `<|im_start|>` token** (which always opens the assistant turn) and skipping the fixed-length `assistant\n` header.

**Process each image exactly once.** An earlier design re-rendered the prompt prefix a second time to find the mask boundary — which meant running the heavy vision-patching path *twice per batch* inside the DataLoader, starving the GPU. The current collator is strictly single-pass: images are patched once, and the mask boundary is found by integer tensor ops on the already-tokenized IDs. No re-tokenization, no double vision work.

The acceptance gate ([smoke_collator.py](scripts/smoke_collator.py)) asserts, on CPU, the one thing that matters most: `tokenizer.decode(labels[labels != -100])` is **byte-for-byte equal to the target JSON** — proving the loss lands on the answer and that the native `<|box_start|>…` grounding tokens survive the round-trip intact. It also verifies zero learnable image/pad tokens and that an over-length example trips the truncation guard. Training was not allowed to begin until every assertion passed.

### 3. QLoRA fine-tuning

[model.py](src/forgesight/model.py) loads Qwen2-VL-2B in **4-bit NF4** (double-quantized) and attaches LoRA adapters to the LLM projection layers (`q/k/v/o` + MLP) — **~18.5M trainable parameters, 0.83% of the model.** The vision tower is frozen: cheaper, more stable, and standard for VLM QLoRA. Training is TRL's `SFTTrainer` driving the custom collator; config in [configs/sft.yaml](configs/sft.yaml).

The T4-specific tuning is deliberate, not incidental:

- **fp16, not bf16.** The T4 is a Turing GPU — it has fp16 tensor cores but *no* bf16 tensor cores. Using bf16 (the usual default) silently falls off the fast path. Switching to fp16 was a large speedup on identical hardware.
- **Visual-token budget as the primary lever.** Capping `max_pixels` at `384×28×28` roughly halves the image-token count versus the default, with negligible loss on receipt/form OCR — the single biggest knob on both VRAM and wall-clock.
- **Single-GPU by design.** `device_map={"":0}`, not `"auto"`. Sharding a 2B model across two T4s broke autoregressive generation (KV-cache/hidden states crossing the device boundary produced wrong tokens *despite* a perfect teacher-forced forward pass) — see the [engineering log](#engineering-log-what-actually-broke).
- **paged_adamw_8bit + gradient checkpointing** to fit the optimizer state on a single 16 GB card.

The pipeline was proven end to end with an **overfit-8 sanity check** — 8 examples, loss driven to ~0, model reproduces all 8 target JSONs including correct boxes — before committing to the full run. If the collator, masking, or loss were wrong, this catches it in minutes instead of after an hour of wasted training.

### 4. ForgeBench evaluation

[eval/](src/forgesight/eval/) is a small purpose-built benchmark, not a metrics grab-bag. Its design ([metrics.py](src/forgesight/eval/metrics.py)) reflects how a forensics tool would actually be judged:

- **Detection** (all examples): precision / recall / F1 on the *tampered* positive class, with **Wilson 95% confidence intervals** — honest error bars at this sample size, not point estimates pretending to be exact. Parse failures are counted against detection (a verdict you can't read is a verdict you don't have).
- **Localization** (true positives only): predicted 0–1000 box → pixels → **IoU** vs. ground truth, reported as an **IoU@0.5 hit-rate**, a mean, and — critically — **broken down per tamper type**.
- **Significance:** a zero-shot Qwen2-VL-2B baseline is run through the identical harness, and **McNemar's exact test** (exact binomial, via `scipy.stats.binomtest`) on per-example detection correctness answers the only question that matters: *did fine-tuning actually help, or is the gap noise?*

Run it with [forgebench.py](src/forgesight/eval/forgebench.py); it emits `forgebench_results.json` plus the fine-tuned-vs-baseline deltas.

---

## Design decisions worth defending

These are the choices an interviewer should poke at — each has a real rationale.

<a name="why-native-grounding-tokens"></a>
**Native grounding tokens over raw integer arrays (D2/D3).** The obvious design is `"box": [612, 843, 745, 878]`. It's wrong. Qwen2-VL was pretrained to emit boxes as `<|box_start|>(x1,y1),(x2,y2)<|box_end|>`, where `<|box_start|>` and `<|box_end|>` are *single dedicated vocabulary tokens* — IDs **151648** and **151649** — not text the model spells out character by character. Those two tokens are the switch that gates the model's pretrained spatial-grounding attention heads. A raw int array like `[612, 843, ...]` is just ordinary digit text: it never flips that switch, so it bypasses the grounding machinery entirely and forces the LLM to relearn coordinate geometry from scratch on a few thousand examples. ForgeSight rides the native format *inside* the JSON envelope: the verdict stays cleanly parseable, and the box triggers the grounding heads. The regex parser in [schema.py](src/forgesight/schema.py) matches the `(x,y),(x,y)` core and does **not** depend on the special tokens surviving decode — robust either way.

**Detection and localization are measured separately (D7).** A single "accuracy" number would let a model that always says "tampered" and boxes randomly look mediocre-but-fine. Splitting them exposes the real failure modes and is the honest way to report a two-skill task.

**Hard-negative clean samples instead of preference tuning (D5).** Box hallucination on clean documents is suppressed directly in SFT by training on clean examples with `box=null`, rather than reaching for a second-stage preference method. Simpler, and it works.

**Frozen vision tower.** LoRA on the language projections only. Unfreezing the vision merger is a plausible ablation, but freezing is the cheaper, more stable default and keeps the trainable footprint under 1%.

**Dev/train split across two machines (D10).** All logic is developed and unit-tested on an M3 laptop (CPU/MPS, no bitsandbytes); all real QLoRA training runs on Kaggle T4s. bitsandbytes 4-bit is CUDA-only, so this isn't a preference — it's a constraint turned into a discipline that forced a clean, testable core.

---

## Reproducing it

### Local (CPU/MPS — data, collator, tests; no GPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements-dev.txt

python -c "import forgesight"        # import gate
pytest                               # full unit suite (schema, coords, forgery, collator, metrics)

# build the dataset from SROIE + FUNSD
python -m forgesight.data.build_dataset --config configs/data.yaml
python scripts/viz_sample.py         # eyeball GT boxes on generated forgeries
python scripts/smoke_collator.py     # the collator acceptance gate
```

### Training (Kaggle T4)

The dataset (`data/processed/`, JSONL + images) is uploaded once as a private Kaggle Dataset and attached read-only; the notebook clones this repo and imports `forgesight` so all logic lives in version control, not in cells. See [KAGGLE_UPLOAD.md](KAGGLE_UPLOAD.md) for the upload steps and [notebooks/forgesight_kaggle.ipynb](notebooks/forgesight_kaggle.ipynb) for the run.

```python
# overfit-8 sanity check (proves the whole pipeline in minutes)
!python -m forgesight.train_sft --config configs/sft.yaml --overfit 8

# full SFT run
!python -m forgesight.train_sft --config configs/sft.yaml

# evaluate against the held-out test split + zero-shot baseline
from forgesight.eval import forgebench
forgebench.run("/kaggle/working/adapters/sft", DATA_ROOT, baseline=True)
```

**Pinned environment.** Version drift is the number-one Kaggle failure mode. The tested contract:

```
transformers==4.47.1   trl==0.12.1   peft==0.13.2
accelerate==0.34.2     bitsandbytes==0.49.2   datasets==5.0.0
```

`torch` is intentionally *not* pinned — use Kaggle's preinstalled CUDA build. Attention backend is **SDPA** (FlashAttention-2 requires Ampere; the T4 is Turing).

## Project layout

```
src/forgesight/
├── schema.py            # canonical JSON target + box (de)serialization + validation
├── coords.py            # pixel ↔ 0–1000 normalized conversions, IoU
├── forgery/             # digit_swap · copy_move · splice + orchestration
├── data/
│   ├── ingest.py        # SROIE + FUNSD → common record with OCR boxes
│   ├── build_dataset.py # doc-level split, class balance, leakage assertion
│   └── conversation.py  # record → Qwen2-VL chat messages
├── collator.py          # single-pass, token-search loss masking  ← the crux
├── model.py             # 4-bit NF4 load + LoRA config (GPU-only)
├── train_sft.py         # TRL SFTTrainer entrypoint
├── infer.py             # single-image inference → parsed JSON
└── eval/
    ├── forgebench.py     # run over test split, collect predictions
    └── metrics.py        # F1 + Wilson CI, IoU@0.5, per-type, McNemar
tests/                    # CPU unit suite — the M3 safety net
scripts/                  # smoke_collator · viz_sample · make_manifest
configs/                  # data.yaml · sft.yaml · orpo.yaml
notebooks/                # forgesight_kaggle.ipynb — training + curves
```

## Engineering log: what actually broke

The interesting part of a project is the bugs. A few that cost real time and taught real lessons:

- **Multi-GPU generation silently produced garbage.** With `device_map="auto"` sharding the model across two T4s, teacher-forced forward passes were *100% correct* (loss ≈ 0, argmax matched targets) but `generate()` emitted "clean" for every tampered document. The KV-cache and hidden states crossing the device boundary corrupted autoregressive decoding. Fix: pin to a single GPU (`device_map={"":0}`). A 2B model in 4-bit is ~1.5 GB — it never needed sharding.
- **bitsandbytes version roulette.** `0.44.1` was missing `triton.ops` against Kaggle's newer torch/triton; `0.48+` tripped on `has_avx512bf16`. Pinning `0.49.2` worked — but only after a *factory reset* of the Kaggle session, because a plain restart kept stale layered files from earlier pins.
- **The collator truncation guard fired on every row.** Qwen2 turns end with `<|im_end|>\n`, so the literal last token is a newline, not `<|im_end|>`. The guard had to check for `<|im_end|>` *within the completion span*, not as the final token.
- **A `.gitignore` entry hid a source package.** An unanchored `data/` pattern matched `src/forgesight/data/` and quietly untracked the entire ingest/build package. Anchoring to `/data/` fixed it — a reminder that gitignore globs are not path-anchored by default.
- **`recompress_ghost` was designed, then dropped.** A fourth forgery (double-JPEG-compression ghosting) was built, but the JPEG-downscale used for dataset *storage* destroyed the very compression artifact the op depended on, making it undetectable in the stored data. Rather than poison the benchmark with an impossible class, it was cut — an honest scope decision documented rather than hidden.

## Limitations & honest caveats

- **Synthetic forgeries, not seized evidence.** The model is trained on manufactured tampering. It should transfer to real alterations of the same *kind*, but that is a claim to be tested, not assumed.
- **Two source domains.** SROIE receipts and FUNSD forms only. Other document types (IDs, invoices, contracts) are out of distribution.
- **English, printed text.** No handwriting, no non-Latin scripts.
- **`digit_swap` is deliberately near-perfect visually** — it is meant to be hard, and the per-type localization table is where you see how hard.
- **The zero-shot baseline uses the same prompt** for a fair comparison, but a heavily prompt-engineered baseline might close some of the gap; the McNemar test measures the fine-tuning effect under matched conditions, not the absolute ceiling of prompting.

## License

Released under the **MIT License** — see [LICENSE](LICENSE). The base model ([Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)) and datasets ([SROIE](https://huggingface.co/datasets/arvindrajan92/sroie_document_understanding), [FUNSD](https://huggingface.co/datasets/nielsr/funsd)) are governed by their respective upstream licenses.
