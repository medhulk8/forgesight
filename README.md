# ForgeSight

**Fine-tuning a vision-language model to detect and localize document forgery.**

ForgeSight fine-tunes [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) with QLoRA to inspect a scanned receipt or form and return a structured forensic verdict — whether a field was tampered with, which field, a bounding box for the tampered region, and a short reason:

```json
{
  "tampered": true,
  "field": "total_amount",
  "box": "<|box_start|>(612,843),(745,878)<|box_end|>",
  "reason": "The total field shows signs of digit alteration."
}
```

The bounding box is emitted in Qwen2-VL's **native grounding-token format** (0–1000 normalized coordinates), which steers the model's pretrained spatial-grounding weights instead of asking a language model to regress raw integer coordinates from scratch.

> **Status:** research prototype. The data pipeline, training, and evaluation harness are complete and tested end-to-end. The current model fits the training distribution but does not yet generalize to held-out documents — see [Results & findings](#results--findings).

**At a glance**
- **Model:** Qwen2-VL-2B-Instruct, QLoRA (4-bit NF4), LoRA on the LLM + vision attention, trained merger — ~55M / 2.4% trainable parameters
- **Data:** 2,964 synthetic examples from 801 source documents (SROIE + FUNSD), document-level split, zero cross-split leakage
- **Hardware:** single NVIDIA T4 (16 GB)
- **Eval:** ForgeBench — detection (F1 + Wilson CIs) and localization (IoU@0.5) measured separately, with a McNemar test vs. a zero-shot baseline

---

## Contents

- [Overview](#overview)
- [How it works](#how-it-works)
- [Setup](#setup)
- [Usage](#usage)
- [Dataset](#dataset)
- [Evaluation (ForgeBench)](#evaluation-forgebench)
- [Results & findings](#results--findings)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [License](#license)

---

## Overview

Given a document image, ForgeSight produces a single JSON verdict:

| Field | Type | Meaning |
|-------|------|---------|
| `tampered` | bool | Was any field altered? |
| `field` | string \| null | Which semantic field (e.g. `total_amount`, `date`) |
| `box` | string \| null | Tampered region, as native 0–1000 grounding tokens |
| `reason` | string | Short human-readable justification |

Clean documents return `{"tampered": false, "field": null, "box": null, "reason": "..."}`. The model is trained with hard-negative clean samples so it learns not to hallucinate a box on an untampered page.

Three tamper types are modeled, spanning semantic to signal-level edits:

- **`digit_swap`** — a digit in a numeric field is overwritten with another digit copied from within the same document (font/rendering match). The most realistic for fraud.
- **`copy_move`** — a region is duplicated and pasted elsewhere with a light blend.
- **`splice`** — a crop from a different document of the same type is pasted into a field.

---

## How it works

```
SROIE / FUNSD          synthetic forgery         Qwen2-VL chat            QLoRA SFT             ForgeBench
 receipts + forms  ──▶   pipeline (3 ops)   ──▶   messages + native  ──▶  (4-bit NF4, LoRA  ──▶  detection F1
 with OCR boxes         (image, GT box,           grounding-token          on LLM + vision,      + IoU@0.5
                         verdict) triples          JSON target             trained merger)       + McNemar
```

**Data pipeline.** Real forgery datasets with pixel-accurate boxes are scarce, so ForgeSight generates them. SROIE (receipts) and FUNSD (forms) ship with OCR word boxes; those boxes are both the candidate tamper sites and the ground-truth boxes. Splitting is done at the **document level** — a document's clean and tampered variants never straddle train/test — with an explicit leakage assertion.

**Model & QLoRA.** The Qwen2-VL-2B base is loaded in 4-bit NF4 (double-quantized). LoRA adapters (r=16, α=32) are attached to the LLM projections and the vision-tower attention, and the vision→LLM merger is trained in fp16. Only ~2.4% of parameters are trainable, which fits a single 16 GB T4. Training uses `paged_adamw_8bit` and gradient checkpointing; fp16 (not bf16) is used because the T4 is a Turing GPU without bf16 tensor cores, and SDPA attention because FlashAttention-2 requires Ampere.

**Grounding tokens.** The box rides inside the JSON as Qwen2-VL's native `<|box_start|>(x1,y1),(x2,y2)<|box_end|>` string. These are dedicated vocabulary tokens that trigger the model's pretrained grounding heads; a raw integer array would bypass that machinery. The verdict stays cleanly parseable while the box uses the pretrained representation.

**Collator.** A single-pass SFT collator builds the chat messages, processes each image once, and masks the loss to the assistant completion only (everything else set to `-100`) by locating the answer span via token search — no double image processing. A CPU acceptance test asserts the unmasked label span decodes byte-for-byte to the target JSON before any GPU training.

---

## Setup

**Local (CPU/MPS — data generation, collator, tests; no GPU):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements-dev.txt
python -c "import forgesight"    # import check
pytest                           # unit suite (schema, coords, forgery, collator, metrics)
```

**Training environment (Kaggle T4).** The HF stack is version-pinned — drift is the main source of breakage:

```
transformers==4.47.1   trl==0.12.1   peft==0.13.2
accelerate==0.34.2     bitsandbytes==0.49.2   datasets==5.0.0
```

`torch` is intentionally not pinned — use the platform's preinstalled CUDA build. See [requirements-kaggle.txt](requirements-kaggle.txt).

---

## Usage

**Build the dataset** (local, CPU):

```bash
python -m forgesight.data.build_dataset --config configs/data.yaml
python scripts/viz_sample.py         # eyeball ground-truth boxes on generated forgeries
python scripts/smoke_collator.py     # collator acceptance test
```

**Train** (Kaggle T4) — via [notebooks/forgesight_kaggle.ipynb](notebooks/forgesight_kaggle.ipynb), or:

```bash
python -m forgesight.train_sft --config configs/sft.yaml --overfit 8   # pipeline sanity check
python -m forgesight.train_sft --config configs/sft.yaml               # full run
```

**Evaluate** against the held-out test split and a zero-shot baseline:

```python
from forgesight.eval import forgebench
forgebench.run("adapters/sft", DATA_ROOT, baseline=True)
```

---

## Dataset

Generated from [SROIE](https://huggingface.co/datasets/arvindrajan92/sroie_document_understanding) receipts and [FUNSD](https://huggingface.co/datasets/nielsr/funsd) forms.

| Split | Examples | Source documents |
|-------|----------|------------------|
| Train | 2,244 | 561 |
| Val | 360 | 120 |
| Test | 360 | 120 |

- Document-level split, verified zero cross-split leakage.
- Class balance applied to the training split only; val/test keep their natural distribution.
- Boxes are stored resolution-independently, so image downscaling for storage never corrupts a label.

---

## Evaluation (ForgeBench)

ForgeBench reports detection and localization **separately** — blending them into a single accuracy number hides which skill is failing.

- **Detection** (all examples): precision, recall, F1 on the tampered class, with **Wilson 95% confidence intervals** and a separate parse-failure rate.
- **Localization** (true positives only): predicted box → pixels → **IoU** vs. ground truth, reported as IoU@0.5 hit-rate, mean IoU, and a per-tamper-type breakdown.
- **Significance:** a zero-shot Qwen2-VL-2B baseline is run through the identical harness, and **McNemar's exact test** on per-example detection correctness reports whether fine-tuning made a statistically significant difference.

Run with [src/forgesight/eval/forgebench.py](src/forgesight/eval/forgebench.py); results are written to `forgebench_results.json`.

---

## Results & findings

The pipeline, collator, and benchmark are validated end-to-end, and the model demonstrably learns: small overfit controls memorize the target JSON and boxes, and training loss converges. However, on the **document-level held-out test set the model does not generalize** — it fits training forgeries but defaults to "clean" on unseen documents.

The leading hypothesis is that with a limited number of unique source documents and synthetic forgeries, the model latches onto per-document cues rather than transferable tampering artifacts. This is precisely the kind of failure the leakage-free, separately-measured benchmark is designed to expose rather than hide.

Directions being explored: greater source-document diversity, image augmentation to prevent pixel-level memorization, stronger regularization, and decoupling the detection and localization objectives.

---

## Project structure

```
src/forgesight/
├── schema.py            # canonical JSON target + box (de)serialization + validation
├── coords.py            # pixel <-> 0–1000 normalized conversions, IoU
├── forgery/             # digit_swap · copy_move · splice + orchestration
├── data/
│   ├── ingest.py        # SROIE + FUNSD -> common record with OCR boxes
│   ├── build_dataset.py # doc-level split, class balance, leakage assertion
│   └── conversation.py  # record -> Qwen2-VL chat messages
├── collator.py          # single-pass, token-search loss masking
├── model.py             # 4-bit NF4 load + LoRA config (GPU-only)
├── train_sft.py         # TRL SFTTrainer entrypoint
├── infer.py             # single-image inference -> parsed JSON
└── eval/
    ├── forgebench.py     # run over test split, collect predictions
    └── metrics.py        # F1 + Wilson CI, IoU@0.5, per-type, McNemar
tests/                    # CPU unit suite
scripts/                  # smoke_collator · viz_sample · make_manifest
configs/                  # data.yaml · sft.yaml
notebooks/                # forgesight_kaggle.ipynb
```

---

## Limitations

- Trained on **synthetic** forgeries, not real-world tampering; transfer to genuine alterations is unverified.
- Two source domains only (SROIE receipts, FUNSD forms); other document types are out of distribution.
- English, printed text only — no handwriting or non-Latin scripts.
- The current model does not generalize to held-out documents (see [Results & findings](#results--findings)).

---

## License

Released under the [MIT License](LICENSE). The base model ([Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)) and datasets ([SROIE](https://huggingface.co/datasets/arvindrajan92/sroie_document_understanding), [FUNSD](https://huggingface.co/datasets/nielsr/funsd)) are governed by their respective upstream licenses.
