# ForgeSight — Implementation Plan

> **What this file is:** a complete, self-contained build specification for the ForgeSight project. It is written to be dropped into a *fresh* chat/agent as the single source of truth so implementation can begin immediately. Everything needed — design decisions, repo layout, environment, data schema, the data collator, training config, evaluation, serving, and the build/test order — is here. There is intentionally **no day-by-day breakdown**; work is organized by *component and dependency order* so it survives schedule slippage.

---

## 0. One-paragraph thesis (paste this into interviews)

ForgeSight fine-tunes an open-weight Vision-Language Model (**Qwen2-VL-2B-Instruct**) with **QLoRA** to detect, **spatially localize**, and explain tampering in document images. Given a document, the model emits structured JSON: `{tampered, field, box[x1,y1,x2,y2], reason}`, where the box uses **Qwen2-VL's native 0–1000 normalized coordinate convention**. Training data comes from a **self-built synthetic forgery pipeline** (copy-move, splice, digit-swap, recompression ghosting) that produces `(image, ground-truth box, label)` triples for free. The model is evaluated on a custom benchmark, **ForgeBench**, that *separately* measures detection quality (F1 on tampered-vs-clean) and localization quality (IoU@0.5 on tampered regions), with Wilson confidence intervals and a McNemar test against a baseline. Serving is profiled under 4-bit quantization (latency, VRAM). A stretch ORPO stage penalizes hallucinated boxes/reasons.

**Why it maps to HyperVerge:** forgery/fraud detection + document intelligence + multimodal reasoning + spatial grounding — their exact problem space — demonstrated end-to-end (data → train → eval → serve), not as an API wrapper.

---

## 1. Locked design decisions (do not relitigate)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Model: Qwen2-VL-2B-Instruct** | Fits T4 QLoRA; pretrained for grounding (emits native box tokens); strong OCR. Fallback only if broken: `Qwen2-VL-2B` base or LLaVA-1.5-7B. |
| D2 | **Coordinates: Qwen2-VL native grounding tokens**, `<\|box_start\|>(x1,y1),(x2,y2)<\|box_end\|>`, coords 0–1000 int scale | **(Revised — Gemini review)** Raw int arrays bypass the pretrained spatial-grounding heads and force the LLM to relearn box math from scratch. The `<\|box_start\|>`/`<\|box_end\|>` special tokens are what trigger Qwen2-VL's grounding weights. Use them. |
| D3 | **Output = JSON envelope; box value = the native-token string**; `schema.py` regex-parses it for IoU | Keeps deterministic structured parsing (tampered/field/reason are clean JSON) while the box rides inside as the native grounding string. Best of both. |
| D4 | **Primary technique: QLoRA-SFT via TRL `SFTTrainer`** | Core resume claim. Must land. |
| D5 | **Hard-negative clean samples in SFT** (clean doc → `{tampered:false}`, no box) | Suppresses box hallucination without needing preference tuning. |
| D6 | **ORPO = stretch only** | Touch only if SFT is fully wrapped with time to burn. Pairs = correct-box vs plausibly-wrong-box, grounded vs hallucinated reason. |
| D7 | **Metric split**: Detection **F1** (tampered vs clean) *separate from* Localization **IoU@0.5** (given tamper, boxed correctly) | Two different skills; blending them hides failure modes. This separation is the "benchmark I designed" story. |
| D8 | **Attention backend on T4 = SDPA** | FlashAttention-2 requires Ampere (sm_80+). T4 is Turing (sm_75) → FA2 unavailable. FA2 only if an Ampere GPU (e.g., Colab A100) is used. |
| D9 | **Serving profile: bitsandbytes 4-bit + HF `generate()`** as the reliable path; **vLLM = stretch** | vLLM multimodal on T4 is brittle. bnb-4bit latency/VRAM numbers satisfy the production-optimization requirement regardless. |
| D10 | **Dev on M3 (MPS/CPU) for data-gen + collator + shape tests; all real training on Kaggle 2×T4** | bitsandbytes 4-bit is CUDA-only — cannot QLoRA on M3. M3 validates the *pipeline*, Kaggle runs the *training*. |

---

## 2. Repository structure

```
forgesight/
├── README.md                      # final deliverable — written last, drafted throughout
├── plan.md                        # this file
├── requirements-dev.txt           # M3 / CPU dev deps (no bitsandbytes)
├── requirements-kaggle.txt        # Kaggle T4 training deps (pinned)
├── configs/
│   ├── data.yaml                  # tamper types, severities, split ratios, counts
│   ├── sft.yaml                   # QLoRA + SFTTrainer hyperparams
│   └── orpo.yaml                  # stretch
├── src/
│   └── forgesight/
│       ├── __init__.py
│       ├── schema.py              # canonical JSON target schema + (de)serialization + validation
│       ├── coords.py              # pixel <-> 0..1000 normalized conversions, IoU
│       ├── forgery/
│       │   ├── __init__.py
│       │   ├── base.py            # ForgeryOp interface: apply(image, ocr_boxes) -> (image, gt_box, meta)
│       │   ├── copy_move.py
│       │   ├── splice.py
│       │   ├── digit_swap.py
│       │   ├── recompress_ghost.py
│       │   └── pipeline.py        # orchestrates: pick field -> apply op -> emit record
│       ├── data/
│       │   ├── ingest.py          # load base datasets (FUNSD/SROIE/CORD), normalize to common record
│       │   ├── build_dataset.py   # run forgery pipeline over base -> HF dataset on disk (train/val/test)
│       │   └── conversation.py    # record -> Qwen2-VL chat messages (system/user+image/assistant JSON)
│       ├── collator.py            # THE data collator (first major deliverable — §7)
│       ├── model.py               # load Qwen2-VL + QLoRA config + processor, attn backend switch
│       ├── train_sft.py           # TRL SFTTrainer entrypoint
│       ├── train_orpo.py          # stretch
│       ├── infer.py               # single-image inference -> parsed JSON
│       ├── eval/
│       │   ├── forgebench.py      # run model over test split, collect predictions
│       │   ├── metrics.py         # F1, IoU@0.5, Wilson CI, McNemar
│       │   └── report.py          # tables + plots (per-tamper-type breakdown)
│       └── serve/
│           └── profile.py         # bnb-4bit load, latency p50/p95, peak VRAM, tok/s
├── scripts/
│   ├── smoke_collator.py          # runs collator on 4 hand-made samples on CPU, asserts shapes+masking
│   ├── viz_sample.py              # draws GT box on image to eyeball forgery + label correctness
│   └── make_manifest.py           # writes dataset stats/manifest for README
├── notebooks/
│   └── forgesight_kaggle.ipynb    # the Kaggle training/eval notebook (curves live here)
├── data/                          # gitignored — base + generated (see §5)
│   ├── raw/
│   └── processed/
└── artifacts/                     # gitignored — adapters, eval json, plots
    ├── adapters/
    ├── eval/
    └── plots/
```

**Design rule:** everything under `src/forgesight/` is importable and unit-testable on the M3 with **no GPU and no bitsandbytes**. GPU-only concerns (4-bit load, training) live behind `model.py`/`train_*.py` and are only exercised on Kaggle.

---

## 3. Environment

### 3.1 M3 dev (`requirements-dev.txt`)
Goal: run data generation, the collator, schema/coords/IoU, and shape assertions. No 4-bit, no training.

```
torch>=2.3                 # MPS build
torchvision
transformers==4.47.1
datasets>=3.0
qwen-vl-utils
pillow
opencv-python
numpy
pyyaml
matplotlib
scipy                      # Wilson CI, McNemar
pytest
# NOTE: NO bitsandbytes (CUDA-only), NO flash-attn, NO trl needed for dev-only shape tests
```

- The processor (`Qwen2VLProcessor`) runs fine on CPU/MPS — that is all the collator needs.
- You *can* load Qwen2-VL-2B in fp16/bf16 on M3 for a tiny generate() sanity check, but it will be slow and memory-heavy. Optional, not required.

### 3.2 Kaggle T4 (`requirements-kaggle.txt`)
Kaggle ships torch + CUDA. Pin the HF stack to known-compatible versions and install the rest.

```
transformers==4.47.1
trl==0.12.1
peft==0.13.2
bitsandbytes==0.44.1
accelerate==0.34.2
datasets>=3.0
qwen-vl-utils
# flash-attn: DO NOT install on T4 (Turing). Use attn_implementation="sdpa".
```

Kaggle notebook first cell:
```bash
pip -q install -U transformers==4.47.1 trl==0.12.1 peft==0.13.2 \
    bitsandbytes==0.44.1 accelerate==0.34.2 datasets qwen-vl-utils
```
> Version-drift is the #1 Kaggle failure. If `Qwen2VLForConditionalGeneration` import fails, bump `transformers`; if `SFTConfig`/`ORPOConfig` args mismatch, that's a `trl` pin issue — the versions above are the tested contract. Record the exact working versions in the README.

### 3.3 The bridge (M3 → Kaggle)
- **Code transport:** GitHub repo. M3 pushes; Kaggle notebook does `!git clone` (or pip-install-from-git) at top, then `from forgesight...`. Keeps the notebook thin — logic lives in the repo, not in cells.
- **Data transport:** generate the dataset on M3 (CPU-cheap: drawing on images), then upload `data/processed/` as a **Kaggle Dataset** (private). Notebook attaches it read-only at `/kaggle/input/forgesight-data/`. Do **not** regenerate data on Kaggle — generate once on M3, version it.
- **Artifacts back:** adapters + eval JSON + plots saved to `/kaggle/working/`, downloaded, committed to `artifacts/` (or released via GH release if >100MB; LoRA adapters are small, usually fine).
- **Secrets:** none needed (all open models/data). If you gate the model, put HF token in Kaggle Secrets, not in code.

---

## 4. Canonical data schema (`schema.py`)

One record shape flows through the entire system.

```python
# Intermediate record (dataset on disk), one per training example:
{
  "image_path": "data/processed/images/000123.png",
  "width": 1654, "height": 2339,          # ORIGINAL pixel dims (for coord conversion + IoU in pixel space)
  "tampered": true,
  "field": "total_amount",                # semantic field name, or null if clean
  "tamper_type": "digit_swap",            # one of the ops, or null if clean
  "box_pixel": [x1, y1, x2, y2],          # ground-truth box in ORIGINAL pixels, or null if clean
  "reason": "The total was altered; digit stroke weight and baseline differ from the rest of the row."
}
```

**Model target (what the assistant must output) — box is the native grounding-token string, coords 0–1000 int scale (D2/D3):**
```json
{"tampered": true, "field": "total_amount", "box": "<|box_start|>(612,843),(745,878)<|box_end|>", "reason": "..."}
```
Clean example target:
```json
{"tampered": false, "field": null, "box": null, "reason": "No inconsistencies detected."}
```

`schema.py` provides:
- `format_box(norm_box) -> str` → `"<|box_start|>(x1,y1),(x2,y2)<|box_end|>"` from four 0–1000 ints.
- `parse_box(box_str) -> [x1,y1,x2,y2] | None` → regex `r"\((\d+),(\d+)\),\((\d+),(\d+)\)"` inside the box string; tolerant of the tokens being present or stripped; returns `None` if no 4-int match (counts as a localization miss).
- `to_target_json(record) -> str` (pixel box → 0–1000 via `coords.py` → `format_box`; compact `json.dumps`, stable key order; `box=null` for clean).
- `parse_prediction(text) -> dict | None` (robust: extract first `{...}` JSON block, `json.loads`, validate keys/types, then `parse_box` the `box` field; return `None` on unparseable JSON — counts as a detection miss in eval).
- `validate_record(record)` used by data build to reject malformed entries early.

> **Regex note:** `<|box_start|>` etc. are literal text in the decoded output but are *single special tokens* to the tokenizer. `parse_box` must match on the **decoded string**, and must not depend on the special tokens surviving — match the `(x1,y1),(x2,y2)` core so parsing is robust even if a token is dropped.

`coords.py`:
- `pixel_to_norm(box, w, h) -> [0..1000 ints]`, `norm_to_pixel(box, w, h)`.
- `iou(box_a, box_b) -> float` (operate in a single consistent space — do IoU in **pixel** space at eval by converting predicted 0–1000 back to pixels using the known test-image dims).

---

## 5. Synthetic forgery pipeline (`forgery/`)

**Base datasets** (`data/ingest.py` normalizes each into the §4 record with OCR word boxes):
- **SROIE** (receipts, has field annotations + boxes) — richest for digit/total tampering.
- **FUNSD** (scanned forms, word-level boxes) — good for field splice/copy-move.
- **CORD** (receipts, structured fields) — backup / diversity.
- Start with **SROIE + FUNSD**; add CORD only if you need volume/diversity.

Each dataset gives word/field bounding boxes → those boxes are the **candidate tamper sites** and the source of **ground-truth boxes for free**.

**`ForgeryOp` interface (`forgery/base.py`):**
```python
class ForgeryOp:
    name: str
    def applicable(self, record) -> bool: ...
    def apply(self, image: PIL.Image, ocr_boxes: list) -> tuple[PIL.Image, list[int], dict]:
        """Returns (tampered_image, gt_box_pixel, meta{field, reason})."""
```

**Four ops (implement in this order — easiest/most-visible first):**
1. **`digit_swap`** — pick a numeric field (total/date/ID), render a different digit over one glyph using a **font matched in size/color** sampled from the image's own text region; add subtle noise. GT box = the altered glyph's box. *Most on-domain for fraud (altered amounts).* Build first.
2. **`copy_move`** — copy a patch (e.g., a stamp/signature/field value) from one region, paste elsewhere with slight blend. GT box = paste location.
3. **`splice`** — paste a crop from a *different* document (different font/lighting) into a field. GT box = spliced region.
4. **`recompress_ghost`** — tamper a region then JPEG-recompress the whole image at differing quality so the tampered patch shows double-compression artifacts. GT box = tampered patch. *Adds a "signal-level" forgery the model must catch from texture, not text.*

**`pipeline.py`:** for each base image, with configured probability emit either (a) a clean record (hard negative, D5) or (b) apply one sampled op → tampered record. Balance classes (~50/50 tampered/clean; within tampered, roughly uniform over op types). Emit `(image_file, record)` and write an HF `datasets` Dataset per split.

**Splits (`configs/data.yaml`):** split **by source document** (a doc's clean and tampered variants must not straddle train/test) to prevent leakage — reuse the scene-level-split discipline from the Drywall project. Suggested: train 70 / val 15 / test 15. Target counts: start ~1.5–3k train examples (enough for LoRA on a 2B VLM; scale up only if underfitting).

**`scripts/viz_sample.py`** overlays the GT box on the generated image → you *must* eyeball ~30 samples per op before trusting labels. Bad boxes = poisoned training + meaningless IoU.

---

## 6. Conversation formatting (`data/conversation.py`)

Turn a record into Qwen2-VL chat messages.

### 6.1 System prompt (fixed)
```
You are a document forensics expert. Given a document image, determine whether any field has been tampered with. Respond ONLY with a compact JSON object with keys: tampered (bool), field (string or null), box (string or null), reason (string). For box, output the tampered region using the grounding format <|box_start|>(x1,y1),(x2,y2)<|box_end|> with integer coordinates on a 0-1000 normalized scale. If untampered, set tampered=false, field=null, box=null.
```

### 6.2 Messages
```python
messages = [
  {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
  {"role": "user", "content": [
      {"type": "image", "image": pil_image},                 # or file path per qwen-vl-utils
      {"type": "text", "text": "Inspect this document for tampering."}]},
  {"role": "assistant", "content": [{"type": "text", "text": target_json_str}]},
]
```

### 6.3 Rendering
- **Full text** (for training): `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`. This is the **only** render — no separate prompt-prefix render (see §7.2, single-pass masking).
- Image tensors via `qwen_vl_utils.process_vision_info(messages)` → processed **once** per example.

### 6.4 (Rejected alternative) raw-int box array
An earlier draft used `"box": [x1,y1,x2,y2]` raw ints. **Rejected (Gemini review):** raw arrays bypass Qwen2-VL's pretrained grounding heads and make the LLM relearn spatial math. The native `<|box_start|>`-token string (D2/D3) is the committed format. Keep this note in the README as a documented design tradeoff — shows you understand the tokenization/grounding mechanism, not just JSON.

---

## 7. THE Data Collator (`collator.py`) — first major code deliverable

This is the highest-risk component. Spec it exactly, then implement, then prove it with `scripts/smoke_collator.py` on CPU **before touching Kaggle**.

### 7.1 Responsibilities
1. Accept a batch of records.
2. Build messages (§6) per record.
3. Produce `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw` via the processor (batched in one call — the processor handles variable image-token counts and text padding).
4. Produce `labels` = `input_ids` with **everything that is not the assistant completion set to `-100`**, specifically masking:
   - all prompt tokens (system + user + image placeholder tokens + assistant header),
   - all pad tokens,
   - all image placeholder tokens (belt-and-suspenders; they sit in the prompt anyway).
5. Right-padding; locate the assistant completion by token-search on `input_ids` (single-pass — no prompt re-render, §7.2).

### 7.2 Masking strategy (single-pass, token-search — **revised, Gemini review**)
The earlier "double-render" (processing the prompt prefix a second time *with the image*) would run heavy image patching twice per batch inside the DataLoader → **CPU bottleneck, GPU starvation on the T4** (Gemini fix 2). Replaced with a **single-pass** approach:

1. Render + process each example **once** (images patched once).
2. Find the assistant completion boundary by **searching `input_ids` for the assistant-turn header** rather than re-tokenizing. Qwen2 chat format opens each turn with the single special token `<|im_start|>`; the conversation is `system / user / assistant`, so the **last `<|im_start|>`** opens the assistant turn. The header `"<|im_start|>assistant\n"` is fixed-length in tokens → completion starts right after it.
3. Mask `labels[:completion_start] = -100`. Pure integer-tensor ops, zero extra image work.

This also directly enables the **truncation guard** (Gemini fix 3): after locating the boundary, verify the row still ends in `<|im_end|>` (completion terminated, not sliced off). `max_length` raised **1536 → 2048** so image tokens (700+ at high `max_pixels`) + prompt + completion fit without silently truncating the answer tail.

### 7.3 Reference implementation
```python
import torch, warnings
from qwen_vl_utils import process_vision_info

class ForgeSightCollator:
    """Qwen2-VL SFT collator. Single-pass image processing; masks everything but the
    assistant completion by locating the assistant header span in input_ids (no double-render)."""

    def __init__(self, processor, build_messages, max_length=2048):
        self.processor = processor
        self.build_messages = build_messages           # record -> messages (incl. assistant target)
        self.max_length = max_length
        tok = processor.tokenizer
        tok.padding_side = "right"
        self.pad_id      = tok.pad_token_id
        self.im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id   = tok.convert_tokens_to_ids("<|im_end|>")
        # fixed token span that follows "<|im_start|>" to open the assistant turn
        self.assistant_tail = tok("assistant\n", add_special_tokens=False).input_ids
        self.image_token_id = getattr(processor, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = tok.convert_tokens_to_ids("<|image_pad|>")

    def __call__(self, records):
        texts, image_lists = [], []
        for rec in records:
            msgs = self.build_messages(rec)             # includes assistant target
            texts.append(self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False))
            imgs, _ = process_vision_info(msgs)         # images processed ONCE
            image_lists.append(imgs)

        # single processor pass -> input_ids, attention_mask, pixel_values, image_grid_thw
        batch = self.processor(
            text=texts, images=image_lists,
            padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        n_tail = len(self.assistant_tail)

        for i in range(input_ids.size(0)):
            row = input_ids[i]
            im_starts = (row == self.im_start_id).nonzero(as_tuple=True)[0]
            if len(im_starts) == 0:
                raise ValueError(f"[collator] sample {i}: no <|im_start|> — chat-template mismatch")
            comp_start = int(im_starts[-1].item()) + 1 + n_tail   # skip "<|im_start|>assistant\n"
            labels[i, :comp_start] = -100                          # mask system+user+image+header

            # --- truncation guard (fix 3): completion must exist AND terminate in <|im_end|> ---
            non_pad = (row != self.pad_id).nonzero(as_tuple=True)[0]
            last_real = int(non_pad[-1].item()) if len(non_pad) else -1
            if comp_start > last_real or row[last_real] != self.im_end_id:
                warnings.warn(
                    f"[collator] sample {i}: completion truncated/empty "
                    f"(comp_start={comp_start}, last_real={last_real}). "
                    f"Raise max_length or lower max_pixels.")

        labels[batch["attention_mask"] == 0] = -100                # pad tokens
        labels[input_ids == self.image_token_id] = -100            # belt-and-suspenders (already in prompt)
        batch["labels"] = labels
        return batch
```

> **Why last `<|im_start|>` is safe:** the assistant JSON never contains `<|im_start|>` (it's a single special token, not producible from JSON text), and right-padding puts pad tokens *after* the completion, so the search is unambiguous. If a system message is ever dropped, the "last im_start = assistant" invariant still holds (assistant is always the final turn). The smoke gate (§7.4) verifies this empirically.

### 7.4 `scripts/smoke_collator.py` — acceptance gate (runs on M3 CPU)
Assert, on hand-made records — **2 tampered, 2 clean (varied lengths), + 1 deliberately over-long** to trip the guard:
- `input_ids.shape == attention_mask.shape == labels.shape`.
- `pixel_values` present and non-empty; `image_grid_thw` present.
- Processor called **once** per batch (fix 2): patch a counter / assert `process_vision_info` runs N times for N records, not 2N.
- For every normal row: **at least one** label `!= -100`, and `tokenizer.decode(labels[i][labels[i]!=-100])` **exactly equals** the assistant target string — the single most important assertion; proves loss is on the answer only, and that the native `<|box_start|>...` tokens survive round-trip.
- **Zero** image-placeholder tokens have a label `!= -100`.
- No label `!= -100` at a padding position.
- The over-long sample **emits the truncation `warnings.warn`** (fix 3 verified); normal samples emit none.

**Do not proceed to training until every assertion passes.** This gate is what makes the Kaggle run boring instead of a debugging nightmare.

---

## 8. Model + QLoRA (`model.py`)

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

def load_processor():
    # min_pixels/max_pixels cap visual tokens -> controls seq len & VRAM on T4
    return AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=256*28*28, max_pixels=768*28*28)

def load_model_for_training(use_4bit=True, attn="sdpa"):   # attn="sdpa" on T4 (D8)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    ) if use_4bit else None
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16,
        attn_implementation=attn, device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],   # LLM blocks; vision tower frozen
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model
```

Notes:
- **Vision tower frozen** (LoRA on LLM proj layers only) — cheaper, stable, standard for VLM QLoRA. Document as a deliberate choice; unfreezing the merger is a possible ablation if time.
- `min_pixels/max_pixels` is the main VRAM/seq-length lever on T4 — tune down if OOM.
- `attn="sdpa"` per D8. Switch to `"flash_attention_2"` **only** on Ampere.

---

## 9. SFT training (`train_sft.py` + `configs/sft.yaml`)

TRL `SFTTrainer` with the custom collator. Key config:

```yaml
# configs/sft.yaml
output_dir: /kaggle/working/adapters/sft
per_device_train_batch_size: 1
gradient_accumulation_steps: 8          # effective batch 8 (per GPU)
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
num_train_epochs: 2                     # start 2; watch val loss for over/underfit
bf16: true
gradient_checkpointing: true            # essential for T4 VRAM
logging_steps: 10
eval_strategy: steps
eval_steps: 100
save_strategy: steps
save_steps: 100
save_total_limit: 2
report_to: none
max_grad_norm: 0.3
optim: paged_adamw_8bit                 # bnb paged optimizer -> fits T4
```

Trainer wiring:
- Pass `data_collator=ForgeSightCollator(...)`, `train_dataset`, `eval_dataset`.
- Set `dataset_kwargs={"skip_prepare_dataset": True}` and `remove_unused_columns=False` so TRL does **not** try to tokenize/strip our records — the collator owns all tensor construction.
- **Single-GPU first.** Get one T4 working end-to-end before attempting 2×T4 (DDP via `accelerate`). Multi-GPU is a *nice-to-have*, not required; note it in README if achieved.
- Log train/val loss → the curves are a required deliverable (they live in the Kaggle notebook).

**First Kaggle action = overfit 8 examples** (tiny subset, 50 steps) and confirm train loss → ~0 and the model reproduces target JSON on those 8. This proves the *entire* pipeline (collator + masking + model + loss) before spending real compute.

---

## 10. ForgeBench — evaluation (`eval/`)

### 10.1 Prediction collection (`forgebench.py`)
Run the SFT model over the **test split** with `generate()` (greedy, low `max_new_tokens` ~128). Parse each output via `schema.parse_prediction`. Unparseable → recorded as a structured failure (counts against detection).

### 10.2 Metrics (`metrics.py`) — the D7 split
**Detection (tampered vs clean), over ALL test examples:**
- Confusion matrix; **Precision, Recall, F1** on the "tampered" positive class.
- **Wilson 95% CI** on recall and precision (small-N honest bars — reuse from KickoffAI/PatchLoop).
- **Parse-failure rate** reported separately.

**Localization (IoU), over TRUE-POSITIVE detections only:**
- Convert predicted 0–1000 box → pixels (test-image dims) → **IoU** vs GT pixel box.
- **IoU@0.5 hit-rate** (fraction of TPs with IoU ≥ 0.5) — the headline localization number.
- **Mean IoU** on TPs; IoU histogram.
- Report localization **broken down by `tamper_type`** — this per-type table is the interview centerpiece (shows which forgeries are visually vs semantically detectable).

**Comparison / significance:**
- Baseline = zero-shot Qwen2-VL-2B (no fine-tune) with the same system prompt. Same eval.
- **McNemar test** on per-example detection correctness (fine-tuned vs baseline) → p-value for "fine-tuning helped."
- Report Δ F1, Δ IoU@0.5 with the baseline side-by-side.

### 10.3 Report (`report.py`)
Emits `artifacts/eval/forgebench_results.json` + plots to `artifacts/plots/`: detection confusion matrix, IoU histogram, per-tamper-type F1/IoU bar chart, loss curves. These plots go straight into the README.

---

## 11. Serving / production profile (`serve/profile.py`) — req #6

On Kaggle T4, after training:
- Load base + merged/attached LoRA in **bnb 4-bit** (D9), `attn="sdpa"`.
- Warm up, then measure over N=50 test images:
  - **Latency** p50 / p95 / mean (ms) for a full forensic verdict.
  - **Peak VRAM** (`torch.cuda.max_memory_allocated`).
  - **Throughput** (images/sec, and tokens/sec).
- Compare **fp16 vs 4-bit**: VRAM saved, latency delta, and confirm **quality parity** (re-run ForgeBench detection F1 under 4-bit — the "does quantization hurt accuracy?" table).
- **Stretch:** stand up vLLM for Qwen2-VL and report tok/s vs HF `generate()`. If brittle → document the attempt + failure mode (still a signal) and keep bnb numbers as the deliverable.

Output → `artifacts/eval/serving_profile.json` + a README table.

---

## 12. ORPO stretch (`train_orpo.py` + `configs/orpo.yaml`) — only if SFT wrapped

- **Pairs:** for each tampered example — chosen = correct `{...}` with correct box; rejected = same JSON but box **shifted/oversized** (plausible-wrong) and/or a **hallucinated reason**. For clean examples — chosen = `{tampered:false,...}`; rejected = a fabricated tamper+box.
- ORPO (reference-free, single-stage) → fits T4 without a second model. `beta≈0.1`.
- Start from the SFT adapter. Small run.
- Eval delta on ForgeBench: expect ↓ hallucinated-box rate on clean docs, ↑ localization precision. Report as an ablation (SFT vs SFT+ORPO, with CIs). **If it doesn't help, report that honestly** — a clean negative result with CIs is itself a strong signal (this is the PatchLoop discipline).

---

## 13. Build & test order (dependency-ordered, NOT time-boxed)

Each step has an explicit **gate** — do not advance until it passes. This ordering front-loads the risky, GPU-free work so Kaggle time is spent training, not debugging.

1. **Repo skeleton + `requirements-dev.txt`.** Gate: `import forgesight` works on M3.
2. **`schema.py` + `coords.py`** with unit tests (round-trip pixel↔norm, IoU on known boxes, JSON parse of good/garbage strings). Gate: `pytest` green.
3. **`data/ingest.py`** — load SROIE + FUNSD, normalize to §4 records with OCR boxes. Gate: print N records, boxes land on words when visualized.
4. **`forgery/` ops** in the D-order (digit_swap → copy_move → splice → recompress_ghost) + `pipeline.py`. Gate: `scripts/viz_sample.py` shows correct GT box on ~30 samples/op by eye.
5. **`data/build_dataset.py`** — generate balanced train/val/test with **doc-level split**. Gate: class balance + split-leakage check (no source doc in two splits); `make_manifest.py` stats look right.
6. **`data/conversation.py`** — record → messages. Gate: printed chat template looks correct; image inserted in user turn.
7. **`collator.py`** — implement §7. Gate: **`scripts/smoke_collator.py` all assertions pass on CPU** (esp. decoded unmasked span == target JSON). ← *hardest gate; the crux.*
8. **Push to GitHub. Upload `data/processed/` as Kaggle Dataset.** Gate: Kaggle notebook clones repo + attaches data + imports `forgesight`.
9. **`model.py`** on Kaggle. Gate: 4-bit Qwen2-VL-2B loads on one T4; `print_trainable_parameters` shows LoRA params only; a single forward pass runs.
10. **Overfit-8 sanity** (§9). Gate: train loss → ~0; model reproduces the 8 target JSONs. ← *proves whole pipeline before real spend.*
11. **`train_sft.py` full run** (single T4). Gate: val loss decreases and stabilizes; adapter saved; curves logged.
12. **`infer.py` + `eval/forgebench.py` + `metrics.py`** — baseline (zero-shot) and fine-tuned. Gate: `forgebench_results.json` with F1, IoU@0.5, Wilson CIs, McNemar p all populated.
13. **`serve/profile.py`** — bnb-4bit latency/VRAM/throughput + fp16-vs-4bit parity. Gate: `serving_profile.json` + README table.
14. **README** — thesis, method, ForgeBench design, results tables, plots, serving numbers, reproduction steps, honest limitations. Gate: a stranger can understand and rerun it.
15. **(Stretch) 2×T4 DDP → ORPO → vLLM**, in that priority. Each is independently droppable.

**Definition of done (minimum viable, resume-ready):** steps 1–14 complete. That already satisfies every JD requirement: QLoRA-SFT of a VLM (1–4), PEFT/LoRA + SFT (D4), custom benchmark + stat error analysis (10), document/forgery domain (5), quantization + latency/VRAM profiling (11 / req #6). Stretch items (15) are pure upside.

---

## 14. Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Collator masking wrong → model learns prompt / learns nothing | High impact | §7.4 smoke gate with decoded-span assertion; the overfit-8 test (step 10) catches any residual issue. |
| **GPU starvation** — heavy image patching done twice per batch in the DataLoader | High (was in draft) | **Fixed (Gemini review):** §7.2/7.3 single-pass — images processed once, mask found by token-search on `input_ids`, no prefix re-render. |
| **Silent completion truncation** — image tokens + prompt exceed `max_length`, JSON answer sliced off tail → loss on nothing | Medium | **Fixed (Gemini review):** `max_length` 1536→2048; §7.3 truncation guard warns if row doesn't end in `<\|im_end\|>`; §7.4 asserts the warning fires on an over-long sample. |
| Raw-int boxes bypass grounding heads → poor localization | Medium (was in draft) | **Fixed (Gemini review):** D2/D3 use native `<\|box_start\|>` grounding-token string; `schema.parse_box` regex-extracts coords for IoU. |
| Kaggle HF version drift breaks imports/TRL args | High | Pinned versions §3.2; record working set in README; overfit-8 early surfaces it. |
| T4 OOM | Medium | 4-bit + gradient_checkpointing + paged_adamw_8bit + lower `max_pixels`; batch 1 × grad-accum. |
| FA2 assumed, silently unavailable | Medium | D8: SDPA is the default; FA2 gated behind Ampere check. |
| Synthetic forgeries too easy/unrealistic → inflated metrics | Medium | Mix a "hard" op (recompress_ghost, signal-level); report per-type so easy vs hard is visible, not hidden; eyeball step 4. |
| Bad GT boxes poison IoU | Medium | viz gate (step 4) mandatory before build. |
| vLLM multimodal brittle on T4 | Low (it's stretch) | bnb `generate()` is the committed path; vLLM failure is documented, not blocking. |
| Data leakage across split | Medium | Doc-level split + explicit leakage assertion (step 5) — same discipline as prior projects. |

---

## 15. Resume framing (write bullets against these, backfill numbers from ForgeBench)

- Fine-tuned **Qwen2-VL-2B with QLoRA (4-bit NF4)** for document **tamper detection + spatial grounding**, emitting native 0–1000 bounding boxes + structured fraud verdicts; **[F1]** detection / **[IoU@0.5]** localization on a self-built benchmark, **+[Δ]** over zero-shot (**McNemar p=[·]**).
- Built a **synthetic adversarial forgery pipeline** (copy-move / splice / digit-swap / recompression-ghost) generating **[N]** labeled `(image, GT-box, verdict)` triples from SROIE/FUNSD, with doc-level splitting to eliminate leakage.
- Designed **ForgeBench**, a custom benchmark separating detection (F1 + Wilson CIs) from localization (IoU@0.5, per-tamper-type), with McNemar significance vs baseline.
- Profiled **4-bit vs fp16 serving** (latency p50/p95, peak VRAM, throughput) on T4 with SDPA attention, demonstrating **[X]% VRAM reduction at [Y] quality parity**.

---

## 16. What to hand the fresh implementation chat

Paste this file. Then first instruction: *"Start at §13 step 1. Build steps 1–7 on the M3 (no GPU). Do not write any Kaggle/training code until `scripts/smoke_collator.py` passes every assertion in §7.4. Implement `collator.py` (§7) as the priority deliverable and prove it before anything else."*
