"""Load Qwen2-VL-2B + QLoRA config + processor (§8). GPU-only path (4-bit is
CUDA/bitsandbytes-only, D10) — exercised on Kaggle, not the M3.

`load_processor` is CPU-safe (used by the M3 collator smoke test too). The 4-bit
model load lives behind `load_model_for_training` and requires CUDA.
"""

from __future__ import annotations

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# min/max visual tokens — the main VRAM/seq-length AND SPEED lever on T4 (§8).
# Lowered from 768 -> 384 max: receipts/forms are readable at ~300k px and this
# ~halves the image-token count → ~2x faster training with negligible field-OCR loss.
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 384 * 28 * 28

# LoRA on the LLM projection layers only; vision tower frozen (cheaper, stable,
# standard for VLM QLoRA). Unfreezing the merger is a possible ablation (§8).
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_processor(min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=min_pixels, max_pixels=max_pixels)


def lora_config(r=16, lora_alpha=32, lora_dropout=0.05):
    """Build the LoRA config (import-safe on M3 — peft has no CUDA requirement)."""
    from peft import LoraConfig

    return LoraConfig(
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
    )


def load_model_for_training(use_4bit=True, attn="sdpa", lora=True,
                            r=16, lora_alpha=32, lora_dropout=0.05,
                            device_map=None):
    """Load Qwen2-VL-2B (4-bit NF4 by default) + attach LoRA. CUDA-only when use_4bit.

    attn="sdpa" on T4 (Turing, no FA2 — D8). Switch to "flash_attention_2" only on
    Ampere (sm_80+).

    device_map defaults to SINGLE GPU ({"":0}), NOT "auto". A 2B model in 4-bit is
    ~1.5 GB and fits one T4; sharding it across 2 T4s made Qwen2-VL's autoregressive
    generate() emit wrong tokens (KV-cache/hidden states crossing the device split)
    even though the teacher-forced forward was 100% correct. Single-GPU load fixes
    generation and training both (multi-GPU DDP is a stretch item, §9, not this).
    """
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    if device_map is None:
        device_map = {"": 0}

    # T4 (Turing) has fp16 tensor cores but NOT bf16 → use fp16 compute for speed.
    bnb = None
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.float16,
        attn_implementation=attn, device_map=device_map,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    if lora:
        model = get_peft_model(model, lora_config(r, lora_alpha, lora_dropout))
        model.print_trainable_parameters()

    return model
