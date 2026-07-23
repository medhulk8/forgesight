"""Load Qwen2-VL-2B + QLoRA config + processor (§8). GPU-only path (4-bit is
CUDA/bitsandbytes-only, D10) — exercised on Kaggle, not the M3.

`load_processor` is CPU-safe (used by the M3 collator smoke test too). The 4-bit
model load lives behind `load_model_for_training` and requires CUDA.
"""

from __future__ import annotations

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# min/max visual tokens — the main VRAM/seq-length AND SPEED lever on T4 (§8).
# 512 (up from a 384 first pass): the 1-epoch/384 SFT collapsed to always-"clean"
# (underfit of the positive class); 512 restores tamper signal — esp. the single
# altered glyph in digit_swap — at ~1.3x the 384 token cost. Train + inference MUST
# use the same value (this constant feeds both load_processor paths).
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

# LoRA targets. The first SFT collapsed to always-"clean" because LoRA touched
# only the LLM while the VISION tower stayed frozen — so the trainable params
# never adapted to forgery artifacts and had no separable signal (SESSIONS
# 2026-07-23). Fix: LoRA on the LLM projections AND the vision blocks' attention
# (qkv/proj), plus fully training the vision->LLM merger (modules_to_save).
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",     # LLM attention
    "gate_proj", "up_proj", "down_proj",        # LLM MLP
]
# Qwen2-VL vision blocks: VisionAttention.qkv (fused Q/K/V Linear). We target ONLY
# "qkv", NOT "proj": in transformers 4.47.1 modeling_qwen2_vl, PatchEmbed.proj is a
# nn.Conv3d (L287) and the "proj" suffix would match it — LoRA can't wrap a Conv3d
# and get_peft_model raises. "qkv" is unique to vision attention (Linear, no LLM or
# Conv3d collision); the vision out-projection is covered by training the merger.
VISION_LORA_MODULES = ["qkv"]
# vision->LLM bridge (Qwen2VLPatchMerger); trained fully (kept in fp16, see below).
MERGER_MODULE = "visual.merger"


def load_processor(min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=min_pixels, max_pixels=max_pixels)


def lora_config(r=16, lora_alpha=32, lora_dropout=0.05, vision=True):
    """Build the LoRA config (import-safe on M3 — peft has no CUDA requirement).

    vision=True adds the vision blocks' attention to the LoRA targets and marks
    the merger for full training (modules_to_save). vision=False reproduces the
    original LLM-only config (the collapsed baseline — kept for ablation).
    """
    from peft import LoraConfig

    targets = list(LORA_TARGET_MODULES)
    save = None
    if vision:
        targets += VISION_LORA_MODULES
        save = [MERGER_MODULE]

    return LoraConfig(
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
        target_modules=targets, modules_to_save=save, task_type="CAUSAL_LM",
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
    # Skip quantizing the merger: modules_to_save trains it, and a trainable copy
    # of a 4-bit module is unsafe — keep it in fp16.
    bnb = None
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["merger"],
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
