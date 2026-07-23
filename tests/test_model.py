"""Import-safe / CPU-only tests for the GPU modules. The 4-bit model load and
training run on Kaggle (gated there); here we only verify the non-CUDA surface:
constants, LoRA config, and that the modules import without side effects.
"""

from forgesight import model


def test_model_id():
    assert model.MODEL_ID == "Qwen/Qwen2-VL-2B-Instruct"


def test_lora_target_modules_are_llm_projections():
    assert model.LORA_TARGET_MODULES == [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def test_lora_config_builds():
    cfg = model.lora_config(r=16, lora_alpha=32, lora_dropout=0.05)
    assert cfg.r == 16 and cfg.lora_alpha == 32
    assert cfg.task_type == "CAUSAL_LM" and cfg.bias == "none"
    assert set(cfg.target_modules) == set(model.LORA_TARGET_MODULES)


def test_train_sft_import_safe():
    # must import without pulling trl/transformers/torch (Kaggle-only, deferred).
    import forgesight.train_sft as t
    assert hasattr(t, "build_trainer") and hasattr(t, "load_config")


def test_pixel_bounds_constants():
    assert model.MIN_PIXELS == 256 * 28 * 28 and model.MAX_PIXELS == 768 * 28 * 28
