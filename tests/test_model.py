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


def test_vision_lora_modules():
    # ONLY "qkv": "proj" would collide with PatchEmbed.proj (a Conv3d) which LoRA
    # cannot wrap. "qkv" is unique to the vision attention Linear.
    assert model.VISION_LORA_MODULES == ["qkv"]


def test_lora_config_vision_on():
    cfg = model.lora_config(r=16, lora_alpha=32, lora_dropout=0.05)   # vision=True default
    assert cfg.r == 16 and cfg.lora_alpha == 32
    assert cfg.task_type == "CAUSAL_LM" and cfg.bias == "none"
    assert set(cfg.target_modules) == set(model.LORA_TARGET_MODULES) | set(model.VISION_LORA_MODULES)
    assert cfg.modules_to_save == [model.MERGER_MODULE]


def test_lora_config_vision_off_matches_legacy():
    cfg = model.lora_config(vision=False)
    assert set(cfg.target_modules) == set(model.LORA_TARGET_MODULES)
    assert cfg.modules_to_save is None


def test_train_sft_import_safe():
    # must import without pulling trl/transformers/torch (Kaggle-only, deferred).
    import forgesight.train_sft as t
    assert hasattr(t, "build_trainer") and hasattr(t, "load_config")


def test_pixel_bounds_constants():
    # 512 balances tamper signal vs T4 cost (§8 seq-length lever); max must stay a
    # 28x28-patch multiple. Train + inference read the same constant.
    assert model.MIN_PIXELS == 128 * 28 * 28 and model.MAX_PIXELS == 512 * 28 * 28
    assert model.MAX_PIXELS % (28 * 28) == 0 and model.MIN_PIXELS < model.MAX_PIXELS
