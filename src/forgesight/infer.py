"""Single-image inference → parsed JSON verdict (§10.1). GPU-side; heavy imports
deferred so the module imports on the M3.

Used by eval/forgebench.py (fine-tuned + zero-shot baseline) and for ad-hoc demos.
"""

from __future__ import annotations

from . import model as model_mod
from . import schema
from .data import conversation


def load_for_inference(adapter_dir=None, use_4bit=True, attn="sdpa"):
    """Load base Qwen2-VL-2B (4-bit) + optionally attach a trained LoRA adapter.
    adapter_dir=None → zero-shot baseline (§10.2). Single GPU (device_map={"":0})."""
    import torch
    from peft import PeftModel
    from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    processor = model_mod.load_processor()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    ) if use_4bit else None
    net = Qwen2VLForConditionalGeneration.from_pretrained(
        model_mod.MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16,
        attn_implementation=attn, device_map={"": 0},
    )
    if adapter_dir:
        net = PeftModel.from_pretrained(net, adapter_dir)
    net.eval()
    return net, processor


def generate_text(net, processor, record, data_root=None, max_new_tokens=128):
    """Run greedy generation for one record; return the raw decoded completion."""
    import torch
    from qwen_vl_utils import process_vision_info

    msgs = conversation.build_messages(record, data_root=data_root, include_target=False)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs)
    inputs = processor(text=[text], images=[imgs], return_tensors="pt")
    inputs = inputs.to(next(net.parameters()).device)
    with torch.no_grad():
        out = net.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


def predict(net, processor, record, data_root=None, max_new_tokens=128):
    """Full single-image forensic verdict → parse_prediction dict (or None if the
    model output was unparseable)."""
    text = generate_text(net, processor, record, data_root, max_new_tokens)
    return schema.parse_prediction(text)
