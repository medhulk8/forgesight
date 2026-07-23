"""TRL SFTTrainer entrypoint for QLoRA-SFT (§9). Kaggle/GPU-only — heavy imports
are deferred into functions so the module stays import-safe on the M3.

Overfit-8 sanity (step 10, the pipeline proof): `--overfit 8` trains on the first
8 train examples for a few dozen steps; train loss must fall to ~0 and the model
must reproduce those 8 target JSONs. Proves collator + masking + model + loss
before spending real compute.

Run (Kaggle):
    python -m forgesight.train_sft --config configs/sft.yaml --overfit 8
    python -m forgesight.train_sft --config configs/sft.yaml            # full run
"""

from __future__ import annotations

import argparse

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _load_splits(data_root, overfit=None):
    from datasets import load_dataset

    files = {
        "train": f"{data_root}/train.jsonl",
        "val": f"{data_root}/val.jsonl",
    }
    ds = load_dataset("json", data_files=files)
    train = ds["train"]
    val = ds["val"]
    if overfit:
        # balanced pick: half tampered, half clean — else the 5+ identical clean
        # targets memorize instantly and starve the harder unique tampered targets,
        # and greedy decode collapses to "clean" (seen with a first-N pick).
        tamp = [i for i in range(len(train)) if train[i]["tampered"]]
        clean = [i for i in range(len(train)) if not train[i]["tampered"]]
        half = overfit // 2
        idx = tamp[:half] + clean[:overfit - half]
        train = train.select(sorted(idx))
        val = None
    return train, val


def build_trainer(cfg, overfit=None):
    from trl import SFTConfig, SFTTrainer

    from . import model as model_mod
    from .collator import ForgeSightCollator
    from .data import conversation

    processor = model_mod.load_processor()
    net = model_mod.load_model_for_training(
        use_4bit=True, attn="sdpa",
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
    )
    net.config.use_cache = False  # required with gradient checkpointing

    data_root = cfg["data_root"]
    train_ds, val_ds = _load_splits(data_root, overfit=overfit)

    collator = ForgeSightCollator(
        processor,
        build_messages=lambda rec: conversation.build_messages(rec, data_root=data_root),
        max_length=cfg["max_length"],
    )

    sft_kwargs = dict(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        bf16=cfg.get("bf16", False),
        fp16=cfg.get("fp16", True),
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg["logging_steps"],
        save_total_limit=cfg["save_total_limit"],
        report_to=cfg["report_to"],
        max_grad_norm=cfg["max_grad_norm"],
        optim=cfg["optim"],
        remove_unused_columns=False,          # our records are not HF-tokenizable columns
        dataset_kwargs={"skip_prepare_dataset": True},  # collator owns all tensor construction
        max_seq_length=cfg["max_length"],
    )

    if overfit:
        # memorize 8 examples: tiny batch, many steps, no eval/checkpoints
        sft_kwargs.update(
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            num_train_epochs=1, max_steps=250, logging_steps=5,
            save_strategy="no", eval_strategy="no", warmup_ratio=0.0,
        )
    else:
        sft_kwargs.update(
            num_train_epochs=cfg["num_train_epochs"],
            eval_strategy=cfg["eval_strategy"], eval_steps=cfg["eval_steps"],
            save_strategy=cfg["save_strategy"], save_steps=cfg["save_steps"],
        )

    args = SFTConfig(**sft_kwargs)
    trainer = SFTTrainer(
        model=net, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, processing_class=processor.tokenizer,
    )
    return trainer, processor, train_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--overfit", type=int, default=None,
                    help="train on first N examples for a memorization sanity check")
    args = ap.parse_args()

    cfg = load_config(args.config)
    trainer, processor, train_ds = build_trainer(cfg, overfit=args.overfit)

    result = trainer.train()
    print("train result:", result.metrics)

    if args.overfit:
        _report_overfit(trainer, processor, train_ds, cfg)
    else:
        trainer.save_model(cfg["output_dir"])
        print("adapter saved ->", cfg["output_dir"])


def _report_overfit(trainer, processor, train_ds, cfg):
    """Generate on the overfit examples; print reproduced JSON vs target (step-10 gate)."""
    import torch

    from . import schema
    from .data import conversation

    from qwen_vl_utils import process_vision_info

    from . import coords

    model = trainer.model
    model.eval()
    device = next(model.parameters()).device
    det_ok = 0          # detection (tampered bool) correct
    ious = []           # IoU on true-positive boxes
    for i in range(len(train_ds)):
        rec = train_ds[i]
        msgs = conversation.build_messages(rec, data_root=cfg["data_root"],
                                           include_target=False)
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        inputs = processor(text=[text], images=[imgs], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        gen = processor.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

        pred = schema.parse_prediction(gen)
        det = pred is not None and bool(pred["tampered"]) == bool(rec["tampered"])
        det_ok += det
        iou = None
        if det and rec["tampered"] and pred.get("box_norm"):
            tgt_norm = coords.pixel_to_norm(rec["box_pixel"], rec["width"], rec["height"])
            iou = coords.iou(pred["box_norm"], tgt_norm)
            ious.append(iou)
        print(f"[{i}] {rec.get('tamper_type') or 'clean':16s} det={det}"
              + (f" IoU={iou:.2f}" if iou is not None else ""))
        print(f"    gen: {gen.strip()[:150]}")

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    hit50 = sum(v >= 0.5 for v in ious)
    print(f"\noverfit-{len(train_ds)}: detection {det_ok}/{len(train_ds)} correct | "
          f"boxes: mean IoU {mean_iou:.2f}, IoU@0.5 {hit50}/{len(ious)}")


if __name__ == "__main__":
    main()
