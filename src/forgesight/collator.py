"""ForgeSight Qwen2-VL SFT data collator (§7, §13 step 7) — THE crux component.

Single-pass, token-search masking (§7.2/7.3, revised after peer review):
  1. Render + process each example ONCE (images patched once — no prompt re-render,
     which would double image work in the DataLoader and starve the T4 GPU).
  2. Locate the assistant completion by searching `input_ids` for the LAST
     `<|im_start|>` (the assistant turn is always last) and skipping the fixed
     `assistant\n` header token span — pure integer-tensor ops.
  3. Mask everything before the completion, all pad tokens, and all image
     placeholder tokens to -100, so loss is computed on the answer only.

Truncation guard (§7.3): image tokens (700+ at high max_pixels) + prompt +
completion must fit in max_length or the JSON answer tail is silently sliced off
(loss on nothing). Warn if a row's completion is empty or does not terminate in
`<|im_end|>`.
"""

from __future__ import annotations

import warnings

import torch
from qwen_vl_utils import process_vision_info


class ForgeSightCollator:
    def __init__(self, processor, build_messages, max_length=2048):
        self.processor = processor
        self.build_messages = build_messages          # record -> messages (incl. target)
        self.max_length = max_length

        tok = processor.tokenizer
        tok.padding_side = "right"
        # Qwen2-VL's processor can re-pad internally; force right-padding on both
        # so no internal helper defaults to left-padding and breaks the mask math.
        if hasattr(processor, "padding_side"):
            processor.padding_side = "right"

        self.pad_id = tok.pad_token_id
        self.im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        # fixed token span that follows "<|im_start|>" to open the assistant turn
        self.assistant_tail = tok("assistant\n", add_special_tokens=False).input_ids
        self.image_token_id = getattr(processor, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = tok.convert_tokens_to_ids("<|image_pad|>")

    def __call__(self, records):
        texts, image_lists = [], []
        for rec in records:
            msgs = self.build_messages(rec)            # includes assistant target
            texts.append(self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False))
            imgs, _ = process_vision_info(msgs)        # images processed ONCE
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
                raise ValueError(
                    f"[collator] sample {i}: no <|im_start|> — chat-template mismatch")
            comp_start = int(im_starts[-1].item()) + 1 + n_tail  # skip "<|im_start|>assistant\n"
            labels[i, :comp_start] = -100                        # mask system+user+image+header

            # --- truncation guard (§7.3): completion must exist AND terminate ---
            # A well-formed assistant turn ends "<|im_end|>\n", so the final token is
            # the newline, not <|im_end|>. Termination = an <|im_end|> exists *within
            # the completion span* (absent only when the answer was sliced off the tail).
            non_pad = (row != self.pad_id).nonzero(as_tuple=True)[0]
            last_real = int(non_pad[-1].item()) if len(non_pad) else -1
            completion = row[comp_start:last_real + 1]
            terminated = bool((completion == self.im_end_id).any())
            if comp_start > last_real or not terminated:
                warnings.warn(
                    f"[collator] sample {i}: completion truncated/empty "
                    f"(comp_start={comp_start}, last_real={last_real}, terminated={terminated}). "
                    f"Raise max_length or lower max_pixels.")

        labels[batch["attention_mask"] == 0] = -100              # pad tokens
        labels[input_ids == self.image_token_id] = -100          # belt-and-suspenders

        # final safety net: warn if any row has zero learnable tokens (fully masked)
        learnable_per_row = (labels != -100).sum(dim=1)
        if (learnable_per_row == 0).any():
            bad = (learnable_per_row == 0).nonzero(as_tuple=True)[0].tolist()
            warnings.warn(
                f"[collator] rows {bad} have no learnable (!=-100) label — "
                f"completion truncated away. Raise max_length or lower max_pixels.")

        batch["labels"] = labels
        return batch
