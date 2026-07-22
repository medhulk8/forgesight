"""Record → Qwen2-VL chat messages (§6, §13 step 6).

`build_messages` is the `build_messages` callable the collator (§7) invokes per
record. It is PURE + disk-agnostic: it only does path math (no image read) and
reuses `schema.to_target_json` as the single source of truth for the assistant
target JSON. The image is referenced by a `file://<abs path>` URI (NOT a PIL
object) — passing PIL objects through a DataLoader with num_workers>0 pickles
badly and leaks memory; `qwen_vl_utils.process_vision_info` loads the file
just-in-time inside the collator (Gemini review, step 6).
"""

from __future__ import annotations

import os

from .. import schema

# System prompt — native-box format (D2/D3), updated from §6.1 for the token string.
SYSTEM_PROMPT = (
    "You are a document forensics expert. Given a document image, determine "
    "whether any field has been tampered with. Respond ONLY with a compact JSON "
    "object with keys: tampered (bool), field (string or null), box (string "
    "format '<|box_start|>(x1,y1),(x2,y2)<|box_end|>' in 0-1000 normalized "
    "coordinates, or null), reason (string). If untampered, set tampered=false, "
    "field=null, box=null."
)

USER_INSTRUCTION = "Inspect this document for tampering."


def image_uri(record, data_root=None) -> str:
    """Resolve record['image_path'] to a `file://<abs>` URI (pure path math)."""
    path = record["image_path"]
    if data_root is not None:
        path = os.path.join(data_root, path)
    return "file://" + os.path.abspath(path)


def build_messages(record, data_root=None, include_target=True):
    """Return Qwen2-VL chat messages for a record.

    include_target=True  → system / user(+image) / assistant(target JSON)  [training]
    include_target=False → system / user(+image)                          [inference]
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image_uri(record, data_root)},
            {"type": "text", "text": USER_INSTRUCTION},
        ]},
    ]
    if include_target:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": schema.to_target_json(record)}],
        })
    return messages
