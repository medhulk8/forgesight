"""Canonical JSON target schema + (de)serialization + validation (§4).

One record shape flows through the whole system. The model's *target* is a JSON
envelope whose `box` value is Qwen2-VL's native grounding-token string (D2/D3):

    {"tampered": true, "field": "total_amount",
     "box": "<|box_start|>(612,843),(745,878)<|box_end|>", "reason": "..."}

Clean target:

    {"tampered": false, "field": null, "box": null, "reason": "No inconsistencies detected."}

The `<|box_start|>` / `<|box_end|>` markers are *single special tokens* to the
tokenizer but literal text in the decoded string. `parse_box` therefore matches
the `(x1,y1),(x2,y2)` core on the decoded string and does NOT depend on either
token surviving — robust if a token is dropped (§4 regex note).
"""

from __future__ import annotations

import hashlib
import json
import re

from . import coords

BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"

# Clean-verdict reason templates. A single constant clean reason (the original
# "No inconsistencies detected." repeated on every clean example) is a gradient
# sink: it becomes the most-frequent target sequence by far and the model
# collapses onto always-"clean" (see SESSIONS 2026-07-23). We diversify the clean
# reason deterministically per image (stable md5 pick) so no single string
# dominates, while keeping targets reproducible. Chosen at render time, so no
# dataset rebuild/re-upload is needed.
CLEAN_REASONS = (
    "No inconsistencies detected.",
    "No visual evidence of tampering found.",
    "Document layout and font structures are consistent.",
    "All text fields appear authentic.",
    "Fonts, spacing, and alignment are uniform throughout.",
    "No signs of digit or field alteration.",
    "Ink, stroke weight, and baselines are consistent across the document.",
    "No copy-paste or splice artifacts detected.",
    "The document appears original and unmodified.",
    "Field values are consistent with the surrounding text.",
    "No compression or edge artifacts found around any field.",
    "Nothing indicates the document has been edited.",
)


def clean_reason_for(key: str) -> str:
    """Deterministic clean-reason pick from CLEAN_REASONS, stable across machines.

    Keyed on the image path (md5, not Python's salted hash) so a given clean
    image always maps to the same reason — reproducible, yet spread across the
    template set to break the single-constant-target attractor.
    """
    h = int(hashlib.md5((key or "").encode("utf-8")).hexdigest(), 16)
    return CLEAN_REASONS[h % len(CLEAN_REASONS)]

# match the (x1,y1),(x2,y2) core regardless of surrounding tokens (§4).
_BOX_RE = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")

# keys of the model target envelope, in stable output order.
_TARGET_KEYS = ("tampered", "field", "box", "reason")


# --------------------------------------------------------------------------- #
# box (de)serialization
# --------------------------------------------------------------------------- #
def format_box(norm_box) -> str:
    """Four 0..1000 ints -> native grounding-token string.

    >>> format_box([612, 843, 745, 878])
    '<|box_start|>(612,843),(745,878)<|box_end|>'
    """
    if norm_box is None or len(norm_box) != 4:
        raise ValueError(f"format_box: expected 4 ints, got {norm_box!r}")
    x1, y1, x2, y2 = (int(v) for v in norm_box)
    return f"{BOX_START}({x1},{y1}),({x2},{y2}){BOX_END}"


def parse_box(box_str):
    """Native box string -> [x1,y1,x2,y2] ints, or None if no 4-int match.

    Tolerant of the special tokens being present, stripped, or partially dropped
    — it only needs the (x1,y1),(x2,y2) core. Returning None counts as a
    localization miss at eval (§4).
    """
    if not isinstance(box_str, str):
        return None
    m = _BOX_RE.search(box_str)
    if m is None:
        return None
    return [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))]


# --------------------------------------------------------------------------- #
# record -> target JSON
# --------------------------------------------------------------------------- #
def to_target_json(record) -> str:
    """Intermediate record (§4) -> assistant target JSON string.

    Pixel box -> 0..1000 via coords.pixel_to_norm -> format_box. Clean records
    emit box=null. Stable key order, compact single-line json.dumps.
    """
    if record.get("tampered"):
        box_pixel = record.get("box_pixel")
        if box_pixel is None:
            raise ValueError("to_target_json: tampered record missing box_pixel")
        norm = coords.pixel_to_norm(box_pixel, record["width"], record["height"])
        obj = {
            "tampered": True,
            "field": record.get("field"),
            "box": format_box(norm),
            "reason": record.get("reason", ""),
        }
    else:
        obj = {
            "tampered": False,
            "field": None,
            "box": None,
            # diversify per-image to break the constant-target gradient sink
            "reason": clean_reason_for(record.get("image_path")),
        }
    # dict literal above already fixes key order (py3.7+ preserves insertion).
    return json.dumps(obj, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# prediction parsing
# --------------------------------------------------------------------------- #
def _extract_first_json_object(text: str):
    """Return the first balanced {...} substring, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def parse_prediction(text):
    """Robustly parse a model completion into a validated dict, or None.

    Steps: extract the first balanced {...} block -> json.loads -> validate keys
    and types -> parse_box on the `box` field. Returns None on unparseable JSON
    or type violations (counts as a detection miss in eval, §4).

    On success returns the envelope with an added `box_norm` key = parsed
    [x1,y1,x2,y2] 0..1000 ints (or None). The raw `box` string is preserved.
    """
    if not isinstance(text, str):
        return None
    blob = _extract_first_json_object(text)
    if blob is None:
        return None
    try:
        obj = json.loads(blob)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    tampered = obj.get("tampered")
    field = obj.get("field")
    box = obj.get("box")
    reason = obj.get("reason")

    # type contract
    if not isinstance(tampered, bool):
        return None
    if field is not None and not isinstance(field, str):
        return None
    if box is not None and not isinstance(box, str):
        return None
    if reason is not None and not isinstance(reason, str):
        return None

    return {
        "tampered": tampered,
        "field": field,
        "box": box,
        "box_norm": parse_box(box) if box is not None else None,
        "reason": reason,
    }


# --------------------------------------------------------------------------- #
# record validation (used by data build to reject malformed entries early)
# --------------------------------------------------------------------------- #
_RECORD_KEYS = (
    "image_path",
    "width",
    "height",
    "tampered",
    "field",
    "tamper_type",
    "box_pixel",
    "reason",
)


def validate_record(record) -> bool:
    """Validate an intermediate record (§4). Returns True, or raises ValueError.

    Enforces the tampered/clean invariants:
      tampered -> box_pixel is 4 numbers, field + tamper_type are strings.
      clean    -> box_pixel, field, tamper_type are all None.
    """
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")
    for k in _RECORD_KEYS:
        if k not in record:
            raise ValueError(f"record missing key: {k}")

    if not isinstance(record["image_path"], str) or not record["image_path"]:
        raise ValueError("image_path must be a non-empty string")
    for dim in ("width", "height"):
        if not isinstance(record[dim], int) or record[dim] <= 0:
            raise ValueError(f"{dim} must be a positive int")
    if not isinstance(record["tampered"], bool):
        raise ValueError("tampered must be a bool")
    if not isinstance(record["reason"], str):
        raise ValueError("reason must be a string")

    if record["tampered"]:
        box = record["box_pixel"]
        if not (isinstance(box, (list, tuple)) and len(box) == 4
                and all(isinstance(v, (int, float)) for v in box)):
            raise ValueError("tampered record: box_pixel must be 4 numbers")
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("tampered record: box_pixel must have x2>x1, y2>y1")
        if not isinstance(record["field"], str) or not record["field"]:
            raise ValueError("tampered record: field must be a non-empty string")
        if not isinstance(record["tamper_type"], str) or not record["tamper_type"]:
            raise ValueError("tampered record: tamper_type must be a non-empty string")
    else:
        for k in ("field", "tamper_type", "box_pixel"):
            if record[k] is not None:
                raise ValueError(f"clean record: {k} must be null")

    return True
