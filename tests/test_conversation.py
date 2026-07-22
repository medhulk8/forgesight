"""Offline tests for data/conversation.py (pure, no processor / no disk read)."""

import os

from forgesight import schema
from forgesight.data import conversation as conv


def _tampered():
    return {
        "image_path": "images/train/sroie-train-0__digit_swap.png",
        "width": 1000, "height": 1000, "tampered": True,
        "field": "total", "tamper_type": "digit_swap",
        "box_pixel": [612, 843, 745, 878],
        "reason": "digit altered.",
    }


def _clean():
    return {
        "image_path": "images/train/funsd-train-1__clean.png",
        "width": 762, "height": 1000, "tampered": False,
        "field": None, "tamper_type": None, "box_pixel": None,
        "reason": "No inconsistencies detected.",
    }


def test_image_uri_absolute_and_file_scheme():
    uri = conv.image_uri(_tampered(), data_root="/data/processed")
    assert uri.startswith("file://")
    assert uri == "file://" + os.path.abspath("/data/processed/images/train/sroie-train-0__digit_swap.png")


def test_build_messages_training_structure():
    rec = _tampered()
    msgs = conv.build_messages(rec, data_root="/root", include_target=True)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"][0]["text"] == conv.SYSTEM_PROMPT
    # user turn: image dict (file:// uri) + instruction text
    user = msgs[1]["content"]
    assert user[0]["type"] == "image" and user[0]["image"].startswith("file://")
    assert user[1]["text"] == conv.USER_INSTRUCTION
    # assistant target is exactly schema.to_target_json (single source of truth)
    assert msgs[2]["content"][0]["text"] == schema.to_target_json(rec)


def test_build_messages_inference_omits_assistant():
    msgs = conv.build_messages(_tampered(), include_target=False)
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_build_messages_clean_target():
    rec = _clean()
    msgs = conv.build_messages(rec, include_target=True)
    target = msgs[2]["content"][0]["text"]
    assert '"tampered": false' in target and '"box": null' in target


def test_system_prompt_mentions_native_box_tokens():
    assert "<|box_start|>" in conv.SYSTEM_PROMPT and "0-1000" in conv.SYSTEM_PROMPT
