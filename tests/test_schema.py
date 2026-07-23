"""Unit tests for schema.py (§4): box format/parse (clean/garbage/token-dropped),
target JSON build, prediction parsing (valid JSON / unparseable), record validation.
"""

import json

import pytest

from forgesight import schema


# --------------------------------------------------------------------------- #
# format_box / parse_box
# --------------------------------------------------------------------------- #
def test_format_box():
    assert schema.format_box([612, 843, 745, 878]) == \
        "<|box_start|>(612,843),(745,878)<|box_end|>"


def test_format_box_roundtrip():
    box = [1, 2, 3, 4]
    assert schema.parse_box(schema.format_box(box)) == box


def test_format_box_bad_len_raises():
    with pytest.raises(ValueError):
        schema.format_box([1, 2, 3])


def test_parse_box_clean():
    s = "<|box_start|>(612,843),(745,878)<|box_end|>"
    assert schema.parse_box(s) == [612, 843, 745, 878]


def test_parse_box_token_dropped_start():
    # leading token stripped — core still parses.
    assert schema.parse_box("(612,843),(745,878)<|box_end|>") == [612, 843, 745, 878]


def test_parse_box_token_dropped_end():
    assert schema.parse_box("<|box_start|>(612,843),(745,878)") == [612, 843, 745, 878]


def test_parse_box_both_tokens_dropped():
    assert schema.parse_box("(1,2),(3,4)") == [1, 2, 3, 4]


def test_parse_box_with_whitespace():
    assert schema.parse_box("<|box_start|>( 10 , 20 ) , ( 30 , 40 )<|box_end|>") == \
        [10, 20, 30, 40]


def test_parse_box_garbage_returns_none():
    for junk in ["", "no box here", "the total looks wrong", "[612,843,745,878]",
                 "(612,843)", "(a,b),(c,d)"]:
        assert schema.parse_box(junk) is None, junk


def test_parse_box_non_string_returns_none():
    assert schema.parse_box(None) is None
    assert schema.parse_box([1, 2, 3, 4]) is None


def test_parse_box_first_match_wins():
    s = "junk (1,2),(3,4) tail (5,6),(7,8)"
    assert schema.parse_box(s) == [1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# to_target_json
# --------------------------------------------------------------------------- #
def _tampered_record():
    return {
        "image_path": "data/processed/images/000123.png",
        "width": 1000,
        "height": 1000,
        "tampered": True,
        "field": "total_amount",
        "tamper_type": "digit_swap",
        "box_pixel": [612, 843, 745, 878],
        "reason": "digit stroke weight differs from the rest of the row.",
    }


def _clean_record():
    return {
        "image_path": "data/processed/images/000999.png",
        "width": 1654,
        "height": 2339,
        "tampered": False,
        "field": None,
        "tamper_type": None,
        "box_pixel": None,
        "reason": "No inconsistencies detected.",
    }


def test_to_target_json_tampered():
    out = schema.to_target_json(_tampered_record())
    obj = json.loads(out)
    # 1000x1000 image => pixel == norm.
    assert obj["tampered"] is True
    assert obj["field"] == "total_amount"
    assert obj["box"] == "<|box_start|>(612,843),(745,878)<|box_end|>"
    assert schema.parse_box(obj["box"]) == [612, 843, 745, 878]
    # stable key order
    assert list(obj.keys()) == ["tampered", "field", "box", "reason"]


def test_to_target_json_clean():
    obj = json.loads(schema.to_target_json(_clean_record()))
    assert obj["tampered"] is False and obj["field"] is None and obj["box"] is None
    # reason is diversified per-image (breaks the constant-target sink) but must
    # come from the fixed template set and be deterministic for a given image.
    assert obj["reason"] in schema.CLEAN_REASONS
    assert obj["reason"] == json.loads(schema.to_target_json(_clean_record()))["reason"]


def test_clean_reason_deterministic_and_spread():
    # same key -> same reason; different keys -> not all identical (sink broken).
    assert schema.clean_reason_for("a.png") == schema.clean_reason_for("a.png")
    picks = {schema.clean_reason_for(f"img{i}.png") for i in range(50)}
    assert len(picks) > 1


def test_to_target_json_tampered_missing_box_raises():
    rec = _tampered_record()
    rec["box_pixel"] = None
    with pytest.raises(ValueError):
        schema.to_target_json(rec)


# --------------------------------------------------------------------------- #
# parse_prediction
# --------------------------------------------------------------------------- #
def test_parse_prediction_valid_tampered():
    text = ('{"tampered": true, "field": "total_amount", '
            '"box": "<|box_start|>(612,843),(745,878)<|box_end|>", "reason": "x"}')
    p = schema.parse_prediction(text)
    assert p is not None
    assert p["tampered"] is True
    assert p["field"] == "total_amount"
    assert p["box_norm"] == [612, 843, 745, 878]
    assert p["reason"] == "x"


def test_parse_prediction_valid_clean():
    text = '{"tampered": false, "field": null, "box": null, "reason": "clean"}'
    p = schema.parse_prediction(text)
    assert p["tampered"] is False
    assert p["box"] is None
    assert p["box_norm"] is None


def test_parse_prediction_extracts_from_chatter():
    text = ('Sure! Here is my verdict:\n'
            '{"tampered": true, "field": "date", '
            '"box": "<|box_start|>(1,2),(3,4)<|box_end|>", "reason": "y"}\n'
            'Hope that helps.')
    p = schema.parse_prediction(text)
    assert p is not None and p["box_norm"] == [1, 2, 3, 4]


def test_parse_prediction_first_object_only():
    text = ('{"tampered": false, "field": null, "box": null, "reason": "a"} '
            '{"tampered": true, "field": "x", "box": null, "reason": "b"}')
    p = schema.parse_prediction(text)
    assert p["tampered"] is False and p["reason"] == "a"


def test_parse_prediction_box_present_but_unparseable():
    # valid envelope, box string has no coord core -> box_norm None, still a dict.
    text = '{"tampered": true, "field": "x", "box": "somewhere top-left", "reason": "z"}'
    p = schema.parse_prediction(text)
    assert p is not None
    assert p["box"] == "somewhere top-left"
    assert p["box_norm"] is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "the document looks fine to me",
        "{not valid json at all",
        '{"tampered": true, "field": "x"',        # truncated / unbalanced
        '["tampered", "field"]',                  # not an object
        "42",
    ],
)
def test_parse_prediction_unparseable_returns_none(text):
    assert schema.parse_prediction(text) is None


@pytest.mark.parametrize(
    "obj",
    [
        {"tampered": "yes", "field": None, "box": None, "reason": "r"},   # tampered not bool
        {"tampered": True, "field": 5, "box": None, "reason": "r"},        # field wrong type
        {"tampered": True, "field": "x", "box": 123, "reason": "r"},       # box wrong type
        {"tampered": True, "field": "x", "box": None, "reason": 9},        # reason wrong type
        {"field": "x", "box": None, "reason": "r"},                        # missing tampered
    ],
)
def test_parse_prediction_type_violations_return_none(obj):
    assert schema.parse_prediction(json.dumps(obj)) is None


def test_parse_prediction_non_string_returns_none():
    assert schema.parse_prediction(None) is None


# --------------------------------------------------------------------------- #
# validate_record
# --------------------------------------------------------------------------- #
def test_validate_record_tampered_ok():
    assert schema.validate_record(_tampered_record()) is True


def test_validate_record_clean_ok():
    assert schema.validate_record(_clean_record()) is True


def test_validate_record_roundtrips_to_target():
    # a valid record must serialize without error.
    for rec in (_tampered_record(), _clean_record()):
        schema.validate_record(rec)
        schema.to_target_json(rec)


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("width"),                       # missing key
    lambda r: r.__setitem__("width", 0),            # non-positive dim
    lambda r: r.__setitem__("tampered", "true"),    # tampered not bool
    lambda r: r.__setitem__("box_pixel", [1, 2, 3]),   # bad box len
    lambda r: r.__setitem__("box_pixel", [50, 50, 10, 10]),  # x2<=x1
    lambda r: r.__setitem__("field", None),         # tampered w/ null field
    lambda r: r.__setitem__("tamper_type", ""),     # empty tamper_type
])
def test_validate_record_tampered_bad_raises(mutate):
    rec = _tampered_record()
    mutate(rec)
    with pytest.raises(ValueError):
        schema.validate_record(rec)


@pytest.mark.parametrize("mutate", [
    lambda r: r.__setitem__("field", "total"),       # clean must have null field
    lambda r: r.__setitem__("box_pixel", [1, 2, 3, 4]),  # clean must have null box
    lambda r: r.__setitem__("tamper_type", "splice"),
])
def test_validate_record_clean_bad_raises(mutate):
    rec = _clean_record()
    mutate(rec)
    with pytest.raises(ValueError):
        schema.validate_record(rec)


def test_validate_record_non_dict_raises():
    with pytest.raises(ValueError):
        schema.validate_record(["not", "a", "dict"])
