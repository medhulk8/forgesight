"""Offline tests for eval/metrics.py (§10.2): detection F1 + Wilson CI,
localization IoU@0.5 + per-type, McNemar."""

import pytest

from forgesight.eval import metrics


def _rec(tampered, pred_tampered, box_norm=None, gt=None, ttype=None, parse_fail=False):
    """Build a metric record. Uses a 1000x1000 image so pixel==norm for IoU."""
    pred = None if parse_fail else {"tampered": pred_tampered, "box_norm": box_norm}
    return {
        "doc_id": "d", "source": "s", "tamper_type": ttype,
        "true_tampered": tampered, "true_box_pixel": gt,
        "width": 1000, "height": 1000, "pred": pred,
    }


# --------------------------------------------------------------------------- #
# wilson_ci
# --------------------------------------------------------------------------- #
def test_wilson_ci_basic():
    p, lo, hi = metrics.wilson_ci(8, 10)
    assert p == pytest.approx(0.8)
    assert 0 <= lo < p < hi <= 1


def test_wilson_ci_edges():
    assert metrics.wilson_ci(0, 0) == (0.0, 0.0, 0.0)
    _, lo, hi = metrics.wilson_ci(10, 10)
    assert hi == pytest.approx(1.0) and lo < 1.0     # never a degenerate [1,1]


# --------------------------------------------------------------------------- #
# detection_metrics
# --------------------------------------------------------------------------- #
def test_detection_confusion_and_f1():
    recs = [
        _rec(True, True), _rec(True, True),      # 2 TP
        _rec(True, False),                       # 1 FN
        _rec(False, True),                       # 1 FP
        _rec(False, False), _rec(False, False),  # 2 TN
    ]
    d = metrics.detection_metrics(recs)
    assert d["confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert d["precision"] == pytest.approx(2 / 3)
    assert d["recall"] == pytest.approx(2 / 3)
    assert d["f1"] == pytest.approx(2 / 3)
    assert all("correct_det" in r for r in recs)


def test_detection_parse_failure_counts_against():
    # parse failure on a tampered doc -> treated as not-flagged -> FN + counted in rate
    recs = [_rec(True, None, parse_fail=True), _rec(False, False)]
    d = metrics.detection_metrics(recs)
    assert d["confusion"]["fn"] == 1
    assert d["parse_failures"] == 1
    assert d["parse_failure_rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# localization_metrics
# --------------------------------------------------------------------------- #
def test_localization_iou_and_hit_rate():
    recs = [
        # TP, perfect box -> IoU 1.0
        _rec(True, True, box_norm=[100, 100, 200, 200], gt=[100, 100, 200, 200],
             ttype="digit_swap"),
        # TP, half-overlap -> IoU 1/3 (<0.5 miss)
        _rec(True, True, box_norm=[100, 100, 200, 200], gt=[150, 100, 250, 200],
             ttype="splice"),
        # TP, missing box (parseable verdict, no box) -> IoU 0
        _rec(True, True, box_norm=None, gt=[10, 10, 50, 50], ttype="copy_move"),
        # not a TP (clean) -> ignored by localization
        _rec(False, False),
    ]
    metrics.detection_metrics(recs)  # not required, but mirrors real flow
    loc = metrics.localization_metrics(recs)
    assert loc["n_true_positive"] == 3
    assert loc["iou@0.5_hit_rate"] == pytest.approx(1 / 3)   # only the perfect one hits
    assert loc["mean_iou"] == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)
    assert set(loc["by_tamper_type"]) == {"digit_swap", "splice", "copy_move"}
    assert loc["by_tamper_type"]["digit_swap"]["iou@0.5"] == 1.0
    assert loc["by_tamper_type"]["copy_move"]["mean_iou"] == 0.0


# --------------------------------------------------------------------------- #
# mcnemar
# --------------------------------------------------------------------------- #
def test_mcnemar_all_equal_p1():
    a = [True, False, True]
    r = metrics.mcnemar(a, a)
    assert r["b"] == 0 and r["c"] == 0 and r["p_value"] == 1.0


def test_mcnemar_discordant():
    # A right/B wrong 8 times, A wrong/B right 1 time -> significant-ish
    a = [True] * 8 + [False] + [True]
    b = [False] * 8 + [True] + [True]
    r = metrics.mcnemar(a, b)
    assert r["b"] == 8 and r["c"] == 1
    assert 0.0 <= r["p_value"] <= 1.0 and r["p_value"] < 0.05


def test_mcnemar_length_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.mcnemar([True], [True, False])


# --------------------------------------------------------------------------- #
# compute_report with baseline
# --------------------------------------------------------------------------- #
def test_compute_report_with_baseline():
    ft = [_rec(True, True, [100, 100, 200, 200], [100, 100, 200, 200], "digit_swap"),
          _rec(False, False)]
    base = [_rec(True, False), _rec(False, False)]   # baseline misses the tamper
    rep = metrics.compute_report(ft, baseline_records=base)
    assert "detection" in rep and "localization" in rep
    assert "mcnemar" in rep and "baseline" in rep
    assert rep["delta"]["f1"] > 0     # fine-tuned beats baseline on F1
