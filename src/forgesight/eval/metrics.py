"""ForgeBench metrics (§10.2) — the D7 split: detection (F1) SEPARATE from
localization (IoU@0.5). Pure functions, unit-tested on the M3.

Prediction record shape (produced by forgebench.py), one per test example:
    {
      "doc_id": str, "source": str, "tamper_type": str|None,
      "true_tampered": bool,
      "true_box_pixel": [x1,y1,x2,y2]|None, "width": int, "height": int,
      "pred": <parse_prediction dict>|None,   # None = unparseable model output
      "correct_det": bool,                     # filled by detection_metrics
    }
"""

from __future__ import annotations

import math

from .. import coords


# --------------------------------------------------------------------------- #
# Wilson score interval (small-N honest CIs)
# --------------------------------------------------------------------------- #
def wilson_ci(k, n, z=1.96):
    """95% Wilson CI for a binomial proportion k/n. Returns (point, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


# --------------------------------------------------------------------------- #
# Detection (tampered-vs-clean), over ALL examples
# --------------------------------------------------------------------------- #
def _pred_tampered(rec):
    """A parse failure (pred is None) counts as 'not flagged' → against detection."""
    p = rec.get("pred")
    return bool(p["tampered"]) if p is not None else False


def detection_metrics(records):
    """Precision/Recall/F1 on the 'tampered' positive class + Wilson CIs +
    parse-failure rate. Mutates each record with `correct_det`."""
    tp = fp = fn = tn = 0
    parse_fail = 0
    for r in records:
        if r.get("pred") is None:
            parse_fail += 1
        yt = bool(r["true_tampered"])
        yp = _pred_tampered(r)
        r["correct_det"] = (yt == yp)
        if yt and yp:
            tp += 1
        elif yp and not yt:
            fp += 1
        elif yt and not yp:
            fn += 1
        else:
            tn += 1

    prec_p, prec_lo, prec_hi = wilson_ci(tp, tp + fp)
    rec_p, rec_lo, rec_hi = wilson_ci(tp, tp + fn)
    f1 = (2 * prec_p * rec_p / (prec_p + rec_p)) if (prec_p + rec_p) > 0 else 0.0
    n = len(records)
    return {
        "n": n,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": prec_p, "precision_ci": [prec_lo, prec_hi],
        "recall": rec_p, "recall_ci": [rec_lo, rec_hi],
        "f1": f1,
        "accuracy": (tp + tn) / n if n else 0.0,
        "parse_failure_rate": parse_fail / n if n else 0.0,
        "parse_failures": parse_fail,
    }


# --------------------------------------------------------------------------- #
# Localization (IoU), over TRUE-POSITIVE detections only
# --------------------------------------------------------------------------- #
def _iou_for_record(rec):
    """IoU in 0..1000 normalized space (resolution-independent). None if the
    prediction has no parseable box."""
    p = rec.get("pred")
    if p is None or not p.get("box_norm"):
        return None
    gt_norm = coords.pixel_to_norm(rec["true_box_pixel"], rec["width"], rec["height"])
    return coords.iou(p["box_norm"], gt_norm)


def localization_metrics(records, iou_thresh=0.5):
    """IoU@0.5 hit-rate + mean IoU over TRUE POSITIVES (true tampered AND predicted
    tampered), plus a per-tamper-type breakdown. A TP with an unparseable/missing
    box scores IoU 0 (localization miss)."""
    tps = [r for r in records if r["true_tampered"] and _pred_tampered(r)]
    ious = [(_iou_for_record(r) or 0.0) for r in tps]
    hits = sum(1 for v in ious if v >= iou_thresh)
    n_tp = len(tps)

    _, lo, hi = wilson_ci(hits, n_tp)
    by_type = {}
    for r, v in zip(tps, ious):
        t = r.get("tamper_type") or "unknown"
        by_type.setdefault(t, []).append(v)
    per_type = {
        t: {"n": len(vs), "mean_iou": sum(vs) / len(vs),
            "iou@0.5": sum(1 for v in vs if v >= iou_thresh) / len(vs)}
        for t, vs in by_type.items()
    }
    return {
        "n_true_positive": n_tp,
        "mean_iou": (sum(ious) / n_tp) if n_tp else 0.0,
        "iou@0.5_hit_rate": (hits / n_tp) if n_tp else 0.0,
        "iou@0.5_ci": [lo, hi],
        "by_tamper_type": per_type,
    }


# --------------------------------------------------------------------------- #
# McNemar (fine-tuned vs baseline detection correctness)
# --------------------------------------------------------------------------- #
def mcnemar(correct_a, correct_b):
    """Exact McNemar test on paired per-example correctness (two bool lists).
    Returns discordant counts + two-sided exact-binomial p-value.

    b = A right / B wrong; c = A wrong / B right. Under H0 (no difference), b ~
    Binomial(b+c, 0.5). Exact binomial is honest at the small discordant counts
    typical of a few-hundred-example benchmark."""
    if len(correct_a) != len(correct_b):
        raise ValueError("mcnemar: mismatched lengths")
    b = sum(1 for a, c in zip(correct_a, correct_b) if a and not c)
    c = sum(1 for a, cc in zip(correct_a, correct_b) if (not a) and cc)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    from scipy.stats import binomtest

    p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return {"b": b, "c": c, "n_discordant": n, "p_value": float(p)}


# --------------------------------------------------------------------------- #
# full report
# --------------------------------------------------------------------------- #
def compute_report(records, baseline_records=None):
    """Assemble the ForgeBench report. If baseline_records given (same order /
    doc_ids), add a McNemar test + deltas vs baseline."""
    det = detection_metrics(records)
    loc = localization_metrics(records)
    report = {"detection": det, "localization": loc}

    if baseline_records is not None:
        base_det = detection_metrics(baseline_records)
        base_loc = localization_metrics(baseline_records)
        mc = mcnemar([r["correct_det"] for r in records],
                     [r["correct_det"] for r in baseline_records])
        report["baseline"] = {"detection": base_det, "localization": base_loc}
        report["mcnemar"] = mc
        report["delta"] = {
            "f1": det["f1"] - base_det["f1"],
            "iou@0.5_hit_rate": loc["iou@0.5_hit_rate"] - base_loc["iou@0.5_hit_rate"],
        }
    return report
