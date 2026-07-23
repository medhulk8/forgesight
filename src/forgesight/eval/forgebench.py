"""ForgeBench prediction collection (§10.1). Run a model over the test split, parse
each output, and emit metric records for eval/metrics.py. GPU-side; heavy imports
deferred so the module imports on the M3.

CLI runs the fine-tuned adapter AND the zero-shot baseline over the test set, then
writes artifacts/eval/forgebench_results.json with the full D7 report.
"""

from __future__ import annotations

import json
import os


def _load_test(data_root):
    from datasets import load_dataset

    return load_dataset("json", data_files={"test": f"{data_root}/test.jsonl"},
                        split="test")


def collect_predictions(net, processor, test_ds, data_root, max_new_tokens=128,
                        limit=None, log_every=50):
    """Run `net` over test_ds → list of metric records (schema in eval/metrics.py)."""
    from .. import infer

    records = []
    n = len(test_ds) if limit is None else min(limit, len(test_ds))
    for i in range(n):
        rec = test_ds[i]
        pred = infer.predict(net, processor, rec, data_root=data_root,
                             max_new_tokens=max_new_tokens)
        records.append({
            "doc_id": rec["doc_id"], "source": rec["source"],
            "tamper_type": rec.get("tamper_type"),
            "true_tampered": bool(rec["tampered"]),
            "true_box_pixel": rec.get("box_pixel"),
            "width": rec["width"], "height": rec["height"],
            "pred": pred,
        })
        if (i + 1) % log_every == 0:
            print(f"  [{i + 1}/{n}] collected")
    return records


def run(adapter_dir, data_root, out_path="artifacts/eval/forgebench_results.json",
        baseline=True, limit=None):
    """Collect predictions for the fine-tuned adapter (+ zero-shot baseline),
    compute the report, and write JSON. Returns the report dict."""
    from .. import infer
    from . import metrics

    test_ds = _load_test(data_root)

    print("=> fine-tuned predictions")
    net, processor = infer.load_for_inference(adapter_dir=adapter_dir)
    ft = collect_predictions(net, processor, test_ds, data_root, limit=limit)
    del net

    base = None
    if baseline:
        print("=> zero-shot baseline predictions")
        import gc

        import torch
        gc.collect(); torch.cuda.empty_cache()
        bnet, bproc = infer.load_for_inference(adapter_dir=None)
        base = collect_predictions(bnet, bproc, test_ds, data_root, limit=limit)
        del bnet

    report = metrics.compute_report(ft, baseline_records=base)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"report": report, "predictions": _slim(ft),
                   "baseline_predictions": _slim(base) if base else None}, f, indent=2)
    print(f"wrote {out_path}")
    _print_summary(report)
    return report


def _slim(records):
    """Drop bulky/derived fields for the saved predictions dump."""
    out = []
    for r in records:
        p = r.get("pred")
        out.append({
            "doc_id": r["doc_id"], "tamper_type": r["tamper_type"],
            "true_tampered": r["true_tampered"],
            "pred_tampered": (bool(p["tampered"]) if p else None),
            "pred_box_norm": (p.get("box_norm") if p else None),
        })
    return out


def _print_summary(report):
    d = report["detection"]; l = report["localization"]
    print(f"\nDetection: F1={d['f1']:.3f} P={d['precision']:.3f} R={d['recall']:.3f} "
          f"| parse-fail={d['parse_failure_rate']:.2%}")
    print(f"Localization: IoU@0.5={l['iou@0.5_hit_rate']:.3f} meanIoU={l['mean_iou']:.3f} "
          f"(over {l['n_true_positive']} TPs)")
    for t, s in l["by_tamper_type"].items():
        print(f"  {t:12s} n={s['n']:3d} IoU@0.5={s['iou@0.5']:.3f} mean={s['mean_iou']:.3f}")
    if "mcnemar" in report:
        print(f"McNemar vs baseline: p={report['mcnemar']['p_value']:.4g} "
              f"| ΔF1={report['delta']['f1']:+.3f} ΔIoU@0.5={report['delta']['iou@0.5_hit_rate']:+.3f}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/kaggle/working/adapters/sft")
    ap.add_argument("--data-root", default="/kaggle/input/datasets/medhulkhandelwal/forgesight-data")
    ap.add_argument("--out", default="artifacts/eval/forgebench_results.json")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.adapter, args.data_root, out_path=args.out,
        baseline=not args.no_baseline, limit=args.limit)


if __name__ == "__main__":
    main()
