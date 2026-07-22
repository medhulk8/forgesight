# ForgeSight

QLoRA fine-tuning of **Qwen2-VL-2B-Instruct** for document **tamper detection**
with **spatial grounding** — given a document image, emit structured JSON
`{tampered, field, box, reason}` where `box` uses Qwen2-VL's native 0–1000
grounding-token format.

See [plan.md](plan.md) for the full build specification (single source of truth).

> **Status:** under construction. §13 steps 1–2 complete (repo skeleton, `schema.py`,
> `coords.py`, unit tests). README written last — this is a placeholder.

## Dev (M3 / CPU — no GPU, no bitsandbytes)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest          # or: pip install -r requirements-dev.txt
python -c "import forgesight"   # skeleton gate
pytest                          # schema/coords gate
```

Training runs on Kaggle 2×T4 — see `requirements-kaggle.txt` (added in a later step).
