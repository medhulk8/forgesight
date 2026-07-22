"""ForgeSight — QLoRA fine-tuning of Qwen2-VL-2B for document tamper detection
with spatial grounding.

Public surface (M3-buildable, GPU-free):
    schema  — canonical JSON target (de)serialization + validation
    coords  — pixel <-> 0..1000 normalized box conversions + IoU

GPU-only concerns (4-bit load, training) live behind model.py / train_*.py and
are exercised only on Kaggle 2xT4.
"""

from . import coords, schema

__all__ = ["schema", "coords"]
__version__ = "0.0.1"
