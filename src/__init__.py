"""Classical/quantum affinity baseline, intentionally without an SAE."""
import os
from pathlib import Path

# Writable default in cloud tasks; existing user configuration takes precedence.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
