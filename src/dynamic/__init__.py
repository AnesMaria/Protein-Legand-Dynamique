"""Trajectory-based protein-ligand affinity; static baseline modules remain available."""
import os
from pathlib import Path

cache = Path(__file__).resolve().parents[2] / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache))
