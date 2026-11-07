"""Resolve repository paths for directly executable developer tools."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
(ROOT / "artifacts").mkdir(exist_ok=True)
