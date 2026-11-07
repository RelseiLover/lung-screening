"""Load literal local settings without executing shell commands."""

import os
from pathlib import Path

ALLOWED = {
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "HF_HUB_OFFLINE",
    "HF_TB_REPO",
    "HF_PNEUMONIA_REPO",
    "MEDICAL223_REPO_DIR",
    "MEDICAL223_MODEL_PATH",
    "MEDICAL223_MODEL_ARCH",
    "MEDICAL223_IMAGE_SIZE",
}


def load_environment(path=None):
    """OS settings win. Values are literal; blank values preserve defaults."""
    path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not sep or key not in ALLOWED:
                raise ValueError(f"Invalid configuration key: {key}")
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if value:
                os.environ.setdefault(key, value)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
