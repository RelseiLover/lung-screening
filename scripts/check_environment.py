"""Check dependencies, frontend assets and fully downloaded model weights."""

import importlib
import re

from _bootstrap import ROOT as ROOT


def main():
    errors = []
    for name in (
        "fastapi",
        "uvicorn",
        "multipart",
        "streamlit",
        "langgraph.graph",
        "numpy",
        "PIL",
        "sklearn",
        "torch",
        "torchvision",
        "timm",
        "transformers",
        "huggingface_hub",
        "requests",
    ):
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    for name in (
        "backend/medical223_core.py",
        "backend/medical223_langgraph.py",
        "backend/presentation.py",
        "backend/runtime_config.py",
        "backend/web_api.py",
        "frontend/dist/index.html",
        "models/efficientnetv2s_best.pth",
        "models/densenet121_best.pth",
    ):
        path = ROOT / name
        if not path.is_file() or not path.stat().st_size:
            errors.append(f"Missing or empty: {name}")
        elif path.suffix == ".pth":
            with path.open("rb") as stream:
                if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/"):
                    errors.append(f"Git LFS pointer instead of weights: {name}; run git lfs pull")
    index = ROOT / "frontend/dist/index.html"
    if index.is_file():
        for asset in re.findall(
            r'(?:src|href)="(/assets/[^"]+)"', index.read_text(encoding="utf-8")
        ):
            if not (ROOT / "frontend/dist" / asset.lstrip("/")).is_file():
                errors.append(f"Missing frontend build asset: {asset}")
    for error in errors:
        print(error)
    print("Environment check: " + ("FAILED" if errors else "OK"))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
