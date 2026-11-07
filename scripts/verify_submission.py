"""Repeatable local verification. Never sends symptoms to paid cloud APIs."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from _bootstrap import ROOT as ROOT


def worker():
    from runtime_config import load_environment

    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / ".env"
        config.write_text('DEEPSEEK_MODEL="file-value"\nHF_HUB_OFFLINE=0\n', encoding="utf-8")
        saved = {k: os.environ.get(k) for k in ("DEEPSEEK_MODEL", "HF_HUB_OFFLINE")}
        try:
            os.environ["DEEPSEEK_MODEL"] = "system-value"
            os.environ.pop("HF_HUB_OFFLINE", None)
            load_environment(config)
            assert os.environ["DEEPSEEK_MODEL"] == "system-value"
            assert os.environ["HF_HUB_OFFLINE"] == "0"
            config.write_text("UNSUPPORTED=bad\n", encoding="utf-8")
            try:
                load_environment(config)
            except ValueError:
                pass
            else:
                raise AssertionError("Unknown settings were accepted")
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    print("CONFIG OK", flush=True)
    import check_environment

    assert not check_environment.main()
    import medical223_core as core

    # Disable cloud calls even if a private .env or legacy JSON exists.
    core.call_openai_compatible = lambda prompt: None
    result = {"configuration": "passed", "environment": "passed", "models": {}}
    image = str(ROOT / "examples/images/01_clear_normal_low_risk.jpg")
    for name, arch in [
        ("efficientnetv2s_best.pth", "tf_efficientnetv2_s.in21k_ft_in1k"),
        ("densenet121_best.pth", "densenet121"),
    ]:
        core.MEDICAL223_MODEL_PATH = str(ROOT / "models" / name)
        core.MEDICAL223_MODEL_ARCH = arch
        output = core.run_medical223_primary_model(image)
        assert output and abs(sum(output["scores"].values()) - 1) < 1e-4, name
        result["models"][name] = output["scores"]
        print("MODEL OK " + name, flush=True)
    core.MEDICAL223_MODEL_PATH = str(ROOT / "models/efficientnetv2s_best.pth")
    core.MEDICAL223_MODEL_ARCH = "tf_efficientnetv2_s.in21k_ft_in1k"
    from medical223_langgraph import run_medical223

    state = run_medical223("无明显不适，体检胸片筛查。", image)
    assert state.screening_result and state.agent_trace
    assert not state.used_llm
    result["graph"] = {
        "status": "passed",
        "result": state.screening_result,
        "used_llm": state.used_llm,
    }
    from streamlit.testing.v1 import AppTest

    page = AppTest.from_file(str(ROOT / "backend/streamlit_app.py"), default_timeout=90).run()
    assert not page.exception, str(page.exception)
    result["page_render"] = "passed"
    result["scope"] = "Local inference and workflow; no live API or clinical accuracy evaluation"
    result["timestamp_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (ROOT / "artifacts/runtime-verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker()
    else:
        env = os.environ.copy()
        for key in list(env):
            if key.startswith(("DEEPSEEK_", "OPENAI_", "MEDICAL223_")):
                env.pop(key)
        env["HF_HUB_OFFLINE"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        raise SystemExit(
            subprocess.call(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"], env=env
            )
        )
