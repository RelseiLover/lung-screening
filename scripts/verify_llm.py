"""Offline failure checks plus an explicitly requested single live synthetic test.

Credentials are only read into memory and are never copied into reports.
"""

import argparse
import io
import json
import os
import time
import urllib.error
from pathlib import Path
from unittest.mock import patch

import medical223_core as core
from _bootstrap import ROOT as ROOT


def failure_checks():
    checks = {}
    config = {
        "api_key": "synthetic-test-key",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    }
    cases = {
        "timeout": TimeoutError("synthetic timeout"),
        "connection_reset": ConnectionResetError("synthetic reset"),
        "network_unreachable": urllib.error.URLError("synthetic network failure"),
        "http_401": urllib.error.HTTPError(
            config["base_url"], 401, "Unauthorized", {}, io.BytesIO(b"{}")
        ),
    }
    with (
        patch.dict(
            os.environ,
            {k: v for k, v in os.environ.items() if not k.startswith(("DEEPSEEK_", "OPENAI_"))},
            clear=True,
        ),
        patch.object(core, "load_local_deepseek_config", return_value=config),
    ):
        for name, failure in cases.items():
            with patch.object(core.urllib.request, "urlopen", side_effect=failure):
                result = core.diagnosis_agent(
                    core.MedicalState(symptom="虚构功能测试：普通体检，无个人信息")
                )
            assert not result.used_llm and result.screening_result, name
            checks[name] = "local_fallback_passed"
        for name, body in [
            ("invalid_json", b"not-json"),
            ("missing_choices", b"{}"),
            ("invalid_model_content", b'{"choices":[{"message":{"content":"not structured"}}]}'),
        ]:
            with patch.object(core.urllib.request, "urlopen", return_value=io.BytesIO(body)):
                result = core.diagnosis_agent(
                    core.MedicalState(symptom="虚构功能测试：普通体检，无个人信息")
                )
            assert not result.used_llm and result.screening_result, name
            checks[name] = "local_fallback_passed"
    return checks


def live_check(config_path):
    if config_path:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    else:
        config = core.load_local_deepseek_config()
    key = os.getenv("DEEPSEEK_API_KEY") or config.get("api_key", "")
    if not key:
        return {"status": "not_run", "reason": "No private credential available"}
    base_url = os.getenv("DEEPSEEK_BASE_URL") or config.get(
        "base_url", "https://api.deepseek.com/v1"
    )
    # Restrict this live validation to the existing, requested DeepSeek endpoint.
    if base_url.rstrip("/") != "https://api.deepseek.com/v1":
        return {
            "status": "not_run",
            "reason": "Configured endpoint differs from DeepSeek; no request sent",
        }
    settings = {
        "DEEPSEEK_API_KEY": key,
        "DEEPSEEK_BASE_URL": base_url,
        "DEEPSEEK_MODEL": config.get("model", "deepseek-chat"),
    }
    captured = {}
    actual_urlopen = core.urllib.request.urlopen

    def observed(request, **kwargs):
        try:
            response = actual_urlopen(request, **kwargs)
            captured["http_status"] = response.status
            return response
        except urllib.error.HTTPError as exc:
            captured["http_status"] = exc.code
            raise
        except OSError:
            captured["network_error"] = True
            raise

    start = time.perf_counter()
    with (
        patch.dict(os.environ, settings),
        patch.object(core.urllib.request, "urlopen", side_effect=observed),
    ):
        state = core.MedicalState(symptom="虚构演示病例，无个人信息：体检，无明显不适。")
        state.image_scores = {"正常倾向": 0.99, "感染性异常": 0.005, "慢性异常": 0.005}
        state.image_model_name = "synthetic-evidence-for-api-test"
        state.image_result = "虚构模型证据，仅用于接口测试"
        state = core.diagnosis_agent(state)
    return dict(
        status="passed" if state.used_llm else "fallback",
        used_llm=state.used_llm,
        elapsed_seconds=round(time.perf_counter() - start, 2),
        synthetic_input=True,
        credential_copied=False,
        **captured,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", action="store_true", help="Make one real API request using a synthetic case"
    )
    parser.add_argument("--config", help="Private JSON configuration path; never copied")
    args = parser.parse_args()
    output = ROOT / "artifacts/deepseek-validation.json"
    result = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    result["offline_failure_checks"] = failure_checks()
    if args.live:
        result["live"] = live_check(args.config)
    else:
        result.setdefault("live", {"status": "not_requested"})
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
