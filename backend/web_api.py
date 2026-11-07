"""Serve the React build and delegate screening to the unchanged Python workflow."""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from medical223_langgraph import run_medical223
from PIL import Image, UnidentifiedImageError
from presentation import (
    build_structured_symptom,
    display_model_name,
    display_route_name,
    result_tone,
    sanitize_trace,
)
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "frontend/dist"
MAX_IMAGE_BYTES = 200 * 1024 * 1024
MODEL_LOCK = threading.Lock()
app = FastAPI(title="肺部异常辅助筛查 API", docs_url=None, redoc_url=None)


class Answers(BaseModel):
    symptom: str = ""
    duration_value: str = "未说明"
    fever_value: str = "未说明"
    sputum_value: str = "未说明"
    sputum_color: str = "未说明"
    chest_value: str = "未说明"
    tb_value: str = "未说明"


def present_result(state, enriched_symptom):
    """Expose the original displayed fields, without local temp paths."""
    result = {
        name: getattr(state, name)
        for name in (
            "screening_result",
            "confidence",
            "image_scores",
            "rationale",
            "differential",
            "check_suggestion",
            "review_note",
            "used_llm",
            "iteration",
            "rag_infectious_score",
            "rag_chronic_score",
            "rag_route_note",
            "retrieved_knowledge",
            "infectious_specialist_score",
            "infectious_specialist_note",
            "chronic_specialist_score",
            "chronic_specialist_note",
        )
    }
    result.update(
        enriched_symptom=enriched_symptom,
        route_decision=display_route_name(state.route_decision),
        route_reason=sanitize_trace(state.route_reason),
        agent_trace=[sanitize_trace(item) for item in state.agent_trace],
        infectious_specialist_source=display_model_name(state.infectious_specialist_source),
        chronic_specialist_source=display_model_name(state.chronic_specialist_source),
    )
    color, background, tag = result_tone(state.screening_result)
    result["tone"] = dict(color=color, background=background, tag=tag)
    result["structured_data"] = {
        "screening_result": state.screening_result,
        "confidence": state.confidence,
        "image_model": display_model_name(state.image_model_name),
        "image_scores": state.image_scores,
        "route_decision": display_route_name(state.route_decision),
        "rag_infectious_score": state.rag_infectious_score,
        "rag_chronic_score": state.rag_chronic_score,
        "infectious_specialist_score": state.infectious_specialist_score,
        "infectious_specialist_model": display_model_name(state.infectious_specialist_source),
        "chronic_specialist_score": state.chronic_specialist_score,
        "chronic_specialist_model": display_model_name(state.chronic_specialist_source),
        "used_llm": state.used_llm,
        "iteration": state.iteration,
    }
    return result


@app.get("/api/health")
def health():
    return {"status": "ok", "frontend_ready": (DIST / "index.html").is_file()}


@app.post("/api/screen")
def screen(answers: str = Form(...), image: UploadFile | None = File(None)):
    temp_path = None
    try:
        try:
            values = Answers.model_validate_json(answers)
        except (ValidationError, json.JSONDecodeError):
            raise HTTPException(422, "问诊数据格式无效。") from None
        enriched = build_structured_symptom(**values.model_dump())
        if not enriched.strip():
            raise HTTPException(422, "请先选择问诊信息或补充症状描述。")
        if image is not None:
            suffix = Path(image.filename or "").suffix.lower()
            if suffix not in (".png", ".jpg", ".jpeg"):
                raise HTTPException(415, "请选择 PNG、JPG 或 JPEG 影像。")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as target:
                temp_path = Path(target.name)
                total = 0
                while chunk := image.file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_IMAGE_BYTES:
                        raise HTTPException(413, "影像文件不能超过200MB。")
                    target.write(chunk)
            try:
                with Image.open(temp_path) as uploaded:
                    if uploaded.format not in ("PNG", "JPEG"):
                        raise ValueError("Unsupported image format")
                    uploaded.verify()
            except (UnidentifiedImageError, OSError, ValueError):
                raise HTTPException(415, "影像无法读取，请上传有效的 PNG 或 JPEG 文件。") from None
        with MODEL_LOCK:
            state = run_medical223(enriched, str(temp_path) if temp_path else "")
            return present_result(state, enriched)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(500, "筛查未完成，请检查配置与输入后重试。") from None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if image is not None:
            image.file.close()


@app.get("/")
def index():
    if not (DIST / "index.html").is_file():
        raise HTTPException(503, "前端尚未构建，请运行 frontend 中的 npm run build。")
    return FileResponse(DIST / "index.html")


if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")
