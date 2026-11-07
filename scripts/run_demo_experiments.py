"""Functional model comparison and no-retrieval ablation on six curated demos.

This intentionally does not output clinical accuracy or claim independent data.
"""

import json
import time
from unittest.mock import patch

import medical223_core as core
from _bootstrap import ROOT as ROOT
from medical223_langgraph import run_medical223

CASES = [
    ("01_clear_normal_low_risk.jpg", "无明显不适，体检胸片筛查。"),
    ("02_clear_pneumonia_acute_infection.jpg", "发热3天，咳嗽咳黄痰，伴胸痛。"),
    ("03_clear_tuberculosis_chronic.jpg", "咳嗽1个月，夜间盗汗，体重下降，偶有咯血。"),
    ("04_boundary_pneumonia_review.jpg", "咳嗽1周，黄痰，低热，活动后有些气促。"),
    ("05_boundary_tuberculosis_review.jpg", "咳嗽2周以上，乏力，夜间出汗，食欲下降。"),
    ("06_normal_conflict_case.jpg", "高热4天，咳嗽咳黄痰，胸痛气促明显。"),
]


def snapshot(state):
    return {
        key: getattr(state, key)
        for key in [
            "screening_result",
            "confidence",
            "route_decision",
            "rag_infectious_score",
            "rag_chronic_score",
            "iteration",
        ]
    }


def main():
    results = []
    original_path, original_arch = core.MEDICAL223_MODEL_PATH, core.MEDICAL223_MODEL_ARCH
    models = ROOT / "models"
    try:
        with (
            patch.object(core, "call_openai_compatible", return_value=None),
            patch.object(core, "run_hf_tb", return_value=None),
            patch.object(core, "run_hf_pneumonia", return_value=None),
        ):
            for filename, symptom in CASES:
                image = str(ROOT / "examples/images" / filename)
                row = {"image": filename, "symptom": symptom, "models": {}}
                for weight, arch in [
                    ("efficientnetv2s_best.pth", "tf_efficientnetv2_s.in21k_ft_in1k"),
                    ("densenet121_best.pth", "densenet121"),
                ]:
                    core.MEDICAL223_MODEL_PATH = str(models / weight)
                    core.MEDICAL223_MODEL_ARCH = arch
                    start = time.perf_counter()
                    output = core.run_medical223_primary_model(image)
                    assert output is not None, filename
                    row["models"][arch] = {
                        "scores": output["scores"],
                        "top_class": max(output["scores"], key=output["scores"].get),
                        "elapsed_seconds_including_possible_initial_load": round(
                            time.perf_counter() - start, 3
                        ),
                    }
                core.MEDICAL223_MODEL_PATH = str(models / "efficientnetv2s_best.pth")
                core.MEDICAL223_MODEL_ARCH = "tf_efficientnetv2_s.in21k_ft_in1k"
                row["with_retrieval"] = snapshot(run_medical223(symptom, image))
                with patch.object(core, "retrieve_medical_knowledge", return_value=[]):
                    row["without_retrieved_documents"] = snapshot(run_medical223(symptom, image))
                row["result_changed"] = (
                    row["with_retrieval"]["screening_result"]
                    != row["without_retrieved_documents"]["screening_result"]
                )
                results.append(row)
                print("Completed", filename, flush=True)
    finally:
        core.MEDICAL223_MODEL_PATH, core.MEDICAL223_MODEL_ARCH = original_path, original_arch
    report = {
        "purpose": "Functional comparison on six deliberately selected demo images; not independent clinical evaluation.",
        "ablation_scope": "Return no retrieved documents; symptom rules, fallback knowledge and downstream guards remain enabled. Not a full removal of every RAG-associated rule.",
        "cloud_llm_enabled": False,
        "remote_specialists_enabled": False,
        "cases": results,
    }
    (ROOT / "artifacts/demo-ablation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Functional experiment report saved", flush=True)


if __name__ == "__main__":
    main()
