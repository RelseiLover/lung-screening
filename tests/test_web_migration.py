"""Regression checks for the Web adapter; no cloud API calls."""

import ast
import io
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import medical223_core as core
import presentation
import web_api
from fastapi.testclient import TestClient
from medical223_langgraph import run_medical223
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


class WebMigrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(web_api.app)

    def test_presentation_matches_original_functions(self):
        source = (ROOT / "backend/streamlit_app.py").read_text(encoding="utf-8")
        namespace = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and hasattr(presentation, node.name):
                exec(
                    compile(ast.Module(body=[node], type_ignores=[]), "<original>", "exec"),
                    namespace,
                )
                migrated = ast.parse(Path(presentation.__file__).read_text(encoding="utf-8"))
                other = next(
                    n
                    for n in migrated.body
                    if isinstance(n, ast.FunctionDef) and n.name == node.name
                )
                self.assertEqual(
                    ast.dump(node, include_attributes=False),
                    ast.dump(other, include_attributes=False),
                )

    def test_build_and_api_contract(self):
        self.assertTrue(self.client.get("/api/health").json()["frontend_ready"])
        html = self.client.get("/")
        self.assertEqual(html.status_code, 200)
        self.assertIn('id="root"', html.text)
        for data in ("{}", "not json", '{"symptom":42}'):
            self.assertEqual(
                self.client.post("/api/screen", data={"answers": data}).status_code, 422
            )

    def test_upload_validation_and_cleanup_on_error(self):
        params = {"answers": json.dumps({"symptom": "演示症状"})}
        self.assertEqual(
            self.client.post(
                "/api/screen", data=params, files={"image": ("x.txt", b"x")}
            ).status_code,
            415,
        )
        self.assertEqual(
            self.client.post(
                "/api/screen", data=params, files={"image": ("x.jpg", b"bad-image")}
            ).status_code,
            415,
        )
        image = io.BytesIO()
        Image.new("RGB", (32, 32)).save(image, format="PNG")
        paths = []

        def failing(symptom, path):
            self.assertTrue(Path(path).is_file())
            paths.append(Path(path))
            raise RuntimeError("internal failure must not reach client")

        with patch.object(web_api, "run_medical223", failing):
            response = self.client.post(
                "/api/screen", data=params, files={"image": ("x.png", image.getvalue())}
            )
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("internal failure", response.text)
        self.assertTrue(paths)
        self.assertTrue(all(not path.exists() for path in paths))

    @unittest.skipUnless(
        os.getenv("RUN_MODEL_TESTS") == "1", "Enable with python scripts/test.py --models"
    )
    def test_actual_outputs_match_original_workflow(self):
        cases = [
            ("无明显不适，体检胸片筛查。", "01_clear_normal_low_risk.jpg"),
            ("发热3天，咳嗽咳黄痰，伴胸痛。", "02_clear_pneumonia_acute_infection.jpg"),
            ("咳嗽1个月，夜间盗汗，体重下降，偶有咯血。", "03_clear_tuberculosis_chronic.jpg"),
            ("咳嗽1周，黄痰，低热，活动后有些气促。", "04_boundary_pneumonia_review.jpg"),
            ("咳嗽2周以上，乏力，夜间出汗，食欲下降。", "05_boundary_tuberculosis_review.jpg"),
            ("高热4天，咳嗽咳黄痰，胸痛气促明显。", "06_normal_conflict_case.jpg"),
            ("发热3天，咳嗽。", None),
        ]
        results = []
        with (
            patch.object(core, "call_openai_compatible", return_value=None),
            patch.object(core, "run_hf_tb", return_value=None),
            patch.object(core, "run_hf_pneumonia", return_value=None),
        ):
            for symptom, filename in cases:
                with self.subTest(filename=filename):
                    params = {"answers": json.dumps({"symptom": symptom}, ensure_ascii=False)}
                    image = ROOT / "examples/images" / filename if filename else None
                    original = run_medical223(symptom, str(image) if image else "")
                    if image:
                        self.assertTrue(
                            original.image_model_name.startswith("medical223:"),
                            "Model parity must use actual CNN weights, not heuristic fallback",
                        )
                    expected = web_api.present_result(original, symptom)
                    if image:
                        with image.open("rb") as stream:
                            response = self.client.post(
                                "/api/screen",
                                data=params,
                                files={"image": (image.name, stream, "image/jpeg")},
                            )
                    else:
                        response = self.client.post("/api/screen", data=params)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json(), expected)
                    results.append(
                        {
                            "image": filename,
                            "result": original.screening_result,
                            "route": original.route_decision,
                            "parity": "passed",
                            "image_model": original.image_model_name,
                        }
                    )
        report_path = ROOT / "artifacts/frontend-parity.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        report.update(
            cases=results,
            scope="Real local CNN and original workflow. Cloud LLM and optional remote specialists disabled for reproducibility.",
        )
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main(verbosity=2)
