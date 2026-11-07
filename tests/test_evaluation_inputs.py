"""Verify evaluation refuses duplicate images, wrong splits and leakage."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluate_dataset import evaluate, read_manifest


class EvaluationInputTest(unittest.TestCase):
    def test_metrics_pipeline_with_synthetic_predictions(self):
        import medical223_core as core

        before = core.MEDICAL223_MODEL_PATH, core.MEDICAL223_MODEL_ARCH
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            rows = ["path,label,split"]
            for label in ["normal", "pneumonia", "tuberculosis"]:
                (root / f"{label}.png").write_bytes(label.encode())
                rows.append(f"{label}.png,{label},test")
            (root / "test.csv").write_text("\n".join(rows))
            (root / "model.pth").write_bytes(b"synthetic-model-placeholder")
            with patch(
                "medical223_core.run_medical223_primary_model",
                return_value={"scores": {"normal": 0.7, "pneumonia": 0.2, "tuberculosis": 0.1}},
            ):
                result = evaluate(
                    root / "test.csv", root / "model.pth", "densenet121", root / "out"
                )
            self.assertAlmostEqual(result["accuracy"], 1 / 3)
            self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [1, 0, 0], [1, 0, 0]])
            self.assertAlmostEqual(result["macro_ovr_auc"], 0.5)
            self.assertFalse(result["train_overlap_checked"])
            self.assertEqual((core.MEDICAL223_MODEL_PATH, core.MEDICAL223_MODEL_ARCH), before)

    def test_duplicate_content_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            (root / "one.png").write_bytes(b"identical")
            (root / "two.png").write_bytes(b"identical")
            (root / "test.csv").write_text(
                "path,label,split\none.png,normal,test\ntwo.png,pneumonia,test\n"
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_manifest(root / "test.csv", "test")

    def test_wrong_split_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            (root / "test.csv").write_text("path,label,split\none.png,normal,train\n")
            with self.assertRaisesRegex(ValueError, "split"):
                read_manifest(root / "test.csv", "test")

    def test_train_test_content_overlap_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            (root / "one.png").write_bytes(b"same-patient-image")
            for split in ["test", "train"]:
                (root / f"{split}.csv").write_text(f"path,label,split\none.png,normal,{split}\n")
            with self.assertRaisesRegex(ValueError, "overlap"):
                evaluate(
                    root / "test.csv",
                    root / "nonexistent-model.pth",
                    "densenet121",
                    root / "out",
                    root / "train.csv",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
