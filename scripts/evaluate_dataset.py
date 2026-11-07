"""Evaluate a supplied independent image manifest without downloading a dataset.

CSV columns: path,label,split. Paths are relative to the CSV, labels are
normal/pneumonia/tuberculosis, and every evaluated row must say split=test.
No results are described as independent unless provenance is checked by the user.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from _bootstrap import ROOT as ROOT

LABELS = ["normal", "pneumonia", "tuberculosis"]


def read_manifest(path, expected_split):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "label", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest must contain path,label,split")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    seen = set()
    records = []
    for row in rows:
        if row["label"] not in LABELS or row["split"] != expected_split:
            raise ValueError("Unexpected label or split")
        image = (path.parent / row["path"]).resolve()
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError("Duplicate image content within manifest")
        seen.add(digest)
        records.append({"path": image, "label": row["label"], "sha256": digest})
    return records


def evaluate(manifest, model_path, architecture, output, train_manifest=None):
    # Validate inputs before importing expensive model dependencies.
    records = read_manifest(manifest, "test")
    if train_manifest:
        training = read_manifest(train_manifest, "train")
        if {r["sha256"] for r in training} & {r["sha256"] for r in records}:
            raise ValueError("Train/test content overlap detected")
    import medical223_core as core
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
    )

    previous_model = core.MEDICAL223_MODEL_PATH, core.MEDICAL223_MODEL_ARCH
    try:
        core.MEDICAL223_MODEL_PATH = str(Path(model_path).resolve())
        core.MEDICAL223_MODEL_ARCH = architecture
        predictions = []
        for record in records:
            result = core.run_medical223_primary_model(str(record["path"]))
            if not result:
                raise RuntimeError("Inference failed; evaluation stops instead of using heuristics")
            scores = result["scores"]
            predictions.append(
                dict(
                    path=str(record["path"]),
                    sha256=record["sha256"],
                    label=record["label"],
                    prediction=max(scores, key=scores.get),
                    **scores,
                )
            )
    finally:
        core.MEDICAL223_MODEL_PATH, core.MEDICAL223_MODEL_ARCH = previous_model
    truth = [r["label"] for r in predictions]
    guessed = [r["prediction"] for r in predictions]
    summary = {
        "count": len(predictions),
        "architecture": architecture,
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "accuracy": accuracy_score(truth, guessed),
        "classification_report": classification_report(
            truth, guessed, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(truth, guessed, labels=LABELS).tolist(),
        "train_overlap_checked": bool(train_manifest),
        "limitation": "The operator must verify authorization, patient-level split and independence from training. Hash checks do not detect related images from the same patient.",
    }
    if set(truth) == set(LABELS):
        summary["macro_ovr_auc"] = float(
            roc_auc_score(
                truth,
                [[r[label] for label in LABELS] for r in predictions],
                labels=LABELS,
                multi_class="ovr",
                average="macro",
            )
        )
    else:
        summary["macro_ovr_auc"] = None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (output / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-manifest")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--arch", required=True, choices=["tf_efficientnetv2_s.in21k_ft_in1k", "densenet121"]
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args.manifest, args.model, args.arch, args.output, args.train_manifest)
    print(
        "Evaluation completed:",
        result["count"],
        "images. Review metrics.json and provenance before citing.",
    )
