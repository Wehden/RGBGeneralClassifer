#!/usr/bin/env python3
"""Replay exported material students on held-out pixels using the NUC runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgb.materials import MaterialClassifier


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def probabilities(logits, temperature):
    values = logits / temperature
    values = np.exp(values - values.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def summarize(records, source_kind):
    selected = [row for row in records if row["source_kind"] == source_kind]
    material = [row for row in selected if row["reference_kind"] == "material"]
    rejection = [row for row in selected if row["reference_kind"] == "rejection"]
    unresolved = [row for row in selected if row["reference_kind"] == "unresolved"]
    accepted = [row for row in material if row["label"] is not None]
    by_class = {}
    for label in sorted({row["reference"] for row in material}):
        cohort = [row for row in material if row["reference"] == label]
        by_class[label] = {
            "references": len(cohort),
            "top1_correct": sum(row["best_candidate"] == label for row in cohort),
            "accepted": sum(row["label"] is not None for row in cohort),
            "accepted_correct": sum(row["label"] == label for row in cohort),
        }
    return {
        "examples": len(selected),
        "material_references": len(material),
        "top1_correct": sum(row["best_candidate"] == row["reference"] for row in material),
        "accepted": sum(row["label"] is not None for row in selected),
        "accepted_material_references": len(accepted),
        "accepted_material_correct": sum(row["label"] == row["reference"] for row in accepted),
        "rejection_references": len(rejection),
        "rejection_references_accepted_as_material": sum(row["label"] is not None for row in rejection),
        "unresolved_references": len(unresolved),
        "unresolved_references_accepted": sum(row["label"] is not None for row in unresolved),
        "by_class": by_class,
    }


def verify(data, output, variants, splits, threads, tolerance):
    rows = json.loads((output / "evaluation-rows.json").read_text())
    with np.load(output / "evaluation-logits.npz", allow_pickle=False) as cached:
        if cached["ids"].tolist() != [row["id"] for row in rows]:
            raise ValueError("Evaluation sample alignment mismatch")
        logits = {name: cached[key].copy() for name, key in
                  [("frozen-head", "frozen"), ("fine-tuned", "tuned")]}
    selected = [i for i, row in enumerate(rows) if row["split"] in splits]
    if not selected:
        raise ValueError("No held-out samples selected")
    report = {
        "runtime": "rgb.materials.MaterialClassifier",
        "device": "CPU",
        "threads": threads,
        "splits": splits,
        "score_absolute_error_gate": tolerance,
        "variants": {},
    }
    for variant in variants:
        classifier = MaterialClassifier(output / "artifacts" / variant, threads=threads)
        manifest = classifier.manifest
        for name, filename in [("dataset_sha256", "dataset.json"),
                               ("controls_sha256", "controls.json"),
                               ("labels_sha256", "teacher-labels.json")]:
            if manifest["training"][name] != digest(data / filename):
                raise ValueError("Artifact training provenance mismatch")
        policy = manifest["classifier"]
        expected = probabilities(logits[variant], policy["temperature"])
        supported = manifest["training"]["train_material_counts"]
        rejection_labels = manifest["rejection_labels"]
        records = []
        mismatches = []
        max_error = 0.0
        for sequence, i in enumerate(selected):
            row = rows[i]
            path = (data / row["file"]).resolve()
            if not path.is_relative_to(data.resolve()) or digest(path) != row["sha256"]:
                raise ValueError("Held-out pixel identity mismatch")
            with Image.open(path) as image:
                result = classifier.classify(np.array(image.convert("RGB")))
            ordered = np.argsort(-expected[i], kind="stable")
            best, second = map(int, ordered[:2])
            best_label = manifest["labels"][best]
            margin = float(expected[i, best] - expected[i, second])
            accepted = (policy["qualified_for_acceptance"]
                        and expected[i, best] >= policy["minimum_probability"]
                        and margin >= policy["minimum_margin"]
                        and best_label not in rejection_labels
                        and supported.get(best_label, 0) >= 3)
            expected_label = best_label if accepted else None
            score_error = abs(result["score"] - float(expected[i, best]))
            margin_error = abs(result["margin"] - margin)
            candidate_error = max(abs(candidate["score"] - float(
                expected[i, manifest["labels"].index(candidate["label"])]))
                for candidate in result["candidates"])
            error = max(score_error, margin_error, candidate_error)
            max_error = max(max_error, error)
            if (result["best_candidate"] != best_label or result["label"] != expected_label
                    or error > tolerance):
                mismatches.append({"id": row["id"], "score_error": error,
                                   "expected_candidate": best_label, "expected_label": expected_label,
                                   "actual_candidate": result["best_candidate"], "actual_label": result["label"]})
            kind = ("unresolved" if row["target"] < 0 else
                    "rejection" if row["material"] in rejection_labels else "material")
            records.append({"id": row["id"], "split": row["split"],
                            "source_kind": row["source_kind"], "reference": row["material"],
                            "reference_kind": kind,
                            **{key: result[key] for key in
                               ["best_candidate", "label", "score", "margin", "inference_ms"]}})
            if (sequence + 1) % 25 == 0:
                print(json.dumps({"variant": variant, "checked": sequence + 1,
                                  "total": len(selected), "mismatches": len(mismatches)}), flush=True)
        elapsed = [row["inference_ms"] for row in records]
        report["variants"][variant] = {
            "manifest_sha256": classifier.config["manifest_sha256"],
            "runtime_version": classifier.config["runtime_version"],
            "checked": len(records),
            "maximum_score_or_margin_absolute_error": max_error,
            "decision_mismatches": mismatches,
            "inference_ms": {"median": float(np.median(elapsed)), "p95": float(np.percentile(elapsed, 95))},
            "cohorts": {split: {kind: summarize([row for row in records if row["split"] == split], kind)
                               for kind in ["MINC", "camera"]} for split in splits},
            "records": records,
        }
        del classifier
    report["passed"] = all(not row["decision_mismatches"] for row in report["variants"].values())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=["frozen-head", "fine-tuned"],
                        default=["frozen-head", "fine-tuned"])
    parser.add_argument("--splits", nargs="+", choices=["validation", "test"], default=["test"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    result = verify(args.data, args.output, args.variants, args.splits, args.threads, args.tolerance)
    result["elapsed_s"] = time.monotonic() - started
    report_path = args.report or args.output / "nuc-verification.json"
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "report": str(report_path),
                      "elapsed_s": result["elapsed_s"]}), flush=True)
    if not result["passed"]:
        raise SystemExit("NUC replay failed; inspect the verification receipt")
