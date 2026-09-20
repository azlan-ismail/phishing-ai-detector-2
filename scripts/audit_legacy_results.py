"""Reconstruct the tutorial split and check saved metrics; never load models.

Requires NumPy (already listed in requirements.txt). Split reconstruction assumes
the current CSV row order and the split settings in train_and_evaluate_all_models.py.
It does not verify which data were used to fit the saved model objects.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def check_metrics(metrics):
    matrix = np.asarray(metrics["confusion_matrix"])
    if matrix.shape != (2, 2) or not np.issubdtype(matrix.dtype, np.integer) or (matrix < 0).any():
        raise ValueError("Expected a nonnegative integer 2x2 confusion matrix")
    tn, fp, fn, tp = [int(x) for x in matrix.ravel()]
    total = tn + fp + fn + tp
    if total == 0:
        raise ValueError("Confusion matrix is empty")
    values = {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1-score": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "support": tp + fn,
    }
    report = metrics["classification_report"]
    checks = {key: math.isclose(value, report["1"][key], rel_tol=1e-10, abs_tol=1e-12)
              for key, value in values.items()}
    checks["accuracy"] = math.isclose((tn + tp) / total, report["accuracy"], rel_tol=1e-10)
    return {"class_1_recomputed": values, "checks": checks,
            "all_checked_values_match": all(checks.values()),
            "accuracy": (tn + tp) / total, "matrix_total": total}


def audit_legacy(root):
    root = Path(root)
    raw = (root / "data/phishing.csv").read_bytes()
    with (root / "data/phishing.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = [tuple(row) for row in reader if row]
    if not rows or header[-1] != "Result" or any(len(r) != len(header) for r in rows):
        raise ValueError("Expected the tutorial CSV schema with Result as last column")
    order = np.random.RandomState(42).permutation(len(rows))
    ntest = math.ceil(0.3 * len(rows))
    train = [rows[i] for i in order[ntest:]]
    test = [rows[i] for i in order[:ntest]]
    train_rows = set(train)
    train_features = {r[:-1] for r in train}
    return {
        "method": "Reconstruct tutorial split with NumPy RandomState(42); first ceil(0.3*n) indices are test. No model loading or training.",
        "limitations": "Assumes current row order and recorded split settings. Repeated coarse features do not establish duplicate URLs. Class semantics remain unverified.",
        "numpy_version": np.__version__,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "train_rows": len(train), "test_rows": len(test),
        "raw_test_labels": {label: sum(r[-1] == label for r in test) for label in sorted({r[-1] for r in rows})},
        "test_rows_matching_training_row_including_label": sum(r in train_rows for r in test),
        "test_rows_matching_training_features": sum(r[:-1] in train_features for r in test),
        "metrics": {name: check_metrics(json.loads((root / ("metrics/metrics_" + name + ".json")).read_text()))
                    for name in ("lr", "rf", "xgb")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(audit_legacy(args.repo_root), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
