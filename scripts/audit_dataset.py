"""Read-only CSV audit. Python standard library only; no model deserialization.

Duplicate counts describe recorded values, not necessarily duplicate URLs.
No label semantics are inferred. Outputs contain aggregate statistics only.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path


MISSING = frozenset(("", "na", "n/a", "nan", "null", "none", "?"))


def audit_dataset(path, label_column, identifier_columns=()):
    path = Path(path)
    raw = path.read_bytes()
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    try:
        headers = next(reader)
    except StopIteration:
        raise ValueError("CSV is empty") from None
    if not headers or any(not h.strip() for h in headers):
        raise ValueError("Every CSV column needs a non-empty name")
    if len(set(headers)) != len(headers):
        raise ValueError("CSV contains duplicate column names")
    if label_column not in headers:
        raise ValueError("Label column is absent: " + label_column)
    if len(set(identifier_columns)) != len(identifier_columns):
        raise ValueError("Identifier column names must be unique")
    if label_column in identifier_columns:
        raise ValueError("Label column cannot also be an identifier")
    for name in identifier_columns:
        if name not in headers:
            raise ValueError("Identifier column is absent: " + name)
    rows = []
    blank_records = 0
    for row in reader:
        if not row:
            blank_records += 1
            continue
        if len(row) != len(headers):
            raise ValueError("Wrong field count at CSV line %d" % reader.line_num)
        rows.append(tuple(row))
    if not rows:
        raise ValueError("CSV contains no data rows")

    label_index = headers.index(label_column)
    feature_indices = [i for i, h in enumerate(headers)
                       if h != label_column and h not in identifier_columns]
    if not feature_indices:
        raise ValueError("No feature columns remain")
    columns = {}
    for i, name in enumerate(headers):
        values = [row[i] for row in rows]
        present = [v for v in values if v.strip().lower() not in MISSING]
        numeric = []
        nonnumeric = nonfinite = 0
        for value in present:
            try:
                number = float(value)
            except ValueError:
                nonnumeric += 1
                continue
            if math.isfinite(number):
                numeric.append(number)
            else:
                nonfinite += 1
        columns[name] = {
            "missing_count": len(values) - len(present),
            "distinct_nonmissing_values": len(set(present)),
            "finite_numeric_count": len(numeric),
            "nonfinite_numeric_count": nonfinite,
            "nonnumeric_count": nonnumeric,
            "numeric_min": min(numeric) if numeric else None,
            "numeric_max": max(numeric) if numeric else None,
        }

    full_counts = Counter(rows)
    feature_labels = defaultdict(Counter)
    for row in rows:
        feature_labels[tuple(row[i] for i in feature_indices)][row[label_index]] += 1
    conflicts = [counts for counts in feature_labels.values() if len(counts) > 1]
    labels = Counter(row[label_index] for row in rows)
    missing_label_count = columns[label_column]["missing_count"]
    warnings = [
        "Label meanings and dataset provenance are unverified.",
        "Feature-vector repetition does not prove that underlying URLs are duplicates.",
        "Duplicate comparisons use exact CSV strings; numeric equivalences are not normalized.",
    ]
    if not identifier_columns:
        warnings.append("No identifier columns supplied; URL/domain overlap cannot be verified.")
    if missing_label_count:
        warnings.append("Missing labels must be resolved before supervised experiments.")
    if conflicts:
        warnings.append("Identical recorded feature vectors occur with different labels.")
    nonmissing_labels = [v for v in labels if v.strip().lower() not in MISSING]
    if len(nonmissing_labels) != 2:
        warnings.append("Observed nonmissing label count is not two.")
    return {
        "schema_version": 1,
        "input_file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob_sha1": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
        "byte_count": len(raw),
        "row_count": len(rows),
        "column_count": len(headers),
        "feature_count": len(feature_indices),
        "label_column": label_column,
        "identifier_columns": list(identifier_columns),
        "blank_records_skipped": blank_records,
        "missing_tokens_case_insensitive": sorted(MISSING),
        "label_counts_raw": dict(sorted(labels.items())),
        "label_fractions_raw": {k: v / len(rows) for k, v in sorted(labels.items())},
        "label_semantics": "unverified",
        "missing_label_count": missing_label_count,
        "exact_duplicate_rows_beyond_first": sum(n - 1 for n in full_counts.values()),
        "distinct_feature_vectors": len(feature_labels),
        "repeated_feature_rows_beyond_first": len(rows) - len(feature_labels),
        "conflicting_feature_groups": len(conflicts),
        "rows_in_conflicting_feature_groups": sum(sum(c.values()) for c in conflicts),
        "constant_nonmissing_features": [headers[i] for i in feature_indices
                                         if columns[headers[i]]["distinct_nonmissing_values"] == 1],
        "all_missing_features": [headers[i] for i in feature_indices
                                 if columns[headers[i]]["missing_count"] == len(rows)],
        "columns": columns,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--identifier-column", action="append", default=[])
    parser.add_argument("--output", type=Path, help="Optional JSON report; otherwise print to stdout")
    args = parser.parse_args()
    if args.output and args.output.resolve() == args.csv_path.resolve():
        parser.error("Output must not overwrite the input dataset")
    try:
        report = audit_dataset(args.csv_path, args.label_column, args.identifier_column)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        parser.error(str(exc))
    result = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
