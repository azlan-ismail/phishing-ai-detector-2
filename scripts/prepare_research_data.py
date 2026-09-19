"""Prepare exploratory URL-feature experiments from local source tables.

No raw URLs, model fitting, resampling, or feature-importance selection.
Candidate cross-source semantics and Mendeley label provenance remain pending.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["url_length", "domain_length", "dot_count_url", "domain_url_ratio"]
SPLITS = ("train", "validation", "test")
POLICY = {
    "status": "exploratory_pending_source_definition_verification",
    "feature_order": FEATURES,
    "mapping": {
        "iscx": {"url_length": "urlLen", "domain_length": "domainlength", "dot_count_url": "NumberofDotsinURL"},
        "mendeley": {"url_length": "length_url", "domain_length": "domain_length", "dot_count_url": "qty_dot_url"},
    },
    "ratio": "domain_length / url_length, recomputed identically in both sources",
    "invalid_values": "Nonfinite values, nonpositive lengths, and negative dot counts become missing; never perform arithmetic on them.",
    "missing_indicators": "One fixed indicator per candidate feature, even if source training has no missing values.",
    "labels": {"iscx": {"benign": 0, "phishing": 1}, "mendeley": {"0": 0, "1": 1}},
    "label_provenance": "ISCX labels are explicit strings. Mendeley 0=benign,1=phishing follows the supplied study convention and still requires source documentation.",
    "grouping": "SHA256 of all numeric original predictors, excluding label; identical feature records stay together. This is not URL/domain grouping.",
}


def read_source(path, dataset):
    raw = Path(path).read_bytes()
    frame = pd.read_csv(path, low_memory=False)
    label_col = "URL_Type_obf_Type" if dataset == "iscx" else "phishing"
    if label_col not in frame or frame.empty:
        raise ValueError("Missing labels or empty source: " + dataset)
    values = frame[label_col]
    if dataset == "iscx":
        labels = values.map({"benign": 0, "phishing": 1})
    else:
        labels = pd.to_numeric(values, errors="raise")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}) or labels.nunique() != 2:
        raise ValueError("Expected both binary classes without missing labels: " + dataset)
    predictors = frame.drop(columns=[label_col]).apply(pd.to_numeric, errors="raise")
    data = predictors.to_numpy(dtype="<f8", copy=True)
    data[data == 0] = 0  # Normalize signed zero for identity hashing.
    data[np.isnan(data)] = np.nan  # Normalize NaN payloads, retaining infinity signs.
    groups = np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in data])
    digest = hashlib.sha256(raw).hexdigest()
    ids = np.array([dataset + ":" + digest + ":" + str(i) for i in range(len(frame))])
    return frame, labels.to_numpy(dtype=int), groups, ids, digest


def candidates(frame, dataset):
    mapping = POLICY["mapping"][dataset]
    result = pd.DataFrame(index=frame.index)
    for target, source in mapping.items():
        values = pd.to_numeric(frame[source], errors="raise").astype(float)
        invalid = ~np.isfinite(values) | (values < 0)
        if target != "dot_count_url":
            invalid |= values == 0
        result[target] = values.mask(invalid)
    result["domain_url_ratio"] = result["domain_length"] / result["url_length"]
    return result[FEATURES].replace([np.inf, -np.inf], np.nan)


def grouped_split(labels, groups, seed):
    """Greedy class-balanced 60/20/20 allocation of whole feature groups.

    Large groups first, with seeded random tie ordering. Group indivisibility
    means proportions are approximate. No model scores enter this allocation.
    """
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.zeros((len(unique), 2), dtype=int)
    np.add.at(counts, (inverse, labels), 1)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(unique))
    order = order[np.argsort(-counts[order].sum(axis=1), kind="stable")]
    targets = np.array([.6, .2, .2])[:, None] * np.bincount(labels, minlength=2)
    allocated = np.zeros((3, 2), dtype=int)
    assignments = np.empty(len(unique), dtype=int)
    for group in order:
        before = ((allocated - targets) ** 2 / np.maximum(targets, 1)).sum(axis=1)
        after = ((allocated + counts[group] - targets) ** 2 / np.maximum(targets, 1)).sum(axis=1)
        destination = int(np.argmin(after - before))
        assignments[group] = destination
        allocated[destination] += counts[group]
    row_split = assignments[inverse]
    if (allocated == 0).any():
        raise ValueError("Cannot obtain all three partitions with both classes; inspect group sizes")
    return row_split


def fit_transformer(train):
    if list(train.columns) != FEATURES:
        raise ValueError("Unexpected candidate feature order")
    medians = train.median()
    if medians.isna().any():
        raise ValueError("Source training has an entirely missing candidate feature")
    filled = train.fillna(medians)
    return {"feature_order": FEATURES, "median": medians.to_dict(),
            "minimum": filled.min().to_dict(), "maximum": filled.max().to_dict(),
            "fit_rows": len(train), "fit_partition": "source_train_only",
            "clip": False, "missing_indicator_order": [c + "_missing" for c in FEATURES]}


def transform(frame, state):
    if list(frame.columns) != state["feature_order"]:
        raise ValueError("Unexpected candidate feature order")
    medians = pd.Series(state["median"])
    minimum, maximum = pd.Series(state["minimum"]), pd.Series(state["maximum"])
    scale = (maximum - minimum).replace(0, 1)
    result = (frame.fillna(medians) - minimum) / scale
    for col in FEATURES:
        result[col + "_missing"] = frame[col].isna().astype(int)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Nonfinite output after source-fitted transformation")
    return result


def prepare(paths, output, seed=42, group_by="full"):
    if group_by not in {"full", "candidate"}:
        raise ValueError("group_by must be full or candidate")
    output = Path(output)
    if output.exists():
        raise ValueError("Output directory already exists; use a new run directory")
    prepared = {}
    for name, path in paths.items():
        frame, labels, groups, ids, digest = read_source(path, name)
        x = candidates(frame, name)
        full_groups = groups.copy()
        if group_by == "candidate":
            values = x.to_numpy(dtype="<f8", copy=True)
            values[values == 0] = 0
            values[np.isnan(values)] = np.nan
            groups = np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in values])
        split = grouped_split(labels, groups, seed)
        state = fit_transformer(x.iloc[np.flatnonzero(split == 0)])
        prepared[name] = dict(x=x, labels=labels, groups=groups, full_groups=full_groups, ids=ids, digest=digest, split=split, state=state)
    # Validate both sources before creating outputs. All output names are fixed.
    output.mkdir(parents=True)
    manifest = {"schema_version": 1, "seed": seed, "policy": POLICY, "group_by": group_by,
                "software": {"numpy": np.__version__, "pandas": pd.__version__},
                "datasets": {}, "exports": {}, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name, item in prepared.items():
        partition = pd.DataFrame({"sample_id": item["ids"], "feature_group": item["groups"],
                                  "label": item["labels"], "split": [SPLITS[i] for i in item["split"]]})
        partition.to_csv(output / (name + "_partitions.csv"), index=False)
        (output / (name + "_transformer.json")).write_text(json.dumps(item["state"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Group disjointness is asserted, while lower-dimensional collisions are reported.
        sets = [set(item["groups"][item["split"] == n]) for n in range(3)]
        assert all(not sets[a].intersection(sets[b]) for a, b in [(0, 1), (0, 2), (1, 2)])
        full_sets = [set(item["full_groups"][item["split"] == n]) for n in range(3)]
        assert all(not full_sets[a].intersection(full_sets[b]) for a, b in [(0, 1), (0, 2), (1, 2)])
        train_hashes = set(pd.util.hash_pandas_object(item["x"][item["split"] == 0], index=False))
        manifest["datasets"][name] = {"raw_sha256": item["digest"], "rows": len(item["labels"]),
            "missing_candidates": item["x"].isna().sum().to_dict(), "full_feature_groups_disjoint": True,
            "partitions": {part: {"rows": int((item["split"] == n).sum()),
                                   "class_counts": np.bincount(item["labels"][item["split"] == n], minlength=2).tolist()}
                           for n, part in enumerate(SPLITS)},
            "candidate_test_rows_matching_training": int(pd.util.hash_pandas_object(item["x"][item["split"] == 2], index=False).isin(train_hashes).sum()),
            "partition_sha256": hashlib.sha256((output / (name + "_partitions.csv")).read_bytes()).hexdigest()}
    for source, fitted in prepared.items():
        for target, item in prepared.items():
            partitions = range(3) if source == target else [2]
            for n in partitions:
                indices = np.flatnonzero(item["split"] == n)
                result = transform(item["x"].iloc[indices], fitted["state"]).reset_index(drop=True)
                result.insert(0, "sample_id", item["ids"][indices])
                result["label"] = item["labels"][indices]
                filename = source + "_to_" + target + "_" + SPLITS[n] + ".csv"
                result.to_csv(output / filename, index=False)
                manifest["exports"][filename] = {"rows": len(result), "sha256": hashlib.sha256((output / filename).read_bytes()).hexdigest(), "transformer_source": source}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iscx", type=Path, required=True)
    parser.add_argument("--mendeley", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-by", choices=["full", "candidate"], default="full",
                        help="full: original feature records; candidate: stricter reduced-feature sensitivity split")
    args = parser.parse_args()
    try:
        result = prepare({"iscx": args.iscx, "mendeley": args.mendeley}, args.output, args.seed, args.group_by)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result["datasets"], indent=2))


if __name__ == "__main__":
    main()
