"""Select a common neural update budget using source validation AP only.

No source-test or target-test export is opened by this script.
Pilot seed is separate from the five reporting seeds.
"""
import argparse
import json
from pathlib import Path
import time

import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from run_research_benchmark import file_hash, load_export, save_json, score_model, train_neural


def choose_budget(records, tolerance):
    if tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")
    table = pd.DataFrame(records)
    expected = {(s, m) for s in ("iscx", "mendeley") for m in ("mlp", "dqn", "ddqn")}
    for _, group in table.groupby("updates"):
        if len(group) != len(expected) or set(zip(group.source, group.model)) != expected:
            raise ValueError("Each candidate requires every source/model pair exactly once")
    means = table.groupby("updates").validation_ap.mean().sort_index()
    best = float(means.max())
    selected = int(means[means >= best - tolerance].index.min())
    boundary_best = bool(means.idxmax() == means.index.max())
    return {"selected_updates": selected, "mean_validation_ap": {str(k): float(v) for k, v in means.items()},
            "best_mean_validation_ap": best, "best_at_largest_candidate": boundary_best,
            "convergence_established": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--candidates", nargs="+", type=int, default=[1000, 2000, 4000])
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--tolerance", type=float, default=.005)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if not args.candidates or min(args.candidates) < 1 or len(set(args.candidates)) != len(args.candidates):
        raise ValueError("Candidate budgets must be distinct positive integers")
    if args.tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")
    folder, output = Path(args.prepared), Path(args.output)
    if output.exists():
        raise ValueError("Output exists; use a new run directory")
    output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    manifest = json.loads((folder / "manifest.json").read_text())
    policy = {"rule": "Smallest common budget within tolerance of best mean source-validation AP across 2 sources x 3 neural models; each pair equally weighted.",
              "candidate_updates": sorted(args.candidates), "pilot_seed": args.seed, "ap_tolerance": args.tolerance,
              "preparation_manifest_sha256": file_hash(folder / "manifest.json"),
              "selector_script_sha256": file_hash(__file__),
              "trainer_script_sha256": file_hash(Path(__file__).with_name("run_research_benchmark.py")),
              "gamma": .99, "batch_size": 64, "learning_rate": .001, "weighting": "none",
              "threads": args.threads, "test_data_used_for_selection": False}
    save_json(output / "policy.json", policy)
    records = []
    for source in ["iscx", "mendeley"]:
        _, x, y = load_export(folder, manifest, source, source, "train")
        _, vx, vy = load_export(folder, manifest, source, source, "validation")
        for model_name in ["mlp", "dqn", "ddqn"]:
            for updates in sorted(args.candidates):
                start = time.perf_counter()
                model, history, training = train_neural(x, y, model_name, args.seed, updates)
                score = score_model(model, model_name, vx)
                row = {"source": source, "model": model_name, "updates": updates,
                       "validation_ap": float(average_precision_score(vy, score)),
                       "seconds": time.perf_counter() - start, **training}
                records.append(row)
                pd.DataFrame(history).to_csv(output / f"{source}_{model_name}_{updates}_history.csv", index=False)
                pd.DataFrame(records).to_csv(output / "pilot.csv", index=False)
                print(source, model_name, updates, "validation AP", round(row["validation_ap"], 5), flush=True)
    result = {**policy, **choose_budget(records, args.tolerance), "completed_fits": len(records)}
    save_json(output / "selection.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
