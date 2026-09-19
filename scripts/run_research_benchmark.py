"""Controlled, untuned URL classification benchmark on prepared local exports.

Default settings are experimental configuration, not a tuned recommendation.
Use --purpose smoke for workflow validation; never present that run as final evidence.
"""
import argparse
import copy
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve)
import torch
from torch import nn

from prepare_research_data import FEATURES


INPUTS = FEATURES + [f + "_missing" for f in FEATURES]
MODELS = ["always_malicious", "logistic", "random_forest", "mlp", "dqn", "ddqn"]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class Network(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.layers(x)


def class_weights(y, mode):
    counts = np.bincount(y, minlength=2)
    if (counts == 0).any():
        raise ValueError("Training needs both classes")
    return len(y) / (2.0 * counts) if mode == "balanced" else np.ones(2)


def rewards_for(actions, labels, weights):
    # Weight by true class, never by row position.
    return np.where(actions == labels, 1.0, -1.0) * weights[labels]


@torch.no_grad()
def q_targets(rewards, dones, next_online, next_target, gamma, algorithm):
    if algorithm == "ddqn":
        chosen = next_online.argmax(dim=1, keepdim=True)
        future = next_target.gather(1, chosen).squeeze(1)
    elif algorithm == "dqn":
        future = next_target.max(dim=1).values
    else:
        raise ValueError("Unknown Q-learning algorithm")
    return rewards + gamma * (~dones).float() * future


class Replay:
    def __init__(self, capacity, dim):
        self.capacity = capacity
        self.position = self.size = 0
        self.states = np.empty((capacity, dim), np.float32)
        self.next_states = np.empty((capacity, dim), np.float32)
        self.actions = np.empty(capacity, np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.dones = np.empty(capacity, bool)

    def add(self, states, actions, rewards, next_states, dones):
        for j in range(len(states)):
            i = self.position
            self.states[i], self.actions[i], self.rewards[i] = states[j], actions[j], rewards[j]
            self.next_states[i], self.dones[i] = next_states[j], dones[j]
            self.position = (i + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, size, rng):
        idx = rng.choice(self.size, size=min(size, self.size), replace=False)
        return [torch.from_numpy(a[idx]) for a in
                (self.states, self.actions, self.rewards, self.next_states, self.dones)]


def train_neural(x, y, algorithm, seed, updates, batch_size=64, gamma=.99,
                 learning_rate=.001, weighting="none", capacity=10000, target_interval=100):
    if algorithm not in {"mlp", "dqn", "ddqn"} or updates < 1 or batch_size < 1:
        raise ValueError("Invalid neural training configuration")
    if capacity < batch_size or target_interval < 1 or not 0 <= gamma <= 1:
        raise ValueError("Invalid replay, target update, or discount setting")
    if not np.isfinite(x).all() or len(x) != len(y):
        raise ValueError("Invalid training arrays")
    seed_all(seed)
    # Separate streams prevent replay sampling from changing data order/exploration.
    order_rng = np.random.RandomState(seed)
    action_rng = np.random.RandomState(seed + 1)
    replay_rng = np.random.RandomState(seed + 2)
    net = Network(x.shape[1])
    target = copy.deepcopy(net).eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    weights = class_weights(y, weighting)
    loss_weights = torch.tensor(weights, dtype=torch.float32)
    replay = Replay(capacity, x.shape[1])
    order = order_rng.permutation(len(x))
    cursor = exposures = completed_epochs = 0
    history = []
    for update in range(updates):
        if cursor == len(x):
            order = order_rng.permutation(len(x))
            cursor = 0
        positions = np.arange(cursor, min(cursor + batch_size, len(x)))
        ids = order[positions]
        states = x[ids]
        labels = y[ids]
        done = positions == len(x) - 1
        next_states = x[order[(positions + 1) % len(x)]].copy()
        next_states[done] = 0
        cursor += len(ids)
        exposures += len(ids)
        completed_epochs += int(done.sum())
        epsilon = max(.05, .995 ** update)
        if algorithm == "mlp":
            logits = net(torch.from_numpy(states))
            loss = nn.functional.cross_entropy(logits, torch.from_numpy(labels), weight=loss_weights)
            mean_reward = None
        else:
            with torch.no_grad():
                actions = net(torch.from_numpy(states)).argmax(1).numpy()
            explore = action_rng.random_sample(len(ids)) < epsilon
            random_actions = action_rng.randint(0, 2, size=len(ids))
            actions = np.where(explore, random_actions, actions)
            rewards = rewards_for(actions, labels, weights).astype(np.float32)
            replay.add(states, actions, rewards, next_states, done)
            s, a, r, ns, terminal = replay.sample(batch_size, replay_rng)
            expected = q_targets(r, terminal, net(ns), target(ns), gamma, algorithm)
            current = net(s).gather(1, a[:, None]).squeeze(1)
            loss = nn.functional.mse_loss(current, expected)
            mean_reward = float(rewards.mean())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        if (update + 1) % target_interval == 0:
            target.load_state_dict(net.state_dict())
        history.append({"update": update + 1, "loss": float(loss.detach()),
                        "epsilon": epsilon if algorithm != "mlp" else None,
                        "collection_mean_reward": mean_reward, "training_sample_exposures": exposures})
    return net.eval(), history, {"class_weights": weights.tolist(), "optimizer_updates": updates,
                                "sample_exposures": exposures, "completed_training_traversals": completed_epochs}


def score_model(model, name, x):
    if name == "always_malicious":
        return np.ones(len(x))
    if name == "logistic":
        return model.decision_function(x)
    if name == "random_forest":
        return model.predict_proba(x)[:, list(model.classes_).index(1)]
    with torch.no_grad():
        parts = []
        for start in range(0, len(x), 4096):
            q = model(torch.from_numpy(x[start:start + 4096]))
            parts.append((q[:, 1] - q[:, 0]).numpy())
        return np.concatenate(parts)


def select_thresholds(y, score, default, fpr_limit=.01):
    if not np.isfinite(score).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Threshold selection requires finite scores and both validation classes")
    p, r, ts = precision_recall_curve(y, score)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=p + r > 0)
    above_max = float(np.nextafter(float(np.max(score)), np.inf))
    ts = np.append(ts, above_max)
    best = np.flatnonzero(f1 == f1.max())[-1]
    fp, tp, rt = roc_curve(y, score, drop_intermediate=False)
    rt = np.where(np.isfinite(rt), rt, above_max)
    valid = np.flatnonzero(fp <= fpr_limit)
    chosen = max(valid, key=lambda i: (tp[i], -fp[i], rt[i]))
    return {"default": float(default), "validation_f1": float(ts[best]),
            "validation_fpr_limit": float(rt[chosen])}


def evaluate(y, score, threshold):
    # Strict > makes a zero Q/logit tie choose class 0, matching argmax.
    # Threshold-selected operating points use >=, as PR/ROC definitions require.
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "fpr": fp / (fp + tn), "specificity": tn / (tn + fp),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "average_precision": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)),
            "predicted_malicious_fraction": float(pred.mean()), "n": len(y)}


def load_export(folder, manifest, source, target, split):
    name = source + "_to_" + target + "_" + split + ".csv"
    path = Path(folder) / name
    info = manifest["exports"][name]
    if file_hash(path) != info["sha256"] or info["transformer_source"] != source:
        raise ValueError("Export checksum/transformer mismatch: " + name)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["sample_id"] + INPUTS + ["label"]:
        raise ValueError("Unexpected export schema")
    x = frame[INPUTS].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int64)
    if not frame["sample_id"].is_unique or len(frame) != info["rows"]:
        raise ValueError("Invalid sample IDs or row count")
    if not np.isfinite(x).all() or set(np.unique(frame["label"])) != {0, 1}:
        raise ValueError("Invalid features or labels")
    return frame["sample_id"].to_numpy(), x, y


def run(args):
    folder, output = Path(args.prepared), Path(args.output)
    if output.exists():
        raise ValueError("Output exists; use a new directory")
    if args.updates < 1 or args.trees < 1 or not 0 < args.fpr_limit < 1:
        raise ValueError("Invalid budget or FPR constraint")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    torch.set_num_threads(args.threads)
    manifest = json.loads((folder / "manifest.json").read_text())
    output.mkdir(parents=True)
    configuration = dict(vars(args))
    configuration.update({"input_order": INPUTS, "hidden_layers": [128, 128], "device": "cpu",
                          "replay_capacity": 10000, "target_interval_updates": 100,
                          "epsilon_initial": 1.0, "epsilon_decay_per_update": .995, "epsilon_min": .05,
                          "gradient_clip_norm": 1.0, "tuning": "none", "early_stopping": "none",
                          "preparation_manifest_sha256": file_hash(folder / "manifest.json"),
                          "script_sha256": file_hash(__file__),
                          "feature_status": manifest["policy"]["status"],
                          "software": {"python": platform.python_version(), "torch": torch.__version__,
                                       "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
                          "hardware": {"platform": platform.platform(), "processor": platform.processor()}})
    configuration.pop("prepared")
    configuration.pop("output")
    save_json(output / "run.json", configuration)
    metrics = []
    for source in ["iscx", "mendeley"]:
        train_ids, x, y = load_export(folder, manifest, source, source, "train")
        val_ids, vx, vy = load_export(folder, manifest, source, source, "validation")
        if set(train_ids) & set(val_ids):
            raise ValueError("Training/validation sample overlap")
        for seed in args.seeds:
            for name in MODELS:
                directory = output / (source + "_" + name + "_" + str(seed))
                directory.mkdir()
                start = time.perf_counter()
                training = {}
                if name == "always_malicious":
                    model = None
                elif name == "logistic":
                    model = LogisticRegression(max_iter=2000, random_state=seed,
                                               class_weight="balanced" if args.weighting == "balanced" else None)
                    model.fit(x, y)
                    training["iterations"] = model.n_iter_.tolist()
                elif name == "random_forest":
                    model = RandomForestClassifier(n_estimators=args.trees, random_state=seed, n_jobs=args.threads,
                                                   class_weight="balanced" if args.weighting == "balanced" else None)
                    model.fit(x, y)
                else:
                    model, history, training = train_neural(x, y, name, seed, args.updates,
                        batch_size=args.batch_size, gamma=args.gamma, learning_rate=args.learning_rate, weighting=args.weighting)
                    pd.DataFrame(history).to_csv(directory / "training.csv", index=False)
                fit_seconds = time.perf_counter() - start
                if name in {"mlp", "dqn", "ddqn"}:
                    torch.save(model.state_dict(), directory / "weights.pt")
                elif model is not None:
                    joblib.dump(model, directory / "model.joblib")
                val_score = score_model(model, name, vx).astype(float)
                # sklearn/argmax choose class 0 on exact default ties.
                default = .5 if name in {"always_malicious", "random_forest"} else 0.0
                default = float(np.nextafter(default, np.inf))
                thresholds = select_thresholds(vy, val_score, default, args.fpr_limit)
                validation = {op: evaluate(vy, val_score, threshold) for op, threshold in thresholds.items()}
                save_json(directory / "selection.json", {"threshold_source": "source_validation_only",
                    "thresholds": thresholds, "validation_metrics": validation,
                    "validation_benign_count": int((vy == 0).sum()), "fpr_limit": args.fpr_limit,
                    "fit_seconds": fit_seconds, "training": training})
                pd.DataFrame({"sample_id": val_ids, "label": vy, "score": val_score}).to_csv(directory / "validation_predictions.csv", index=False)
                for target in ["iscx", "mendeley"]:
                    ids, tx, ty = load_export(folder, manifest, source, target, "test")
                    if set(ids) & (set(train_ids) | set(val_ids)):
                        raise ValueError("Test sample overlap")
                    start = time.perf_counter()
                    score = score_model(model, name, tx).astype(float)
                    seconds = time.perf_counter() - start
                    if not np.isfinite(score).all():
                        raise ValueError("Nonfinite prediction score")
                    predictions = pd.DataFrame({"sample_id": ids, "label": ty, "score": score})
                    for op, threshold in thresholds.items():
                        predictions["prediction_" + op] = (score >= threshold).astype(int)
                        result = evaluate(ty, score, threshold)
                        result.update(source=source, target=target, model=name, seed=seed, operating_point=op,
                                      threshold=threshold, fit_seconds=fit_seconds, inference_seconds=seconds)
                        metrics.append(result)
                    predictions.to_csv(directory / (target + "_test_predictions.csv"), index=False)
                    p, r, t = precision_recall_curve(ty, score)
                    pd.DataFrame({"precision": p[:-1], "recall": r[:-1], "threshold": t}).to_csv(directory / (target + "_pr_curve.csv"), index=False)
                    fp, tp, rt = roc_curve(ty, score)
                    pd.DataFrame({"fpr": fp, "tpr": tp, "threshold": rt}).to_csv(directory / (target + "_roc_curve.csv"), index=False)
                pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
                print(source, name, "seed", seed, "completed", round(fit_seconds, 2), "seconds", flush=True)
    table = pd.DataFrame(metrics)
    summary = table.groupby(["source", "target", "model", "operating_point"])[["average_precision", "roc_auc", "f1", "fpr", "recall"]].agg(["mean", "std", "count"])
    summary.to_csv(output / "summary.csv")
    save_json(output / "completion.json", {"complete": True, "metric_rows": len(table), "purpose": args.purpose,
                                          "note": "One-seed sample standard deviations are undefined; no significance claims."})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--purpose", choices=["smoke", "exploratory"], required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[11])
    p.add_argument("--updates", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=.99)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--weighting", choices=["none", "balanced"], default="none")
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--fpr-limit", type=float, default=.01)
    run(p.parse_args())


if __name__ == "__main__":
    main()
