# Controlled training and evaluation

**Provenance update (19 September 2026):** Mendeley's published labels are now confirmed as 0 = legitimate and 1 = phishing. The ISCX file is treated as the IEEE file renamed without content changes, per the user's explicit assumption; this has not been independently byte-verified. Source papers support the four intended feature meanings, but exact extraction equivalence remains unverified. The [feature-compatibility audit](feature-compatibility.md) documents strong class-conditional distribution differences. This update supersedes earlier statements below that Mendeley's published label meaning is unresolved; historical results and manifests are unchanged.


Status: shared trainer and six-model benchmark implemented. The initial real-data run is a single-seed, low-budget smoke experiment. It validates the workflow, not the manuscript's final scientific claims.

The executed smoke run used seed 11, 250 optimizer updates for each neural fit, 50 RF trees, two CPU threads, and the candidate-grouped preparation. It completed 12 model/source configurations and 24 source-target evaluations. All 72 operating-point metric rows were independently recomputed from saved predictions, and source-validation threshold selection was reproduced. All 21 tests passed. Aggregate evidence is in [validation JSON](../research/audits/training-smoke-validation.json) and [metrics CSV](../research/audits/training-smoke-metrics.csv).

The Mendeley neural fits saw 16,000 training examples, less than one complete traversal of 35,186 training rows. ISCX fits saw 15,941 examples over one completed traversal plus part of another. These budgets are deliberately small and cannot support convergence or superiority claims. Initial cross-dataset default F1 did not exceed the always-malicious baseline; interpret this only as a diagnostic at these settings.

## What changed

DQN and DDQN now share architecture, optimizer, batch size, replay capacity, class-based rewards, exploration schedule, terminal handling, target-update interval, and update budget. Only the Q-target calculation differs. Their random streams for ordering, exploration, and replay sampling are independently seeded but matched across agents. Networks are freshly initialized for every model/source/seed.

The supervised MLP uses the same 128/128 ReLU hidden architecture and two outputs. Other baselines are always-malicious, logistic regression, and Random Forest. Every model uses the same prepared input columns and source partitions. The four features are still provisional; see [preprocessing limitations](preprocessing.md). Four missingness indicators are appended and are zero on the current inputs.

Only source-training rows enter optimization or replay. Source validation selects thresholds; both source and target tests use those frozen thresholds. Target labels are used for reporting only. Model checkpoints, source-validation scores, test scores/predictions, training histories, per-run metrics, and PR/ROC curve data are saved locally.

## Environment and execution

Use an isolated Python 3.12 environment. The research dependencies are separate from the original tutorial's requirements. The validated machine used CPU PyTorch; CUDA behavior has not been tested.

```bash
python -m venv .venv-research
# Activate .venv-research using the command appropriate for your shell.
python -m pip install -r requirements-research.txt
python -m unittest discover -s tests -v
python scripts/run_research_benchmark.py --prepared work/prepared-candidate --output work/benchmark-smoke --purpose smoke --seeds 11 --updates 250 --trees 50 --threads 2
```

Prepare inputs first using [the preprocessing entry point](preprocessing.md). Run directories must be new; existing outputs are never silently replaced. A completion.json file is written only when the entire run succeeds. A failure can leave a partial directory without this marker.

An extended exploratory run can use --seeds 11 23 37 51 71 and a larger prespecified update budget. The correct budget must be established from source-only learning diagnostics; 5,000 updates is a configurable starting value, not evidence of convergence. Repeated runs should not be selected according to target-test results.

## Neural training protocol

- Network: input -> 128 ReLU -> 128 ReLU -> two linear outputs.
- Adam learning rate 0.001; gradient norm clipping 1.0; batch size 64.
- Replay capacity 10,000; target network synchronized every 100 optimizer updates.
- Exploration starts at 1.0 and decays as 0.995^update with a floor of 0.05, identically for both RL algorithms.
- Each collection batch traverses the current seeded shuffled source-training order. At the end of a complete traversal, the transition is terminal and the order is reshuffled. No state transition depends on the prediction.
- Each collection batch is followed by one replay update. The last batch of a traversal may be shorter. Sample exposures and completed traversals are recorded.
- Gamma defaults to 0.99. This is a classification environment with action-independent transitions, not evidence of temporal adaptation. --gamma 0 provides a contextual-classification control.
- Symmetric reward is +1 for correct and -1 for incorrect predictions. Optional --weighting balanced multiplies this by n/(2*n_class) for the true source-training class, for both algorithms. It uses no row-position groups. The same option weights the supervised objectives.
- MLP uses cross-entropy. DQN and DDQN use MSE against detached Q-targets. The smoke comparison has no tuning, early stopping, SMOTE, or feature selection. A later exploratory run selects a common neural update budget using source validation as described below.

Equal neural update budgets are a controlled diagnostic, not proof of equivalent optimization difficulty. Logistic regression and Random Forest have different fitting procedures; compare recorded time and iterations alongside predictive metrics.

## Scores and operating points

RL scores are Q(1)-Q(0), MLP scores are logit(1)-logit(0), logistic scores are decision values, and RF scores are class-1 probabilities. Q-values and logits are not calibrated probabilities. Average precision and ROC-AUC use the continuous scores.

Three operating points are saved:

1. Default argmax/classifier decision, including class-0 handling of exact score ties.
2. Maximum F1 on source validation. Ties choose the highest threshold.
3. Maximum recall on source validation with FPR <= 0.01. Ties prefer lower FPR, then higher threshold. A predict-none threshold is allowed and its zero recall is reported honestly.

The initial validation sets contain 1,556 ISCX and 5,600 Mendeley class-0 examples. A 1% constraint is exploratory and still has sampling uncertainty. The achieved target FPR can exceed the source limit; the test result is never used to adjust the threshold. PR curve CSVs contain points with finite decision thresholds; the conventional terminal precision=1, recall=0 plotting endpoint is implicit.

For each operating point, metrics include confusion counts, precision/recall/F1, accuracy, balanced accuracy, FPR, specificity, predicted-positive fraction, AP, AUC, fit time, and inference time. Summary SD is the sample SD across seeds and is undefined for a single seed. No significance tests or superiority claims are produced by the smoke run.

## Artifact integrity and interpretation

The loader checks prepared-file hashes, feature order, finite inputs, class labels, and sample-ID separation. Run metadata records preparation-manifest hash, script hash, exact software versions, hardware, settings, and intended purpose. Neither input CSVs nor original manuscript scripts are edited. Existing external test sets have already informed redesign, and the newly shuffled partitions are not an independent new dataset.

Keep checkpoints and prediction-level files local. Only aggregate smoke validation evidence should be committed by default. Do not deserialize unknown joblib/PyTorch files; this runner writes its own checkpoints and does not load prior model artifacts.

Next scientific work remains: verify source feature definitions and label provenance; establish convergence; conduct full tuning and ablations; and seek an independent confirmation dataset.


## Five-seed exploratory comparison

The completed source-validation budget pilot and five-seed results are documented in [five-seed results](five-seed-results.md). Reproduce on the candidate-grouped preparation with:

```powershell
python scripts/select_training_budget.py --prepared work/prepared-candidate-v2 --output work/budget-pilot-v1 --candidates 1000 2000 4000 --seed 101 --threads 2
python scripts/run_research_benchmark.py --prepared work/prepared-candidate-v2 --output work/benchmark-five-v1 --purpose exploratory --seeds 11 23 37 51 71 --budget-selection work/budget-pilot-v1/selection.json --trees 200 --threads 2
python scripts/verify_benchmark.py --run work/benchmark-five-v1 --output work/verified-five-v1
```

Use fresh output directories for training. The budget-selection input must match preparation hash, discount factor, batch size, learning rate, and reward weighting. The pilot selects a common update budget, not all model hyperparameters; source validation is subsequently reused for decision thresholds. Full hyperparameter tuning, convergence checks and ablations remain pending. The verifier recalculates saved thresholds and metrics using the shared metric routines, checks saved predictions and matched RL training budgets, and produces aggregate mean/sample-SD and paired DDQN-minus-DQN summaries. It is an artifact consistency check, not an independent reimplementation of the metric formulas.


## URL-length and ratio ablation

A separate five-seed diagnostic now removes URL length and domain/URL ratio by zeroing those inputs and their missingness indicators. The original partitions, architecture and training budget are retained. See [ablation results](ablation-length-results.md) for the paired comparison, verification and limitations. This completes one feature-dependence diagnostic, not the full tuning/reward/SMOTE ablation suite.


## Discount-factor diagnostic and experiment register

The completed [gamma-zero comparison](ablation-gamma-results.md) keeps the original features, five seeds and 1,000-update budget. Its 360 metric rows and 180 paired prediction files were verified; gamma-zero DQN/DDQN predictions and histories match exactly. See the [experiment register](experiment-register.md) and artifact inventory for all recorded stages. This does not complete full discount tuning or the remaining reward/resampling ablations.
