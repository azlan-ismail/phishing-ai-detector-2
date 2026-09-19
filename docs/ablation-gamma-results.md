# Discount-factor ablation: gamma 0 versus 0.99

This run tests whether bootstrapped future values help in the current static URL-classification setup. The four-feature baseline uses gamma 0.99; the new run uses gamma 0. Both use the same eight input positions (four values and four missingness indicators), prepared partitions, five seeds, network architecture, optimizer, replay, rewards, 1,000 neural updates and 200 RF trees. This is separate from the earlier feature-removal ablation.

The 1,000-update budget is inherited from the gamma-0.99 source-validation pilot. It is deliberately not reselected for gamma zero. The new runner uses explicit `--updates 1000`, so its `budget_selection_sha256` is null: the pilot configuration guard correctly would reject applying a gamma-0.99 selection as if it had tuned gamma zero. All thresholds are selected separately on source validation and frozen for testing.

All six models are rerun. Always-malicious, logistic regression, RF and MLP are controls; gamma only affects the RL target. At gamma zero both RL targets reduce to the immediate reward. With matching initialization, data, exploration and replay, DQN/DDQN should then coincide. The comparison checks that property on all five seeds and both source datasets, including training histories.

This is a fixed-budget exploratory comparison. The shuffled feature-table environment does not provide action-dependent state transitions or evidence about real temporal adaptation. An improvement from bootstrapping in this setup would not by itself establish the necessity of sequential reinforcement learning.

## Results

Values below are five-seed means. Full per-seed metrics, sample SD, confusion counts, threshold values and paired gamma-zero-minus-gamma-0.99 differences are saved in [aggregate results](../research/results/ablation-gamma-zero/). AP means average precision. Default F1 and source-validation-constrained operating points are distinguished.


### Default decision rule

| Source | Test | Model | average_precision: gamma .99 / gamma 0 / change | f1: gamma .99 / gamma 0 / change |
|---|---|---|---|---|
| iscx | iscx | ddqn | 0.9627 / 0.9656 / +0.0028 | 0.9030 / 0.9058 / +0.0028 |
| iscx | iscx | dqn | 0.9628 / 0.9656 / +0.0028 | 0.9034 / 0.9058 / +0.0024 |
| iscx | mendeley | ddqn | 0.4318 / 0.3465 / -0.0853 | 0.6231 / 0.6249 / +0.0018 |
| iscx | mendeley | dqn | 0.4199 / 0.3465 / -0.0734 | 0.6231 / 0.6249 / +0.0018 |
| mendeley | iscx | ddqn | 0.4470 / 0.5156 / +0.0687 | 0.6609 / 0.6608 / -0.0001 |
| mendeley | iscx | dqn | 0.4520 / 0.5156 / +0.0636 | 0.6609 / 0.6608 / -0.0001 |
| mendeley | mendeley | ddqn | 0.8779 / 0.8851 / +0.0072 | 0.8557 / 0.8560 / +0.0003 |
| mendeley | mendeley | dqn | 0.8780 / 0.8851 / +0.0071 | 0.8552 / 0.8560 / +0.0008 |

### Frozen source-validation FPR constraint

| Source | Test | Model | recall: gamma .99 / gamma 0 / change | fpr: gamma .99 / gamma 0 / change |
|---|---|---|---|---|
| iscx | iscx | ddqn | 0.5991 / 0.6432 / +0.0442 | 0.0080 / 0.0089 / +0.0009 |
| iscx | iscx | dqn | 0.5955 / 0.6432 / +0.0477 | 0.0078 / 0.0089 / +0.0010 |
| iscx | mendeley | ddqn | 0.7038 / 0.7240 / +0.0202 | 0.9561 / 0.9574 / +0.0012 |
| iscx | mendeley | dqn | 0.7033 / 0.7240 / +0.0207 | 0.9544 / 0.9574 / +0.0030 |
| mendeley | iscx | ddqn | 0.2376 / 0.3049 / +0.0674 | 0.5138 / 0.4869 / -0.0269 |
| mendeley | iscx | dqn | 0.2388 / 0.3049 / +0.0662 | 0.5122 / 0.4869 / -0.0253 |
| mendeley | mendeley | ddqn | 0.1498 / 0.1853 / +0.0355 | 0.0151 / 0.0142 / -0.0009 |
| mendeley | mendeley | dqn | 0.1505 / 0.1853 / +0.0348 | 0.0149 / 0.0142 / -0.0006 |

## Findings

At gamma zero, DQN and DDQN have identical saved scores, decisions and training histories for every source/seed pair: all 40 checked files match exactly. Gamma-zero AP mean ± sample SD is 0.96556 ± 0.00058 within ISCX and 0.88512 ± 0.00360 within Mendeley. Relative to DDQN at gamma .99, the respective mean AP changes are +0.0028 and +0.0072. Default F1 changes are small.

Transfer ranking is mixed: ISCX-to-Mendeley AP falls from 0.4318 to 0.3465, whereas Mendeley-to-ISCX AP rises from 0.4470 to 0.5156. Default F1 remains near 0.625 and 0.661, respectively. Gamma zero does not resolve transfer failure. At source-validation thresholds constrained to at most 1% FPR, target mean FPR remains 0.9574 and 0.4869, respectively.

These fixed-budget results do not show a consistent predictive benefit from the future-value term. They also do not prove that gamma zero is universally optimal or that every reinforcement-learning approach is unsuitable. The supervised MLP and RF retain higher within-source mean AP than the gamma-zero RL models on both datasets.

Verification recomputed all 360 metric rows and source-validation thresholds. All 180 paired validation/test prediction files have matching sample IDs and labels. The 120 supervised/trivial control files agree in scores within absolute tolerance 1e-12; the maximum difference is 2.220446049250313e-16. All control decisions and the eight compared metric values are unchanged. All 24 automated tests pass, including the gamma-zero equivalence unit test. Runtime differences are recorded but are not controlled timing benchmarks.

## Recording and verification

The run retains all model checkpoints, continuous scores, predictions, source-validation threshold choices, training histories, PR/ROC curve data, timings, configuration, software versions and script/preparation checksums locally. Public aggregate records include 360 metric rows, summary mean/SD, paired changes and verification outputs. The metric verifier shares the runner's metric routines, so it verifies artifact consistency rather than independently reimplementing all metrics.

The [experiment register](experiment-register.md) links the completed stages. Its [artifact inventory](../research/results/artifact-inventory.json) records relative filenames, sizes and SHA-256 hashes of the retained local run files. Hashes do not make the private artifacts publicly downloadable. No raw data or row-level predictions are published.

## Reproduction

Use the same prepared inputs and original baseline run. Output folders must be new for training.

```text
python scripts/run_research_benchmark.py --prepared work/prepared-candidate-v2 --output work/ablation-gamma-zero-v1 --purpose exploratory --seeds 11 23 37 51 71 --updates 1000 --gamma 0 --trees 200 --threads 2 --feature-condition all
python scripts/verify_benchmark.py --run work/ablation-gamma-zero-v1 --output work/verified-ablation-gamma-zero-v1
python scripts/compare_gamma_ablation.py --baseline work/benchmark-five-v1 --ablation work/ablation-gamma-zero-v1 --output work/compared-ablation-gamma-zero-v1
python scripts/record_experiment_artifacts.py --work work --output research/results/artifact-inventory.json
```

All reported differences are descriptive across matched training seeds on fixed partitions, not significance tests or independent dataset uncertainty. Dataset extraction compatibility and independent confirmation remain open. No target-test metric was used to retune this run or choose its discount factor.
