# Bounded source-validation tuning and longer training

Actual Python runs completed and verified on 20 September 2026. This exploratory stage preserves the earlier results and adds source-only model selection followed by five-seed evaluation.

## Protocol

The same candidate-grouped 60/20/20 partitions, four provisional features, source-fitted preprocessing, gamma 0.99, unweighted training, batch size 64 and 200 RF trees were retained. Six candidates were evaluated for each of five trained models on each source (60 candidate evaluations). Selection maximized source-validation average precision (AP), with exact ties choosing the earliest candidate. The selector did not open test exports.

- Logistic regression: C = 0.001, 0.01, 0.1, 1, 10, 100.
- Random Forest: maximum depth 8, 16 or unlimited, crossed with minimum leaf size 1 or 5.
- MLP, DQN and DDQN: learning rate 0.0003 or 0.001, crossed with 1,000, 4,000 or 8,000 optimizer updates.

Seed 101 was used for selection, reusing the earlier pilot seed. Neural candidates were checkpoints from 12 trajectories; the other candidates required 24 supervised fits. Equal candidate counts do not mean equal compute. Neural trial timings are cumulative within trajectories and must not be summed as independent fit costs.

Selected settings were frozen before final fits with seeds 11, 23, 37, 51 and 71. Each fit selected operating thresholds on source validation and applied them to both tests. DQN and DDQN independently selected the same learning rate and budget, retaining a matched comparison in this run.

## Findings

All trained models improved mean within-dataset AP. Random Forest remained highest: 0.9759 on ISCX and 0.9135 on Mendeley, versus DDQN 0.9690 and 0.8876. These descriptive comparisons do not establish statistical significance.

Cross-dataset performance remains problematic. DDQN AP changed from 0.4318 to 0.3953 for ISCX to Mendeley and from 0.4470 to 0.5016 in the reverse direction. At thresholds chosen for source-validation FPR at most 1%, DDQN target FPR was 96.25% and 45.04%, respectively. The source constraint is not a target guarantee.

All six neural selections reached 8,000 updates, the largest tested budget. Both logistic selections reached the largest C, while RF selected the shallowest tested depth. This bounded search establishes neither convergence nor a global optimum. The saved validation learning curves document the remaining gains from 4,000 to 8,000 updates.


## Selected settings

| Source | Model | Selected setting | Validation AP |
|---|---|---|---:|
| iscx | ddqn | {"learning_rate": 0.001, "updates": 8000} | 0.969625 |
| iscx | dqn | {"learning_rate": 0.001, "updates": 8000} | 0.969937 |
| iscx | logistic | {"C": 100.0} | 0.940290 |
| iscx | mlp | {"learning_rate": 0.001, "updates": 8000} | 0.973411 |
| iscx | random_forest | {"max_depth": 8, "min_samples_leaf": 5} | 0.975924 |
| mendeley | ddqn | {"learning_rate": 0.001, "updates": 8000} | 0.901037 |
| mendeley | dqn | {"learning_rate": 0.001, "updates": 8000} | 0.898883 |
| mendeley | logistic | {"C": 100.0} | 0.885482 |
| mendeley | mlp | {"learning_rate": 0.001, "updates": 8000} | 0.911828 |
| mendeley | random_forest | {"max_depth": 8, "min_samples_leaf": 1} | 0.916422 |

## Final test comparison

Means across five seeds. Changes are tuned minus untuned.

### default

| Source | Test | Model | average_precision: baseline / tuned / change | f1: baseline / tuned / change |
|---|---|---|---|---|
| iscx | iscx | always_malicious | 0.4937 / 0.4937 / +0.0000 | 0.6610 / 0.6610 / +0.0000 |
| iscx | iscx | ddqn | 0.9627 / 0.9690 / +0.0063 | 0.9030 / 0.9047 / +0.0017 |
| iscx | iscx | dqn | 0.9628 / 0.9693 / +0.0065 | 0.9034 / 0.9057 / +0.0023 |
| iscx | iscx | logistic | 0.9365 / 0.9407 / +0.0042 | 0.8256 / 0.8366 / +0.0110 |
| iscx | iscx | mlp | 0.9667 / 0.9725 / +0.0057 | 0.9037 / 0.9070 / +0.0033 |
| iscx | iscx | random_forest | 0.9703 / 0.9759 / +0.0056 | 0.9009 / 0.9179 / +0.0170 |
| iscx | mendeley | always_malicious | 0.5226 / 0.5226 / +0.0000 | 0.6864 / 0.6864 / +0.0000 |
| iscx | mendeley | ddqn | 0.4318 / 0.3953 / -0.0366 | 0.6231 / 0.6375 / +0.0144 |
| iscx | mendeley | dqn | 0.4199 / 0.4037 / -0.0163 | 0.6231 / 0.6395 / +0.0163 |
| iscx | mendeley | logistic | 0.3516 / 0.3535 / +0.0019 | 0.6137 / 0.6199 / +0.0063 |
| iscx | mendeley | mlp | 0.3461 / 0.3439 / -0.0022 | 0.6270 / 0.6342 / +0.0073 |
| iscx | mendeley | random_forest | 0.4461 / 0.4306 / -0.0155 | 0.6478 / 0.6469 / -0.0009 |
| mendeley | iscx | always_malicious | 0.4937 / 0.4937 / +0.0000 | 0.6610 / 0.6610 / +0.0000 |
| mendeley | iscx | ddqn | 0.4470 / 0.5016 / +0.0547 | 0.6609 / 0.6609 / +0.0001 |
| mendeley | iscx | dqn | 0.4520 / 0.5317 / +0.0797 | 0.6609 / 0.6610 / +0.0001 |
| mendeley | iscx | logistic | 0.4725 / 0.5157 / +0.0432 | 0.6607 / 0.6611 / +0.0004 |
| mendeley | iscx | mlp | 0.5970 / 0.6266 / +0.0296 | 0.6608 / 0.6610 / +0.0002 |
| mendeley | iscx | random_forest | 0.5467 / 0.5887 / +0.0420 | 0.6544 / 0.6610 / +0.0066 |
| mendeley | mendeley | always_malicious | 0.5226 / 0.5226 / +0.0000 | 0.6864 / 0.6864 / +0.0000 |
| mendeley | mendeley | ddqn | 0.8779 / 0.8876 / +0.0097 | 0.8557 / 0.8566 / +0.0009 |
| mendeley | mendeley | dqn | 0.8780 / 0.8879 / +0.0098 | 0.8552 / 0.8570 / +0.0018 |
| mendeley | mendeley | logistic | 0.8703 / 0.8709 / +0.0006 | 0.8423 / 0.8425 / +0.0003 |
| mendeley | mendeley | mlp | 0.8894 / 0.8996 / +0.0102 | 0.8578 / 0.8580 / +0.0002 |
| mendeley | mendeley | random_forest | 0.9026 / 0.9135 / +0.0108 | 0.8566 / 0.8587 / +0.0021 |

### validation_fpr_limit

| Source | Test | Model | recall: baseline / tuned / change | fpr: baseline / tuned / change |
|---|---|---|---|---|
| iscx | iscx | always_malicious | 0.0000 / 0.0000 / +0.0000 | 0.0000 / 0.0000 / +0.0000 |
| iscx | iscx | ddqn | 0.5991 / 0.6986 / +0.0995 | 0.0080 / 0.0084 / +0.0004 |
| iscx | iscx | dqn | 0.5955 / 0.6926 / +0.0970 | 0.0078 / 0.0086 / +0.0008 |
| iscx | iscx | logistic | 0.4911 / 0.5214 / +0.0303 | 0.0071 / 0.0077 / +0.0006 |
| iscx | iscx | mlp | 0.6691 / 0.7367 / +0.0676 | 0.0085 / 0.0082 / -0.0003 |
| iscx | iscx | random_forest | 0.8196 / 0.8121 / -0.0075 | 0.0195 / 0.0095 / -0.0100 |
| iscx | mendeley | always_malicious | 0.0000 / 0.0000 / +0.0000 | 0.0000 / 0.0000 / +0.0000 |
| iscx | mendeley | ddqn | 0.7038 / 0.7811 / +0.0773 | 0.9561 / 0.9625 / +0.0064 |
| iscx | mendeley | dqn | 0.7033 / 0.7718 / +0.0685 | 0.9544 / 0.9623 / +0.0079 |
| iscx | mendeley | logistic | 0.6280 / 0.6566 / +0.0286 | 0.9411 / 0.9434 / +0.0023 |
| iscx | mendeley | mlp | 0.7419 / 0.8030 / +0.0611 | 0.9588 / 0.9652 / +0.0064 |
| iscx | mendeley | random_forest | 0.8426 / 0.8357 / -0.0069 | 0.9705 / 0.9691 / -0.0013 |
| mendeley | iscx | always_malicious | 0.0000 / 0.0000 / +0.0000 | 0.0000 / 0.0000 / +0.0000 |
| mendeley | iscx | ddqn | 0.2376 / 0.2866 / +0.0490 | 0.5138 / 0.4504 / -0.0634 |
| mendeley | iscx | dqn | 0.2388 / 0.2875 / +0.0488 | 0.5122 / 0.4004 / -0.1118 |
| mendeley | iscx | logistic | 0.2432 / 0.2577 / +0.0145 | 0.3650 / 0.2995 / -0.0656 |
| mendeley | iscx | mlp | 0.3798 / 0.3612 / -0.0186 | 0.2261 / 0.1704 / -0.0557 |
| mendeley | iscx | random_forest | 0.4100 / 0.2989 / -0.1111 | 0.3125 / 0.3571 / +0.0446 |
| mendeley | mendeley | always_malicious | 0.0000 / 0.0000 / +0.0000 | 0.0000 / 0.0000 / +0.0000 |
| mendeley | mendeley | ddqn | 0.1498 / 0.1805 / +0.0307 | 0.0151 / 0.0107 / -0.0044 |
| mendeley | mendeley | dqn | 0.1505 / 0.1830 / +0.0325 | 0.0149 / 0.0107 / -0.0042 |
| mendeley | mendeley | logistic | 0.1171 / 0.1250 / +0.0078 | 0.0164 / 0.0163 / -0.0002 |
| mendeley | mendeley | mlp | 0.1880 / 0.2276 / +0.0396 | 0.0111 / 0.0117 / +0.0006 |
| mendeley | mendeley | random_forest | 0.1850 / 0.2337 / +0.0487 | 0.0065 / 0.0094 / +0.0029 |



## Verification and records

- Recomputed all 60 candidate validation AP values and reproduced the selected settings.
- Verified 60 final configurations, 120 source-target evaluations and 360 metric rows from saved predictions, including source-selected thresholds.
- Matched sample identities and labels in 180 validation/test prediction files against the original baseline; checked every neural selected update count.
- Recorded per-seed metrics, mean/sample SD/count summaries, paired changes, validation learning curves, settings, software and SHA-256 integrity records.
- The 26-test suite passed, including selection ties and observer/checkpoint equivalence. Metric verification shares the training metric routines; it checks saved-artifact consistency rather than providing an independent implementation of every formula.

Public aggregate evidence is in [research/results/tuned-source](../research/results/tuned-source/). Individual predictions, checkpoints and full training histories remain local and are listed by relative filename and hash in the [artifact inventory](../research/results/artifact-inventory.json). Original datasets were not uploaded.

## Reproduction

From the repository root with the recorded Python environment and prepared data:

```text
python scripts/tune_source_models.py --prepared work/prepared-candidate-v2 --output work/tuning-source-v1
python scripts/run_research_benchmark.py --prepared work/prepared-candidate-v2 --output work/benchmark-tuned-v1 --purpose exploratory --seeds 11 23 37 51 71 --tuning-selection work/tuning-source-v1/selection.json --trees 200 --threads 2
python scripts/verify_benchmark.py --run work/benchmark-tuned-v1 --output work/verified-tuned-v1
python scripts/verify_source_tuning.py --search work/tuning-source-v1 --baseline work/benchmark-five-v1 --tuned work/benchmark-tuned-v1 --output work/compared-tuned-v1
python scripts/record_experiment_artifacts.py --work work --output research/results/artifact-inventory.json
```

Use new output directories for new experiments; preserve these completed records.

## Limits and next work

Selection used one pilot seed and reused source validation for model selection and threshold selection. Five-seed SD reflects training randomness on fixed partitions, not dataset-sampling uncertainty. Earlier target results informed the overall redesign, so this is exploratory rather than independent confirmation.

Feature extraction equivalence remains only partly verified. The ISCX filename rename remains a user-provided assumption. Candidate-vector grouping does not establish domain-independent testing.

The earlier gamma-zero ablation used 1,000 updates. Comparing it directly with this tuned 8,000-update run would confound discounting and training budget. A matched longer-budget gamma comparison and source-only convergence extension are appropriate next experiments. Reward/resampling ablations and independent confirmation remain open. Current evidence does not support a consistent DDQN advantage.
