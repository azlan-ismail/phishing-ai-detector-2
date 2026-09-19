# Five-seed exploratory benchmark

**Provenance update (19 September 2026):** Mendeley's published labels are now confirmed as 0 = legitimate and 1 = phishing. The ISCX file is treated as the IEEE file renamed without content changes, per the user's explicit assumption; this has not been independently byte-verified. Source papers support the four intended feature meanings, but exact extraction equivalence remains unverified. The [feature-compatibility audit](feature-compatibility.md) documents strong class-conditional distribution differences. This update supersedes earlier statements below that Mendeley's published label meaning is unresolved; historical results and manifests are unchanged.


This is an actual Python/PyTorch/scikit-learn run on the supplied ISCX and Mendeley numeric tables. It extends the earlier smoke test with a source-validation budget pilot and five reporting seeds. It is not a reproduction of the manuscript's original experiments or a confirmatory benchmark.

## Design and budget selection

The candidate-feature grouping preparation fixes 60/20/20 train/validation/test partitions and groups identical reduced feature vectors within each dataset. The four provisional features are URL length, domain length, URL dot count, and the recomputed domain/URL length ratio; four fixed missingness indicators are also included. Preprocessing is fitted on source training data only. The run covers six models and all four source-to-test scenarios, with seeds 11, 23, 37, 51 and 71 on the same partitions.

Before the five-seed run, pilot seed 101 compared 1,000, 2,000 and 4,000 optimizer updates for MLP, DQN and DDQN on both sources (18 fits). The rule was saved before fitting: select the smallest common budget within 0.005 absolute average precision (AP) of the best equally weighted mean over the six source/model pairs. The selector reads training and source-validation exports only.

| Updates | Mean source-validation AP |
|---|---:|
| 1,000 | 0.931722 |
| 2,000 | 0.934307 |
| 4,000 | 0.936462 |

The rule selects **1,000 updates**. The best mean is still at the largest candidate, so convergence and optimality are **not established**. This is a compute-budget rule rather than comprehensive hyperparameter tuning. The run metadata field `tuning: none` refers to the absence of further tuning; the separate budget-selection hash records this validation-based choice.

All neural models use the same 128/128 hidden architecture, Adam learning rate 0.001, batch size 64 and 1,000 updates. DQN/DDQN additionally share exploration, replay, target synchronization, discount 0.99, clipping and unweighted class-based rewards. Only their Q-target construction differs. RF uses 200 trees. No SMOTE is applied. See [training implementation](training.md) for full settings and reproduction commands.

Source validation is reused to choose maximum-F1 and FPR-constrained thresholds. Those thresholds are frozen before within-source and transferred test evaluation. Reported AP uses continuous scores and is independent of the operating threshold.

## Results

Values are mean ± sample SD across five training seeds. SD measures training randomness on these fixed partitions, not uncertainty across datasets, domains or alternative test samples. Always-malicious and logistic results can have zero SD. AP is average precision, not trapezoidal PR area. F1 below uses each model's default decision rule.

### Default decisions

| Source | Test | Model | average_precision | f1 |
|---|---|---|---|---|
| iscx | iscx | always_malicious | 0.4937 ± 0.0000 | 0.6610 ± 0.0000 |
| iscx | iscx | ddqn | 0.9627 ± 0.0016 | 0.9030 ± 0.0069 |
| iscx | iscx | dqn | 0.9628 ± 0.0016 | 0.9034 ± 0.0077 |
| iscx | iscx | logistic | 0.9365 ± 0.0000 | 0.8256 ± 0.0000 |
| iscx | iscx | mlp | 0.9667 ± 0.0003 | 0.9037 ± 0.0066 |
| iscx | iscx | random_forest | 0.9703 ± 0.0006 | 0.9009 ± 0.0007 |
| iscx | mendeley | always_malicious | 0.5226 ± 0.0000 | 0.6864 ± 0.0000 |
| iscx | mendeley | ddqn | 0.4318 ± 0.1097 | 0.6231 ± 0.0074 |
| iscx | mendeley | dqn | 0.4199 ± 0.1149 | 0.6231 ± 0.0072 |
| iscx | mendeley | logistic | 0.3516 ± 0.0000 | 0.6137 ± 0.0000 |
| iscx | mendeley | mlp | 0.3461 ± 0.0011 | 0.6270 ± 0.0069 |
| iscx | mendeley | random_forest | 0.4461 ± 0.0031 | 0.6478 ± 0.0007 |
| mendeley | iscx | always_malicious | 0.4937 ± 0.0000 | 0.6610 ± 0.0000 |
| mendeley | iscx | ddqn | 0.4470 ± 0.0527 | 0.6609 ± 0.0003 |
| mendeley | iscx | dqn | 0.4520 ± 0.0521 | 0.6609 ± 0.0002 |
| mendeley | iscx | logistic | 0.4725 ± 0.0000 | 0.6607 ± 0.0000 |
| mendeley | iscx | mlp | 0.5970 ± 0.0273 | 0.6608 ± 0.0003 |
| mendeley | iscx | random_forest | 0.5467 ± 0.0159 | 0.6544 ± 0.0008 |
| mendeley | mendeley | always_malicious | 0.5226 ± 0.0000 | 0.6864 ± 0.0000 |
| mendeley | mendeley | ddqn | 0.8779 ± 0.0026 | 0.8557 ± 0.0011 |
| mendeley | mendeley | dqn | 0.8780 ± 0.0022 | 0.8552 ± 0.0018 |
| mendeley | mendeley | logistic | 0.8703 ± 0.0000 | 0.8423 ± 0.0000 |
| mendeley | mendeley | mlp | 0.8894 ± 0.0015 | 0.8578 ± 0.0012 |
| mendeley | mendeley | random_forest | 0.9026 ± 0.0018 | 0.8566 ± 0.0003 |
### Frozen source-validation FPR limit

The source-validation constraint is FPR <= 0.01. Values below are achieved **test** recall and FPR. The constant-score reference selects predict-none here, so its zero FPR comes with zero recall.

| Source | Test | Model | recall | fpr |
|---|---|---|---|---|
| iscx | iscx | always_malicious | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| iscx | iscx | ddqn | 0.5991 ± 0.0277 | 0.0080 ± 0.0015 |
| iscx | iscx | dqn | 0.5955 ± 0.0342 | 0.0078 ± 0.0014 |
| iscx | iscx | logistic | 0.4911 ± 0.0000 | 0.0071 ± 0.0000 |
| iscx | iscx | mlp | 0.6691 ± 0.0070 | 0.0085 ± 0.0003 |
| iscx | iscx | random_forest | 0.8196 ± 0.0043 | 0.0195 ± 0.0004 |
| iscx | mendeley | always_malicious | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| iscx | mendeley | ddqn | 0.7038 ± 0.0162 | 0.9561 ± 0.0014 |
| iscx | mendeley | dqn | 0.7033 ± 0.0182 | 0.9544 ± 0.0044 |
| iscx | mendeley | logistic | 0.6280 ± 0.0000 | 0.9411 ± 0.0000 |
| iscx | mendeley | mlp | 0.7419 ± 0.0045 | 0.9588 ± 0.0007 |
| iscx | mendeley | random_forest | 0.8426 ± 0.0052 | 0.9705 ± 0.0001 |
| mendeley | iscx | always_malicious | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| mendeley | iscx | ddqn | 0.2376 ± 0.0338 | 0.5138 ± 0.1102 |
| mendeley | iscx | dqn | 0.2388 ± 0.0349 | 0.5122 ± 0.1041 |
| mendeley | iscx | logistic | 0.2432 ± 0.0000 | 0.3650 ± 0.0000 |
| mendeley | iscx | mlp | 0.3798 ± 0.0248 | 0.2261 ± 0.0568 |
| mendeley | iscx | random_forest | 0.4100 ± 0.0089 | 0.3125 ± 0.0081 |
| mendeley | mendeley | always_malicious | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| mendeley | mendeley | ddqn | 0.1498 ± 0.0128 | 0.0151 ± 0.0011 |
| mendeley | mendeley | dqn | 0.1505 ± 0.0145 | 0.0149 ± 0.0015 |
| mendeley | mendeley | logistic | 0.1171 ± 0.0000 | 0.0164 ± 0.0000 |
| mendeley | mendeley | mlp | 0.1880 ± 0.0059 | 0.0111 ± 0.0005 |
| mendeley | mendeley | random_forest | 0.1850 ± 0.0076 | 0.0065 ± 0.0006 |

### Interpretation

Random Forest has the highest mean within-source AP on both ISCX (0.9703) and Mendeley (0.9026), followed by the supervised MLP. DDQN and DQN have almost identical within-source AP and F1. Paired DDQN-minus-DQN AP differences (mean ± sample SD) are -0.000064 ± 0.000804 for ISCX and -0.000163 ± 0.001640 for Mendeley; these do not support a consistent DDQN advantage.

Transfer is weak. ISCX-to-Mendeley default F1 is below the always-malicious reference for every trained model. In the reverse direction, neural default F1 is essentially the always-malicious value; the supervised MLP nevertheless has better mean ranking AP (0.5970) than that constant-score reference (0.4937). Ranking and operating-point performance must be distinguished.

The frozen low-FPR thresholds fail under transfer: DDQN's mean FPR is 0.9561 for ISCX-to-Mendeley and 0.5138 in the reverse direction. Even within-source, a validation constraint does not guarantee the same test FPR (ISCX RF: 0.0195). These results do not establish useful cross-dataset deployment performance. Feature-definition or label mismatches remain possible explanations alongside distribution shift; this experiment cannot isolate those causes.

## Verification and evidence

The completed run contains 60 model/source/seed configurations, including the trivial reference, 120 source-target evaluations, and 360 operating-point metric rows. The verifier reproduces source-validation thresholds, recalculates metrics from saved scores, checks prediction columns, and checks equal DQN/DDQN training budgets. It uses the runner's metric routines, so this checks saved-artifact consistency rather than an independent implementation of all formulas. All 23 automated tests pass.

Aggregate evidence is under [research/results/five-seed](../research/results/five-seed/): budget policy, pilot fits, selected budget, per-seed test metrics, means/SDs, paired DDQN-minus-DQN differences and verification metadata. Raw inputs, row-level predictions and trained weights remain local.

## Limits and next scientific steps

- Feature equivalence and Mendeley numeric-label provenance still need source documentation. The raw tables contain no URLs/domain identifiers, so reduced-feature grouping is not proof of domain independence or cross-dataset decontamination.
- These partitions were made after earlier test results informed the redesign. The pilot excludes test exports, but the study as a whole is exploratory; an independent confirmation dataset remains necessary.
- The common neural budget was chosen from one pilot seed and an average over models/sources. It may hide slower convergence for a particular model. No full hyperparameter search, repeated split evaluation, SMOTE/reward/feature ablations, or discount-zero experimental comparison was run here.
- A 1% source-validation FPR constraint is a selection rule with finite-sample uncertainty; it does not guarantee 1% FPR on the source test or transfer dataset.
- Paired seed differences are descriptive. No significance or causal claim follows from five fixed-partition seeds. Static feature-table training also does not establish benefits for temporal adaptation or interactive decision-making.

Prioritize verifying the feature/label definitions, then source-only tuning and controlled ablations. Any manuscript claim should distinguish within-source discrimination from transferred performance and should be supported against the supervised and trivial baselines.


## URL-length and ratio ablation

A separate five-seed diagnostic now removes URL length and domain/URL ratio by zeroing those inputs and their missingness indicators. The original partitions, architecture and training budget are retained. See [ablation results](ablation-length-results.md) for the paired comparison, verification and limitations. This completes one feature-dependence diagnostic, not the full tuning/reward/SMOTE ablation suite.
