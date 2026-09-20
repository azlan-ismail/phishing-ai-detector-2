# Five-seed comparison at longer matched budgets

## Outcome

Longer training does not establish a consistent DDQN advantage. Mean within-source DDQN AP changes from .969042 to .969793 on ISCX and from .887556 to .887286 on Mendeley. DQN shows a similar pattern (.969299 to .970053; .887859 to .887654). The Mendeley RL gains in the single-seed validation pilot did not translate into mean test-AP gains across the five reporting seeds; this comparison does not by itself identify why.

MLP improves mean AP on both sources, particularly Mendeley (.899568 to .909980). Random Forest retains the highest within-source mean AP: .975892 on ISCX and .913462 on Mendeley. Rankings depend on the metric: MLP has the highest default-threshold mean F1 on Mendeley (.863421), compared with Random Forest .858749 and DDQN .857486. These are descriptive means, not significance claims.

Transfer remains weak. DDQN AP is .411137 for ISCX to Mendeley and .550762 for Mendeley to ISCX. Under thresholds chosen for source-validation FPR at most 1%, target FPR is .964143 and .366967. Under default thresholds, every Mendeley-trained DQN/DDQN seed predicts every ISCX test row as malicious, producing the same F1 (.661002) as the always-malicious control. This illustrates why F1 alone is insufficient here.

## Frozen protocol

The budget rule was fixed before these runs: take the largest source-validation-selected checkpoint among MLP, DQN and DDQN for each source, then give all three that update count. ISCX uses 16,000 and Mendeley 28,000. This is a matched-update comparison, not individually optimized stopping or equal compute.

All models use seeds 11, 23, 37, 51 and 71 on the same candidate-grouped partitions. The original tuned settings are retained: neural learning rate .001, batch size 64, gamma .99, unweighted training; logistic C=100; Random Forest depth 8 with leaf size 5 on ISCX and 1 on Mendeley, 200 trees. No early stopping occurs in these five-seed fits. Imputation/scaling is fitted only on source training, and each model's decision thresholds are selected only on source validation and frozen for both tests.

The run completed 60 model/source/seed configurations, 120 source-target evaluations and 360 operating-point metric rows, including trivial controls. The 30 neural fits performed 660,000 optimizer updates in total. Earlier results are preserved.

## Verification

All 34 automated tests passed. All 360 metric rows and source-selected thresholds were recomputed from saved predictions. All 180 validation/test prediction files matched the reference sample identities and labels. Every neural training history and actual update budget was checked, and DQN/DDQN sample exposures match within each source/seed.

All 90 non-neural control prediction files agree within absolute score tolerance 1e-12 (maximum difference 4.44e-16); their decisions and metrics are unchanged. Metric verification shares the training metric library and establishes artifact consistency, rather than an independent implementation of every formula.

[Public aggregate evidence](../research/results/matched-long/) contains the frozen policy, per-seed metrics, mean/sample-SD summaries, paired changes versus the 8,000-update reference, paired DDQN-minus-DQN summaries, configuration and verification hashes. Predictions, checkpoints, histories and curve points remain local and are indexed in the [artifact inventory](../research/results/artifact-inventory.json). Original datasets remain unchanged.

## Tables

Entries are mean +/- sample SD across five training seeds. Paired change means the longer-budget result minus the 8,000-update reference for each corresponding seed. AP is threshold-independent. The tables show default and source-FPR-constrained operating points; the public per-seed files also retain the source-validation-F1 operating point.


## default

| Source | Target | Model | Metric | 8,000 updates | Longer matched budget | Paired change |
|---|---|---|---|---:|---:|---:|
| iscx | iscx | always_malicious | average_precision | 0.4937 +/- 0.0000 | 0.4937 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | iscx | always_malicious | f1 | 0.6610 +/- 0.0000 | 0.6610 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | iscx | ddqn | average_precision | 0.9690 +/- 0.0003 | 0.9698 +/- 0.0006 | 0.0008 +/- 0.0006 |
| iscx | iscx | ddqn | f1 | 0.9047 +/- 0.0040 | 0.9095 +/- 0.0034 | 0.0048 +/- 0.0048 |
| iscx | iscx | dqn | average_precision | 0.9693 +/- 0.0003 | 0.9701 +/- 0.0006 | 0.0008 +/- 0.0005 |
| iscx | iscx | dqn | f1 | 0.9057 +/- 0.0017 | 0.9047 +/- 0.0064 | -0.0010 +/- 0.0056 |
| iscx | iscx | logistic | average_precision | 0.9407 +/- 0.0000 | 0.9407 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | iscx | logistic | f1 | 0.8366 +/- 0.0000 | 0.8366 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | iscx | mlp | average_precision | 0.9725 +/- 0.0005 | 0.9735 +/- 0.0004 | 0.0010 +/- 0.0006 |
| iscx | iscx | mlp | f1 | 0.9070 +/- 0.0039 | 0.9115 +/- 0.0066 | 0.0045 +/- 0.0068 |
| iscx | iscx | random_forest | average_precision | 0.9759 +/- 0.0004 | 0.9759 +/- 0.0004 | 0.0000 +/- 0.0000 |
| iscx | iscx | random_forest | f1 | 0.9179 +/- 0.0048 | 0.9179 +/- 0.0048 | 0.0000 +/- 0.0000 |
| iscx | mendeley | always_malicious | average_precision | 0.5226 +/- 0.0000 | 0.5226 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | mendeley | always_malicious | f1 | 0.6864 +/- 0.0000 | 0.6864 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | mendeley | ddqn | average_precision | 0.3953 +/- 0.0417 | 0.4111 +/- 0.0828 | 0.0159 +/- 0.0430 |
| iscx | mendeley | ddqn | f1 | 0.6375 +/- 0.0055 | 0.6385 +/- 0.0060 | 0.0009 +/- 0.0026 |
| iscx | mendeley | dqn | average_precision | 0.4037 +/- 0.0516 | 0.3897 +/- 0.0139 | -0.0140 +/- 0.0430 |
| iscx | mendeley | dqn | f1 | 0.6395 +/- 0.0062 | 0.6378 +/- 0.0073 | -0.0017 +/- 0.0047 |
| iscx | mendeley | logistic | average_precision | 0.3535 +/- 0.0000 | 0.3535 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | mendeley | logistic | f1 | 0.6199 +/- 0.0000 | 0.6199 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | mendeley | mlp | average_precision | 0.3439 +/- 0.0006 | 0.3438 +/- 0.0007 | -0.0001 +/- 0.0008 |
| iscx | mendeley | mlp | f1 | 0.6342 +/- 0.0047 | 0.6394 +/- 0.0033 | 0.0052 +/- 0.0056 |
| iscx | mendeley | random_forest | average_precision | 0.4306 +/- 0.0127 | 0.4306 +/- 0.0127 | 0.0000 +/- 0.0000 |
| iscx | mendeley | random_forest | f1 | 0.6469 +/- 0.0010 | 0.6469 +/- 0.0010 | 0.0000 +/- 0.0000 |
| mendeley | iscx | always_malicious | average_precision | 0.4937 +/- 0.0000 | 0.4937 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | iscx | always_malicious | f1 | 0.6610 +/- 0.0000 | 0.6610 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | iscx | ddqn | average_precision | 0.5016 +/- 0.0728 | 0.5508 +/- 0.0474 | 0.0491 +/- 0.0406 |
| mendeley | iscx | ddqn | f1 | 0.6609 +/- 0.0001 | 0.6610 +/- 0.0000 | 0.0001 +/- 0.0001 |
| mendeley | iscx | dqn | average_precision | 0.5317 +/- 0.0821 | 0.5804 +/- 0.0789 | 0.0487 +/- 0.0478 |
| mendeley | iscx | dqn | f1 | 0.6610 +/- 0.0000 | 0.6610 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | iscx | logistic | average_precision | 0.5157 +/- 0.0000 | 0.5157 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | iscx | logistic | f1 | 0.6611 +/- 0.0000 | 0.6611 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | iscx | mlp | average_precision | 0.6266 +/- 0.0104 | 0.6336 +/- 0.0270 | 0.0070 +/- 0.0309 |
| mendeley | iscx | mlp | f1 | 0.6610 +/- 0.0000 | 0.6578 +/- 0.0016 | -0.0032 +/- 0.0016 |
| mendeley | iscx | random_forest | average_precision | 0.5887 +/- 0.0044 | 0.5887 +/- 0.0044 | 0.0000 +/- 0.0000 |
| mendeley | iscx | random_forest | f1 | 0.6610 +/- 0.0000 | 0.6610 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | mendeley | always_malicious | average_precision | 0.5226 +/- 0.0000 | 0.5226 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | mendeley | always_malicious | f1 | 0.6864 +/- 0.0000 | 0.6864 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | mendeley | ddqn | average_precision | 0.8876 +/- 0.0026 | 0.8873 +/- 0.0030 | -0.0003 +/- 0.0018 |
| mendeley | mendeley | ddqn | f1 | 0.8566 +/- 0.0004 | 0.8575 +/- 0.0014 | 0.0008 +/- 0.0013 |
| mendeley | mendeley | dqn | average_precision | 0.8879 +/- 0.0032 | 0.8877 +/- 0.0046 | -0.0002 +/- 0.0015 |
| mendeley | mendeley | dqn | f1 | 0.8570 +/- 0.0008 | 0.8579 +/- 0.0016 | 0.0009 +/- 0.0017 |
| mendeley | mendeley | logistic | average_precision | 0.8709 +/- 0.0000 | 0.8709 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | mendeley | logistic | f1 | 0.8425 +/- 0.0000 | 0.8425 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | mendeley | mlp | average_precision | 0.8996 +/- 0.0012 | 0.9100 +/- 0.0013 | 0.0104 +/- 0.0014 |
| mendeley | mendeley | mlp | f1 | 0.8580 +/- 0.0013 | 0.8634 +/- 0.0018 | 0.0054 +/- 0.0018 |
| mendeley | mendeley | random_forest | average_precision | 0.9135 +/- 0.0011 | 0.9135 +/- 0.0011 | 0.0000 +/- 0.0000 |
| mendeley | mendeley | random_forest | f1 | 0.8587 +/- 0.0007 | 0.8587 +/- 0.0007 | 0.0000 +/- 0.0000 |

## validation_fpr_limit

| Source | Target | Model | Metric | 8,000 updates | Longer matched budget | Paired change |
|---|---|---|---|---:|---:|---:|
| iscx | iscx | always_malicious | recall | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | iscx | always_malicious | fpr | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | iscx | ddqn | recall | 0.6986 +/- 0.0070 | 0.7036 +/- 0.0048 | 0.0050 +/- 0.0093 |
| iscx | iscx | ddqn | fpr | 0.0084 +/- 0.0009 | 0.0096 +/- 0.0016 | 0.0013 +/- 0.0011 |
| iscx | iscx | dqn | recall | 0.6926 +/- 0.0106 | 0.7045 +/- 0.0096 | 0.0120 +/- 0.0114 |
| iscx | iscx | dqn | fpr | 0.0086 +/- 0.0021 | 0.0089 +/- 0.0005 | 0.0003 +/- 0.0021 |
| iscx | iscx | logistic | recall | 0.5214 +/- 0.0000 | 0.5214 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | iscx | logistic | fpr | 0.0077 +/- 0.0000 | 0.0077 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | iscx | mlp | recall | 0.7367 +/- 0.0113 | 0.7598 +/- 0.0116 | 0.0231 +/- 0.0153 |
| iscx | iscx | mlp | fpr | 0.0082 +/- 0.0015 | 0.0064 +/- 0.0018 | -0.0018 +/- 0.0030 |
| iscx | iscx | random_forest | recall | 0.8121 +/- 0.0055 | 0.8121 +/- 0.0055 | 0.0000 +/- 0.0000 |
| iscx | iscx | random_forest | fpr | 0.0095 +/- 0.0018 | 0.0095 +/- 0.0018 | 0.0000 +/- 0.0000 |
| iscx | mendeley | always_malicious | recall | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | mendeley | always_malicious | fpr | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | mendeley | ddqn | recall | 0.7811 +/- 0.0067 | 0.7843 +/- 0.0067 | 0.0032 +/- 0.0108 |
| iscx | mendeley | ddqn | fpr | 0.9625 +/- 0.0006 | 0.9641 +/- 0.0004 | 0.0017 +/- 0.0006 |
| iscx | mendeley | dqn | recall | 0.7718 +/- 0.0065 | 0.7839 +/- 0.0033 | 0.0121 +/- 0.0076 |
| iscx | mendeley | dqn | fpr | 0.9623 +/- 0.0006 | 0.9639 +/- 0.0008 | 0.0016 +/- 0.0008 |
| iscx | mendeley | logistic | recall | 0.6566 +/- 0.0000 | 0.6566 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | mendeley | logistic | fpr | 0.9434 +/- 0.0000 | 0.9434 +/- 0.0000 | 0.0000 +/- 0.0000 |
| iscx | mendeley | mlp | recall | 0.8030 +/- 0.0071 | 0.8123 +/- 0.0098 | 0.0093 +/- 0.0062 |
| iscx | mendeley | mlp | fpr | 0.9652 +/- 0.0007 | 0.9656 +/- 0.0005 | 0.0004 +/- 0.0004 |
| iscx | mendeley | random_forest | recall | 0.8357 +/- 0.0019 | 0.8357 +/- 0.0019 | 0.0000 +/- 0.0000 |
| iscx | mendeley | random_forest | fpr | 0.9691 +/- 0.0004 | 0.9691 +/- 0.0004 | 0.0000 +/- 0.0000 |
| mendeley | iscx | always_malicious | recall | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | iscx | always_malicious | fpr | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | iscx | ddqn | recall | 0.2866 +/- 0.0557 | 0.2984 +/- 0.0321 | 0.0117 +/- 0.0331 |
| mendeley | iscx | ddqn | fpr | 0.4504 +/- 0.1998 | 0.3670 +/- 0.1724 | -0.0834 +/- 0.0833 |
| mendeley | iscx | dqn | recall | 0.2875 +/- 0.0403 | 0.3274 +/- 0.0141 | 0.0398 +/- 0.0292 |
| mendeley | iscx | dqn | fpr | 0.4004 +/- 0.2158 | 0.3667 +/- 0.1945 | -0.0337 +/- 0.1302 |
| mendeley | iscx | logistic | recall | 0.2577 +/- 0.0000 | 0.2577 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | iscx | logistic | fpr | 0.2995 +/- 0.0000 | 0.2995 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | iscx | mlp | recall | 0.3612 +/- 0.0198 | 0.3578 +/- 0.0206 | -0.0034 +/- 0.0355 |
| mendeley | iscx | mlp | fpr | 0.1704 +/- 0.0720 | 0.2258 +/- 0.0875 | 0.0554 +/- 0.1413 |
| mendeley | iscx | random_forest | recall | 0.2989 +/- 0.0044 | 0.2989 +/- 0.0044 | 0.0000 +/- 0.0000 |
| mendeley | iscx | random_forest | fpr | 0.3571 +/- 0.0170 | 0.3571 +/- 0.0170 | 0.0000 +/- 0.0000 |
| mendeley | mendeley | always_malicious | recall | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | mendeley | always_malicious | fpr | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | mendeley | ddqn | recall | 0.1805 +/- 0.0104 | 0.1870 +/- 0.0048 | 0.0065 +/- 0.0138 |
| mendeley | mendeley | ddqn | fpr | 0.0107 +/- 0.0021 | 0.0106 +/- 0.0011 | -0.0000 +/- 0.0010 |
| mendeley | mendeley | dqn | recall | 0.1830 +/- 0.0128 | 0.1986 +/- 0.0085 | 0.0156 +/- 0.0110 |
| mendeley | mendeley | dqn | fpr | 0.0107 +/- 0.0023 | 0.0110 +/- 0.0014 | 0.0003 +/- 0.0015 |
| mendeley | mendeley | logistic | recall | 0.1250 +/- 0.0000 | 0.1250 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | mendeley | logistic | fpr | 0.0163 +/- 0.0000 | 0.0163 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | mendeley | mlp | recall | 0.2276 +/- 0.0084 | 0.2442 +/- 0.0150 | 0.0166 +/- 0.0143 |
| mendeley | mendeley | mlp | fpr | 0.0117 +/- 0.0004 | 0.0111 +/- 0.0009 | -0.0006 +/- 0.0008 |
| mendeley | mendeley | random_forest | recall | 0.2337 +/- 0.0078 | 0.2337 +/- 0.0078 | 0.0000 +/- 0.0000 |
| mendeley | mendeley | random_forest | fpr | 0.0094 +/- 0.0012 | 0.0094 +/- 0.0012 | 0.0000 +/- 0.0000 |



## Limits and manuscript decision

Five-seed SD reflects training randomness on fixed partitions, not uncertainty across independent datasets. The budgets derive from one reused validation pilot seed; these source-specific budgets are not a claim of mathematical convergence or optimality. Prior test inspection informed the overall research redesign, so this is exploratory evidence.

Cross-source feature extraction equivalence and URL/domain identity remain unresolved. Keep the transfer results as exploratory diagnostics. Do not infer gamma effects by comparing this gamma-.99 stage against an earlier gamma-zero stage with different update budgets.

The accumulated evidence does not justify framing the manuscript around DDQN superiority. A defensible revision should emphasize the controlled comparison and its limitations, with claims assessed separately by AP, threshold-dependent performance and transfer. Reassess that contribution before adding further model variants; stronger evaluation alone does not establish novelty.

## Reproduction

Use the recorded environment, prepared exports and original source-only tuning selection. Preserve completed output directories and use new ones for reruns. The frozen policy includes the exact preparation, tuning-selection and convergence-result hashes.

```text
python scripts/run_research_benchmark.py --prepared work/prepared-candidate-v2 --output work/benchmark-matched-long-v1 --purpose exploratory --seeds 11 23 37 51 71 --tuning-selection work/tuning-source-v1/selection.json --matched-budget-policy work/matched-budget-policy-v1.json --trees 200 --threads 2
python scripts/verify_benchmark.py --run work/benchmark-matched-long-v1 --output work/verified-matched-long-v1
python scripts/compare_matched_budgets.py --baseline work/benchmark-tuned-v1 --extended work/benchmark-matched-long-v1 --policy work/matched-budget-policy-v1.json --output work/compared-matched-long-v1
python scripts/record_experiment_artifacts.py --work work --output research/results/artifact-inventory.json
```

The policy is archived publicly under research/results/matched-long/policy.json. Integrity hashes describe the original local files, including their byte representation. A fresh independently regenerated preparation/selection requires its own frozen policy with the new hashes; never bypass mismatch checks.

