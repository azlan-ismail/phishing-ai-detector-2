# URL-length and ratio ablation

This exploratory ablation tests dependence on URL length and domain/URL ratio after the feature audit found reversed class relationships across datasets. The hypothesis arose after examining the earlier benchmark and whole-table diagnostics. It is a post-hoc diagnostic, not a confirmatory experiment or a procedure for selecting the best feature set using target data.

## Fixed comparison

- Baseline: the existing four-feature five-seed run, unchanged.
- Ablation: set URL length, domain/URL ratio and both corresponding missingness indicators to zero after source-fitted preprocessing. Retain domain length, URL dot count and their indicators.
- Keep all eight input positions so neural architecture, parameter count and initial weights remain paired by seed. The zero columns carry no varying signal. RF also receives these constant columns, so its feature-subsampling behavior is part of this particular ablation; this is not a separately tuned two-column RF.
- Keep the exact same prepared exports, sample identities, labels, train/validation/test partitions, seeds (11, 23, 37, 51, 71), neural budget (1,000 updates), gamma (0.99), optimizer, replay and 200 RF trees.
- Reuse the budget chosen by the original source-validation pilot; do not retune it for the ablation. This controls compute but does not establish convergence for the reduced inputs.
- Refit every model and select operating thresholds from its own source-validation scores. Freeze those thresholds before either test evaluation. Reusing baseline threshold numbers would be inappropriate after changing scores.
- Do not regroup samples after removing features. That preserves paired populations but creates additional collisions of the retained two-feature representation. Such collisions are not necessarily duplicate URLs.

## Results

AP is average precision from continuous scores. Values are means across five training seeds; SD and all operating points are included in the linked aggregate CSVs. Changes below are ablation minus baseline.


### Default decision rule

| Source | Test | Model | average_precision: baseline / ablation / change | f1: baseline / ablation / change |
|---|---|---|---|---|
| iscx | iscx | always_malicious | 0.4937 / 0.4937 / +0.0000 | 0.6610 / 0.6610 / +0.0000 |
| iscx | iscx | ddqn | 0.9627 / 0.9432 / -0.0196 | 0.9030 / 0.8507 / -0.0523 |
| iscx | iscx | dqn | 0.9628 / 0.9436 / -0.0192 | 0.9034 / 0.8504 / -0.0530 |
| iscx | iscx | logistic | 0.9365 / 0.9131 / -0.0234 | 0.8256 / 0.8618 / +0.0361 |
| iscx | iscx | mlp | 0.9667 / 0.9454 / -0.0213 | 0.9037 / 0.8571 / -0.0466 |
| iscx | iscx | random_forest | 0.9703 / 0.9499 / -0.0204 | 0.9009 / 0.8717 / -0.0292 |
| iscx | mendeley | always_malicious | 0.5226 / 0.5226 / +0.0000 | 0.6864 / 0.6864 / +0.0000 |
| iscx | mendeley | ddqn | 0.4318 / 0.5912 / +0.1593 | 0.6231 / 0.6096 / -0.0135 |
| iscx | mendeley | dqn | 0.4199 / 0.5904 / +0.1705 | 0.6231 / 0.6134 / -0.0098 |
| iscx | mendeley | logistic | 0.3516 / 0.6320 / +0.2804 | 0.6137 / 0.5768 / -0.0369 |
| iscx | mendeley | mlp | 0.3461 / 0.5919 / +0.2457 | 0.6270 / 0.5966 / -0.0303 |
| iscx | mendeley | random_forest | 0.4461 / 0.5368 / +0.0907 | 0.6478 / 0.6333 / -0.0145 |
| mendeley | iscx | always_malicious | 0.4937 / 0.4937 / +0.0000 | 0.6610 / 0.6610 / +0.0000 |
| mendeley | iscx | ddqn | 0.4470 / 0.8356 / +0.3886 | 0.6609 / 0.7724 / +0.1115 |
| mendeley | iscx | dqn | 0.4520 / 0.8431 / +0.3912 | 0.6609 / 0.7866 / +0.1257 |
| mendeley | iscx | logistic | 0.4725 / 0.8893 / +0.4167 | 0.6607 / 0.8599 / +0.1992 |
| mendeley | iscx | mlp | 0.5970 / 0.7822 / +0.1852 | 0.6608 / 0.6862 / +0.0254 |
| mendeley | iscx | random_forest | 0.5467 / 0.6423 / +0.0955 | 0.6544 / 0.6030 / -0.0514 |
| mendeley | mendeley | always_malicious | 0.5226 / 0.5226 / +0.0000 | 0.6864 / 0.6864 / +0.0000 |
| mendeley | mendeley | ddqn | 0.8779 / 0.6691 / -0.2088 | 0.8557 / 0.5399 / -0.3158 |
| mendeley | mendeley | dqn | 0.8780 / 0.6674 / -0.2106 | 0.8552 / 0.5389 / -0.3162 |
| mendeley | mendeley | logistic | 0.8703 / 0.6394 / -0.2309 | 0.8423 / 0.5741 / -0.2681 |
| mendeley | mendeley | mlp | 0.8894 / 0.6797 / -0.2097 | 0.8578 / 0.5605 / -0.2973 |
| mendeley | mendeley | random_forest | 0.9026 / 0.5630 / -0.3396 | 0.8566 / 0.5334 / -0.3233 |

### Frozen source-validation FPR constraint

| Source | Test | Model | recall: baseline / ablation / change | fpr: baseline / ablation / change |
|---|---|---|---|---|
| iscx | iscx | always_malicious | 0.0000 / 0.0000 / +0.0000 | 0.0000 / 0.0000 / +0.0000 |
| iscx | iscx | ddqn | 0.5991 / 0.5510 / -0.0481 | 0.0080 / 0.0091 / +0.0012 |
| iscx | iscx | dqn | 0.5955 / 0.5647 / -0.0309 | 0.0078 / 0.0091 / +0.0013 |
| iscx | iscx | logistic | 0.4911 / 0.0976 / -0.3935 | 0.0071 / 0.0045 / -0.0026 |
| iscx | iscx | mlp | 0.6691 / 0.5869 / -0.0821 | 0.0085 / 0.0109 / +0.0024 |
| iscx | iscx | random_forest | 0.8196 / 0.6865 / -0.1332 | 0.0195 / 0.0069 / -0.0126 |
| iscx | mendeley | always_malicious | 0.0000 / 0.0000 / +0.0000 | 0.0000 / 0.0000 / +0.0000 |
| iscx | mendeley | ddqn | 0.7038 / 0.3236 / -0.3802 | 0.9561 / 0.2055 / -0.7506 |
| iscx | mendeley | dqn | 0.7033 / 0.3358 / -0.3675 | 0.9544 / 0.2254 / -0.7290 |
| iscx | mendeley | logistic | 0.6280 / 0.0274 / -0.6006 | 0.9411 / 0.0000 / -0.9411 |
| iscx | mendeley | mlp | 0.7419 / 0.3516 / -0.3904 | 0.9588 / 0.2752 / -0.6836 |
| iscx | mendeley | random_forest | 0.8426 / 0.4869 / -0.3556 | 0.9705 / 0.4210 / -0.5495 |
| mendeley | iscx | always_malicious | 0.0000 / 0.0000 / +0.0000 | 0.0000 / 0.0000 / +0.0000 |
| mendeley | iscx | ddqn | 0.2376 / 0.2186 / -0.0190 | 0.5138 / 0.0246 / -0.4892 |
| mendeley | iscx | dqn | 0.2388 / 0.2181 / -0.0207 | 0.5122 / 0.0246 / -0.4877 |
| mendeley | iscx | logistic | 0.2432 / 0.1819 / -0.0613 | 0.3650 / 0.0161 / -0.3490 |
| mendeley | iscx | mlp | 0.3798 / 0.2228 / -0.1570 | 0.2261 / 0.0246 / -0.2015 |
| mendeley | iscx | random_forest | 0.4100 / 0.1612 / -0.2488 | 0.3125 / 0.0195 / -0.2929 |
| mendeley | mendeley | always_malicious | 0.0000 / 0.0000 / +0.0000 | 0.0000 / 0.0000 / +0.0000 |
| mendeley | mendeley | ddqn | 0.1498 / 0.0883 / -0.0615 | 0.0151 / 0.0119 / -0.0032 |
| mendeley | mendeley | dqn | 0.1505 / 0.0861 / -0.0643 | 0.0149 / 0.0116 / -0.0033 |
| mendeley | mendeley | logistic | 0.1171 / 0.0659 / -0.0512 | 0.0164 / 0.0104 / -0.0061 |
| mendeley | mendeley | mlp | 0.1880 / 0.0979 / -0.0901 | 0.0111 / 0.0097 / -0.0014 |
| mendeley | mendeley | random_forest | 0.1850 / 0.0689 / -0.1161 | 0.0065 / 0.0052 / -0.0014 |

## Interpretation

Removing the two features improves mean transfer AP for every trained model in both directions, while lowering within-source AP for every trained model. The effects on operating-point metrics differ.

For DDQN, ISCX-to-Mendeley AP increases from 0.4318 to 0.5912, but default F1 falls from 0.6231 to 0.6096 and remains below the always-malicious reference (0.6864). In the reverse direction, AP rises from 0.4470 to 0.8356 and F1 from 0.6609 to 0.7724. Within-Mendeley F1 falls sharply from 0.8557 to 0.5399. The reduced-condition DDQN F1 sample SDs are 0.0057 (ISCX within-source), 0.0177 (ISCX to Mendeley), 0.0744 (Mendeley to ISCX), and 0.0678 (Mendeley within-source).

The source-validation low-FPR operating point also changes: DDQN ISCX-to-Mendeley test FPR falls from 0.9561 to 0.2055, while recall falls from 0.7038 to 0.3236. Reverse-transfer FPR falls from 0.5138 to 0.0246, with recall falling from 0.2376 to 0.2186. A lower FPR must be assessed with the corresponding recall; the 1% source constraint is still not guaranteed on the target.

Logistic regression has the highest mean transfer AP in the reduced condition in both directions (0.6320 and 0.8893). DDQN does not show a consistent advantage over DQN. These results support feature dependence as a contributor to the observed transfer behavior, but they do not isolate extraction mismatches from collection differences or establish a universally better feature set.

With the original partitions held fixed, 2,967/3,073 ISCX test rows and 11,661/11,729 Mendeley test rows match a training row on the two retained numeric features. Validation counts are 2,964/3,073 and 11,661/11,730. This loss of representation detail is expected when removing features. It limits generalization claims and is not evidence that the samples are the same URL.

## Verification and reproduction

All 24 automated tests pass, including a test that changing only excluded feature values or their indicators cannot change the ablated inputs and that the source arrays remain unmodified. The ablation verifier recomputes 360 operating-point metric rows and source-validation thresholds from saved predictions, and checks matched DQN/DDQN budgets. The comparison checks settings and matches 180 validation/test prediction files by sample ID and label against the baseline. Metric verification shares the runner's metric routines; it is an artifact consistency check rather than an independent implementation of those formulas.

Aggregate evidence: [research/results/ablation-length](../research/results/ablation-length/). Prediction-level outputs and checkpoints remain local. No source CSV or prior experiment result was overwritten.

From a checkout with the local prepared inputs and baseline run available:

```text
python scripts/run_research_benchmark.py --prepared work/prepared-candidate-v2 --output work/ablation-length-v1 --purpose exploratory --seeds 11 23 37 51 71 --budget-selection work/budget-pilot-v1/selection.json --trees 200 --threads 2 --feature-condition without_url_length_ratio
python scripts/verify_benchmark.py --run work/ablation-length-v1 --output work/verified-ablation-length-v1
python scripts/compare_feature_ablation.py --baseline work/benchmark-five-v1 --ablation work/ablation-length-v1 --prepared work/prepared-candidate-v2 --output work/compared-ablation-length-v1
```

The baseline files are from the earlier four-feature run. New runner invocations default to `--feature-condition all`, preserving its behavior.

This ablation cannot establish equivalent extraction, remove all dataset shift, or justify choosing a feature condition from target-test performance. A new independent confirmation dataset is needed for claims based on this diagnosis. See [feature compatibility](feature-compatibility.md) for provenance, source references and the accepted ISCX rename assumption.
