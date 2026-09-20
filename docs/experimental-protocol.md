# Malicious URL detection: experimental protocol

Status: planned; no new experiments or results are claimed by this document.
Repository audit reference: main at 8dbb4ab37b94b18e7dc1a273df74398e54f0cc5f.
See [implementation checklist](implementation-checklist.md) for dependencies and acceptance criteria.

## Research questions

1. Do DQN and DDQN add value over supervised classifiers within and across datasets?
2. How much of transfer failure is attributable to incompatible feature definitions?
3. What effects do feature selection, reward weighting, SMOTE, and tuning have?
4. Can useful recall be achieved with a controlled false-positive burden?

## 1. Audit and provenance

Locate the original DQN/DDQN code, ISCX and Mendeley inputs, predictions, and five-run logs. Confirm label semantics from dataset provenance; standardize malicious=1 and benign=0 only after verification. Record versions, checksums, counts, exclusions, missingness, and class frequencies. Recompute existing metrics from predictions. Reconcile manuscript settings with actual code.

The existing data/phishing.csv must not be assumed to be either manuscript dataset. The existing Result mapping {-1:0, 1:1} does not by itself establish which class is malicious.

## 2. Feature compatibility

Build a dictionary with feature names, definitions, units, extraction rules, missing-value handling, and per-source availability. Prefer one extractor applied to both raw URL collections. Otherwise retain only demonstrably equivalent measurements.

Do not equate average token length with maximum token length. Check duplicated formulas and whether letter counts include digits. Define three retrained diagnostic conditions:
- Original 14-feature mapping, faithfully reconstructed.
- Original mapping minus the extension placeholder.
- Verified common features: primary condition.

If the original mapping cannot be reproduced, disclose that limitation instead of inventing a reconstruction.

## 3. Partitions and leakage control

Create approximately 60/20/20 train/validation/test partitions per dataset. Stratify where feasible and group by registered domain where identifiers permit. Record when grouping cannot be performed. Deduplicate and audit overlap within and across sources using a documented rule. Persist sample IDs and partition hashes; every model uses the same partitions.

Fit imputation, scaling, and feature selection inside training folds. Apply SMOTE only to their training examples. Transform validation/test data with source-fitted preprocessing. Never independently normalize target data in the primary transfer experiment. Establish the order of preprocessing and resampling once and record it.

Existing test sets have informed study redesign. Disclose this; reshuffling them does not establish external confirmation. Reserve a genuinely independent dataset for confirmation when possible.

## 4. Core models and scenarios

Primary models: always-malicious predictor, logistic regression, Random Forest, supervised MLP, DQN, DDQN. The existing XGBoost baseline may be retained as a secondary comparison.

Initially use verified common features, no feature selection, no SMOTE, and unweighted objectives. Match MLP hidden architecture to the Q-networks. For the controlled DQN/DDQN comparison, match rewards, architecture, optimizer, replay, exploration, update budget, and other settings; change only the Q-target rule.

Evaluate ISCX->ISCX, Mendeley->Mendeley, ISCX->Mendeley, and Mendeley->ISCX. Each source-trained model is evaluated on both test partitions with no target updates. Use training seeds [11, 23, 37, 51, 71]. These quantify training variability on fixed partitions, not split variability. The deterministic always-malicious baseline needs no repeated training.

Do not infer temporal adaptation from static held-out evaluation. Specify transitions, terminal states, and discounting explicitly. An optional gamma=0 control can investigate the value of bootstrapping in an action-independent classification environment.

## 5. Model selection

Keep the controlled algorithm comparison distinct from individually tuned best-effort comparisons. Proposed search budget: 20 candidates per trainable model/source using three-fold source-training CV, grouped where feasible. Use mean average precision as the primary selection score. Fit all learned preprocessing within folds. Specify ranges, search seeds, stopping rules, maximum updates, and tie handling before execution.

Use inner training-only validation for early stopping if needed; keep the outer 20% validation partition available for threshold selection. Record trials, optimizer updates, sample exposures, hardware, and elapsed time. Equal trial counts do not imply equal compute. Never select settings using target performance.

## 6. Component ablations

For DQN and DDQN, run the 2x2x2 factorial of feature selection on/off, reward weighting on/off, and SMOTE on/off: eight conditions per algorithm/source. Freeze all remaining settings using source-only decisions and use matched settings across the two algorithms.

Specify exact reward equations. Derive class weights from original source-training frequencies, normalize consistently, and document behavior when combined with oversampling. This combined condition may overcorrect imbalance. Match update budgets and record changed sample exposures after SMOTE.

Add no balancing versus class-weighted loss versus SMOTE for the MLP. Compare tuned and untuned models separately. Report interactions rather than attributing bundled improvements to one component.

## 7. Sensitivity and stress tests

Feature selection: compare no selection and Gini-importance thresholds 0.005, 0.01, 0.02. Report selected names/counts. Mark zero-feature configurations invalid rather than silently substituting a different method. Select the primary threshold using source data.

The original class proportions are close to balanced. For broader imbalance claims, add approximately 50%, 20%, and 5% malicious source-training prevalence. Keep total training size matched where feasible, document any sample-size tradeoff, repeat subsampling, and preserve test prevalence. Label these as synthetic stress tests. They do not establish deployment prevalence.

## 8. Scores, thresholds, and reporting

Save continuous scores and predictions. For Q-networks use Q(malicious)-Q(benign); these scores are not calibrated probabilities. Report average precision, ROC-AUC, malicious-class precision/recall/F1, balanced accuracy, specificity, FPR, confusion counts, predicted-malicious fraction, and compute cost.

Compare default thresholds, source-validation F1-optimal thresholds, and source-validation recall maximization under FPR<=0.01. Predefine tie rules. If the benign validation count cannot support reliable estimation of 1% FPR, choose a documented less stringent constraint before examining test outcomes. Include a predict-none candidate and report zero-recall solutions honestly.

Freeze thresholds before testing. Report achieved target FPR even when it violates the source constraint. Target test curves are descriptive, not a source of deployment threshold selection. Include prevalence baselines on PR plots and calculate always-malicious results on each exact test subset.

## 9. Uncertainty and diagnosis

Report individual runs and mean +/- sample SD, with paired seed-level differences. Separate training variability from uncertainty due to finite test samples. For selected fixed-model comparisons, bootstrap paired test predictions at domain level where possible. Do not pool seeds, folds, and URLs as independent repetitions.

Predefine primary contrasts: DDQN-DQN (controlled), DQN-MLP, and DDQN-MLP (tuned). Prioritize effect sizes and confidence intervals; select justified tests and multiple-comparison handling before inferential claims.

Inspect class-conditional scores, feature/missingness shifts, prediction collapse, and source gains accompanied by transfer losses. Reward curves alone do not demonstrate Q-value overestimation. Either measure it with a justified reference-return protocol or remove the causal claim. Compare learning curves on common evaluation rewards and axes; report variability across seeds.

## 10. Extensions and deliverables

After the core study: independent third dataset, genuine temporal split, or separately labeled target adaptation. Any target-label calibration requires its own development partition and untouched target test partition.

Deliver dataset/partition summary, feature dictionary, search spaces, run manifests, prediction-level outputs, per-run/aggregate metrics, PR curves, operating-point confusion matrices, component results, and environment lock information. Preserve evidence of failures; do not invent results or treat expected tutorial accuracy as empirical evidence.

References:
- https://scikit-learn.org/stable/common_pitfalls.html
- https://scikit-learn.org/stable/modules/classification_threshold.html
