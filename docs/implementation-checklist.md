# Experiment implementation checklist

Status: initial tutorial-data audit completed; manuscript inputs remain unresolved. Check items only when evidence is linked.
Protocol: [experimental protocol](experimental-protocol.md).

## Repository audit

Inspected main at 8dbb4ab37b94b18e7dc1a273df74398e54f0cc5f.

- README describes an adversarial-evaluation tutorial.
- train_and_evaluate_all_models.py fits LR, RF, and XGBoost on data/phishing.csv with a single 70/30 split and no explicit stratification or separate validation partition.
- It maps Result {-1:0, 1:1}; class semantics require provenance verification.
- scripts/run_adversarial_eval.py samples from the full dataset, so its evaluation can include model-training examples. Do not reuse those results as held-out evidence.
- No standalone DQN/DDQN implementation or explicitly named ISCX/Mendeley input appears in the inspected tree. Notebook contents have not been exhaustively audited.
- Existing tutorial metrics are not results of the proposed protocol.

## Completed preliminary audit

Evidence and reproduction commands: [initial data audit](data-audit.md).

- [x] Audit the repository CSV's structure, raw labels, missingness, and exact repetition.
- [x] Reconstruct the existing tutorial split and quantify feature overlap.
- [x] Check saved supervised class-1 metrics and accuracy against confusion matrices.
- [x] Add reproducible audit scripts and nine passing tests.
- [x] Identify full-dataset threshold selection and adversarial evaluation limitations.

These checks do not complete Stage A: the original manuscript code, named datasets, label provenance, and prediction-level evidence are still required.

## Stage A: recover and audit inputs (blocks scientific runs)

- [ ] Locate original DQN/DDQN implementation, reward definitions, and training logs.
- [ ] Identify ISCX/Mendeley versions, checksums, permitted storage locations, and labels.
- [ ] Resolve whether phishing.csv belongs to either dataset; do not assume.
- [ ] Recompute manuscript metrics from saved predictions.
- [ ] Record code/data discrepancies and missing evidence.

Acceptance: verified dataset manifest and reproducible metric audit.

## Stage B: common features and partitions (depends on A)

- [ ] Document exact feature definitions and remove unsupported proxies from primary schema.
- [ ] Implement original, minus-extension, and verified feature conditions.
- [ ] Audit duplicates and domain overlap; persist shared 60/20/20 partitions.
- [ ] Implement fold-local preprocessing and resampling.
- [ ] Verify no training/validation/test overlap or target-fitted preprocessing.

Acceptance: feature dictionary, partition IDs/hashes, and leakage checks.

## Stage C: core models (depends on B)

- [ ] Implement always-malicious, LR, RF, MLP, DQN, and DDQN interfaces.
- [ ] Match MLP hidden architecture and control DQN/DDQN training settings.
- [ ] Add source-only tuning with recorded trial budgets.
- [ ] Run five training seeds and both transfer directions.
- [ ] Persist continuous scores, labels, predictions, and run metadata.

Acceptance: all four scenarios reproducible from one documented entry point.

## Stage D: diagnostic experiments (depends on C)

- [ ] Run feature-mapping diagnostics and 0.005/0.01/0.02 selection sensitivity.
- [ ] Run eight-condition feature-selection/reward/SMOTE ablations per RL model/source.
- [ ] Compare MLP class weighting with SMOTE.
- [ ] Compare tuned versus untuned results.
- [ ] Run controlled imbalance stress tests if retaining broad imbalance claims.

Acceptance: per-run ablation tables with matched conditions and documented exceptions.

## Stage E: operational evaluation and analysis (depends on C; integrates D)

- [ ] Select thresholds using source validation only and freeze before testing.
- [ ] Report actual target FPR, PR curves, confusion counts, and trivial baselines.
- [ ] Report sample SD, paired differences, and appropriately scoped uncertainty.
- [ ] Diagnose one-class collapse and source-overfitting behavior.
- [ ] Plot common-scale learning curves; remove unmeasured causal claims.
- [ ] Record training/inference costs and exact environment.

Acceptance: figures and tables generated from saved predictions with no manual metric editing.

## Stage F: confirmation and manuscript (depends on E)

- [ ] Add independent dataset confirmation where feasible.
- [ ] Keep target adaptation separate from frozen transfer.
- [ ] Disclose prior test-set use and scope of five-seed uncertainty.
- [ ] Rewrite claims to match evidence and publish reproduction instructions.

## Proposed output contract

Each run records: run_id, git_commit, dataset version/hash, partition hash, feature-schema hash, source/target, seed, model, configuration, preprocessing, reward equation/weights, tuning objective/budget, decision threshold and selection source, software/hardware, fit/inference times, and artifact paths.

Prediction records contain stable sample_id, domain_group when available, true_label, continuous_score, predicted_label, and operating_point. Do not include raw sensitive URLs in public artifacts.

Store new research outputs separately from existing tutorial models/metrics. Large/private datasets and checkpoints can remain in approved external storage with retrieval instructions and checksums.
