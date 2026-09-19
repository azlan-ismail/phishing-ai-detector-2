# Experiment implementation checklist

**Provenance update (19 September 2026):** Mendeley's published labels are now confirmed as 0 = legitimate and 1 = phishing. The ISCX file is treated as the IEEE file renamed without content changes, per the user's explicit assumption; this has not been independently byte-verified. Source papers support the four intended feature meanings, but exact extraction equivalence remains unverified. The [feature-compatibility audit](feature-compatibility.md) documents strong class-conditional distribution differences. This update supersedes earlier statements below that Mendeley's published label meaning is unresolved; historical results and manifests are unchanged.


Status: original files inspected, exploratory preparation and corrected six-model trainer implemented; real-data smoke and five-seed exploratory comparisons completed. Source definitions, balancing provenance, and the full scientific experiment suite remain unresolved. Check items only when evidence is linked.
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

These preliminary checks did not complete Stage A. The original manuscript code and named tables were subsequently inspected, as recorded below; label provenance and original prediction-level evidence remain unresolved.

## Completed exploratory preprocessing

See [implementation and run instructions](preprocessing.md) and [aggregate validation evidence](../research/audits/preprocessing-validation.json).

- [x] Inspect the supplied original DQN/DDQN, feature-construction, and evaluation scripts locally.
- [x] Trace original ISCX/Mendeley tables to the supplied processed splits.
- [x] Implement four provisional shared features with explicit invalid-value handling.
- [x] Save stable row IDs, seeded 60/20/20 partitions, and source-training-only transformation parameters.
- [x] Run both full-record grouping and reduced-feature grouping sensitivity preparations.
- [x] Verify all 16 transformed exports; all 14 tests pass.

The source tables have no raw URLs/domains. Candidate feature equivalence and Mendeley label provenance remain pending. These outputs are exploratory, not scientific model results. No original user datasets were uploaded.

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

## Completed training implementation and smoke check

See [training instructions](training.md).

- [x] Implement all six models, with matched DQN/DDQN settings and true-class reward weighting.
- [x] Save source-validation thresholds, continuous scores, predictions, training histories, and run metadata.
- [x] Execute seed 11 with 250 neural updates and 50 RF trees on the actual candidate-grouped datasets.
- [x] Verify 72 operating-point results across 24 source-target evaluations; all 21 tests pass.

A subsequent five-seed exploratory evaluation is recorded below. Full tuning, feature-semantic verification, ablations, and convergence analysis remain open.

## Stage C: core models (depends on B)

- [x] Implement always-malicious, LR, RF, MLP, DQN, and DDQN interfaces. (Exploratory four-feature run; see five-seed results.)
- [x] Match MLP hidden architecture and control DQN/DDQN training settings. (Exploratory four-feature run; see five-seed results.)
- [ ] Add source-only tuning with recorded trial budgets.
- [x] Run five training seeds and both transfer directions. (Exploratory four-feature run; see five-seed results.)
- [x] Persist continuous scores, labels, predictions, and run metadata. (Exploratory four-feature run; see five-seed results.)

Acceptance: all four scenarios reproducible from one documented entry point.

## Stage D: diagnostic experiments (depends on C)

- [ ] Run feature-mapping diagnostics and 0.005/0.01/0.02 selection sensitivity.
- [ ] Run eight-condition feature-selection/reward/SMOTE ablations per RL model/source.
- [ ] Compare MLP class weighting with SMOTE.
- [ ] Compare tuned versus untuned results.
- [ ] Run controlled imbalance stress tests if retaining broad imbalance claims.

Acceptance: per-run ablation tables with matched conditions and documented exceptions.

## Stage E: operational evaluation and analysis (depends on C; integrates D)

- [x] Select thresholds using source validation only and freeze before testing. (Exploratory four-feature run; see five-seed results.)
- [x] Report actual target FPR, PR curves, confusion counts, and trivial baselines. (Exploratory four-feature run; see five-seed results.)
- [x] Report sample SD, paired differences, and appropriately scoped uncertainty. (Exploratory four-feature run; see five-seed results.)
- [ ] Diagnose one-class collapse and source-overfitting behavior.
- [ ] Plot common-scale learning curves; remove unmeasured causal claims.
- [x] Record training/inference costs and exact environment. (Exploratory four-feature run; see five-seed results.)

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

## Completed five-seed exploratory evaluation

See [results and scope](five-seed-results.md).

- [x] Compare 1,000/2,000/4,000 neural updates with pilot seed 101 using source-validation AP only.
- [x] Apply the predefined near-best rule and freeze 1,000 updates for reporting seeds 11, 23, 37, 51, 71; use 200 RF trees.
- [x] Complete 60 model/source/seed configurations and 120 source-target evaluations.
- [x] Recompute 360 metric rows and frozen validation thresholds from saved predictions.
- [x] Record paired DDQN-minus-DQN differences; all 23 automated tests pass.

The best pilot mean occurs at the largest candidate; convergence is not established. Validation reuse, provisional feature semantics, absent URL/domain identifiers, and prior test-informed redesign prevent confirmatory claims. Checked Stage C/E items indicate implementation and exploratory execution only, not satisfaction of all protocol dependencies.
