# Research experiment redesign

A planned DQN/DDQN and supervised-baseline study is documented in the [experimental protocol](docs/experimental-protocol.md) and [implementation checklist](docs/implementation-checklist.md). The protocol distinguishes planned scientific work from completed exploratory checks. The existing repository content below remains an adversarial-evaluation tutorial; the original manuscript implementation has been inspected locally, while dataset provenance and scientific validation remain open.

The [initial data audit](docs/data-audit.md) now documents the existing tutorial CSV and evaluation limitations, with reusable audit scripts and tests. The supplied manuscript data and code have since been inspected locally; the corrected six-model trainer and five-seed exploratory comparison are complete. Source-definition verification, full tuning, ablations and independent confirmation remain open.

The [five-seed results](docs/five-seed-results.md) report an actual Python run on the supplied data, including source-only budget selection, supervised baselines, bidirectional transfer and recomputed metrics. These results use four provisional shared features and do not establish DDQN superiority.

---

# 🧪 Hands-on 2: Adversarial Attack Evaluation & Explainability

## 🎯 Objective
This tutorial focuses on evaluating the robustness of phishing detection models against adversarial attacks and interpreting their decisions using SHAP and LIME.

---

## 📁 Project Folder Structure

```
phishing-ai-detector/
├── data/
│   ├── phishing.csv
│   ├── X_adv_lr.csv
│   ├── X_adv_rf.csv
│   └── X_adv_xgb.csv
├── models/
│   ├── phishing_model_lr.pkl
│   ├── phishing_model_rf.pkl
│   └── phishing_model_xgb.pkl
├── metrics/
│   └── adversarial_metrics.json
├── notebooks/
│   └── Adversarial_Attack_Evaluation.ipynb
├── scripts/
│   └── run_adversarial_eval.py
```

---

## 🧠 Step-by-Step Instructions

### Step 1: Load Original Dataset and Trained Models
- Dataset: `data/phishing.csv`
- Models: Logistic Regression, Random Forest, XGBoost (`models/phishing_model_*.pkl`)

### Step 2: Generate Adversarial Examples
Use the `run_adversarial_eval.py` script to create and evaluate adversarial samples:

```bash
python scripts/run_adversarial_eval.py
```

- This script loads the original models and generates adversarial examples using `BoundaryAttack`.
- Output CSVs will be saved as:
  - `data/X_adv_lr.csv`
  - `data/X_adv_rf.csv`
  - `data/X_adv_xgb.csv`
- Evaluation metrics saved to `metrics/adversarial_metrics.json`

### Step 3: Open the Notebook
Navigate to `notebooks/Adversarial_Attack_Evaluation.ipynb`. Run all cells.

It includes:
- Accuracy evaluation on adversarial inputs
- SHAP explanations (for all three models)
- LIME explanations (for all three models)

### Step 4: SHAP Explanations
- Use `shap.Explainer(model.predict_proba, X)`
- Show individual predictions using `shap.plots.waterfall(...)`
- Understand **which features contributed most** to misclassifications

### Step 5: LIME Explanations
- Use `LimeTabularExplainer(...)` for black-box interpretability
- Explain the same adversarial samples
- View local decision boundaries via `show_in_notebook(...)`

---

## 📈 Expected Outcomes

| Model              | Accuracy on Clean Data | Accuracy on Adversarial Data |
|--------------------|------------------------|-------------------------------|
| Logistic Regression | ~95%                   | ↓ after attack                |
| Random Forest       | ~98%                   | ↓ after attack                |
| XGBoost             | ~99%                   | ↓ after attack                |

You should observe **performance drops**, and use SHAP/LIME to diagnose **why the models failed**.

---

## 📝 Deliverables

1. Completed notebook: `Adversarial_Attack_Evaluation.ipynb`
2. CSVs for adversarial test sets
3. Summary of attack results and interpretations (optional slide/report)

## Exploratory research preprocessing

The supplied manuscript code and original feature tables have now been inspected locally. A [reproducible preparation pipeline](docs/preprocessing.md) creates source-fitted transforms, saved partitions, and both transfer directions. Both grouping modes were exercised on the real tables and all 14 tests passed; see the [validation report](research/audits/preprocessing-validation.json). The four common features remain provisional pending source definitions. No new model results are claimed, and raw datasets remain local.

## Controlled training implementation

The [shared trainer and six-model benchmark](docs/training.md) have now run on the actual prepared ISCX/Mendeley data. The initial single-seed smoke run completed all models and both transfer directions; all 21 tests pass, and 72 operating-point results were verified from saved predictions. The [aggregate evidence](research/audits/training-smoke-validation.json) and [smoke metrics](research/audits/training-smoke-metrics.csv) are reproducible workflow checks, not final manuscript results. Use the separate research requirements and a new local output directory.

## Completed bounded tuning stage (20 September 2026)

Source-only six-candidate tuning and the selected five-seed runs are complete. See the [tuned results and reproducible commands](docs/tuned-source-results.md) for the current stage; earlier sections describe the historical baseline and ablations. All 60 candidate objectives, 360 final metric rows and 180 sample-paired prediction files passed verification; the 26-test suite passed. All neural selections use 8,000 updates and learning rate .001. Convergence, independent confirmation and a matched longer-budget gamma comparison remain open. Public aggregates include per-seed metrics, SD summaries, learning curves and paired changes; private predictions and checkpoints are indexed in the updated artifact inventory.

## Completed matched longer-budget discount ablation (20 September 2026)

The previously listed matched longer-budget gamma comparison is complete. [Results and reproduction commands](docs/ablation-gamma-tuned-results.md) compare gamma 0 against gamma .99 with the same five seeds and inherited 8,000-update tuned settings. All 360 metric rows, 180 paired prediction files and actual neural exposures were verified; all 29 tests passed. Non-RL control decisions and metrics are unchanged, and gamma-zero DQN/DDQN predictions and histories match exactly. Within-dataset AP improves slightly while transfer effects are mixed. Convergence and independent confirmation remain open. Per-seed aggregates, SD summaries and paired changes are recorded under research/results/gamma-zero-tuned; the inventory covers 3,186 local artifacts.

## Cross-dataset validation readiness

The [cross-dataset protocol audit](docs/cross-dataset-validation.md) reproduces both saved transfer exports using source-training-only preprocessing. It separates verified numeric preparation from unresolved extractor equivalence, records target range and feature-collision diagnostics, and specifies the raw-URL/extractor evidence needed for a stronger follow-up. Existing transfer results remain exploratory; no additional model training was performed for this audit.

## Completed source-only plateau diagnostic

The [source-validation extension](docs/source-convergence-results.md) checked six original-pilot-seed trajectories beyond 8,000 updates, under a fixed .001 AP improvement / two-check patience rule capped at 32,000. All 26 checkpoints and stopping/selection decisions were verified; all 32 tests passed. Mendeley RL validation scores benefited from longer training, while ISCX gains were small and Mendeley MLP did not improve. No test data were evaluated. This establishes a bounded single-seed validation plateau diagnostic, not general convergence. Existing five-seed test results are retained. Policy, curves, results and verification are published under research/results/source-convergence.

## Completed matched longer-budget five-seed evaluation

The [latest results](docs/matched-long-results.md) use frozen budgets of 16,000 updates for ISCX and 28,000 for Mendeley, shared by MLP/DQN/DDQN. All 60 configurations and 360 metric rows completed; all 34 tests passed. Verification checked saved metrics, 180 paired prediction files, actual neural budgets and unchanged non-neural controls. Mendeley RL pilot-validation gains did not translate into mean test-AP gains; MLP improved, Random Forest retains highest within-source AP, and transfer remains weak. Policy, per-seed metrics, SD summaries and paired comparisons are recorded under research/results/matched-long. This is exploratory evidence on fixed partitions, not a general convergence or DDQN-superiority claim.

## Revised manuscript draft

The [replacement manuscript](docs/manuscript-revision.md) reframes the study as a controlled four-feature comparison of Q-learning and supervised baselines. It includes revised research questions, methods, verified mean/SD tables, operational false-positive analysis and explicit limits. The [reviewer coverage map](docs/manuscript-revision-review.md) distinguishes addressed concerns from remaining novelty, figure, imbalance and independent-validation gaps. This is an editorial draft rather than a submission-ready typeset article. No models were retrained for this revision.
