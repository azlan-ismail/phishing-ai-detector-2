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
