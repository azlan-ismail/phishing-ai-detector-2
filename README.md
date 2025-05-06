
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
