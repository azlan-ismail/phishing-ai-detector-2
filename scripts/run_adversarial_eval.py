import pandas as pd
import numpy as np
import joblib
import json
import warnings
from sklearn.metrics import accuracy_score
from art.attacks.evasion import BoundaryAttack
from art.estimators.classification import SklearnClassifier, XGBoostClassifier
from tqdm import tqdm

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load dataset
print("📥 Loading dataset...")
phishing_df = pd.read_csv('data/phishing.csv')
X = phishing_df.drop('Result', axis=1)
y = phishing_df['Result'].replace({-1: 0, 1: 1})

# Load models
print("🧠 Loading trained models...")
model_lr = joblib.load('models/phishing_model_lr.pkl')
model_rf = joblib.load('models/phishing_model_rf.pkl')
model_xgb = joblib.load('models/phishing_model_xgb.pkl')

# Wrap models with ART classifiers
clf_lr = SklearnClassifier(model=model_lr, clip_values=(0, 1))
clf_rf = SklearnClassifier(model=model_rf, clip_values=(0, 1))
clf_xgb = XGBoostClassifier(model=model_xgb, nb_classes=2, clip_values=(0, 1))
clf_xgb._input_shape = X.shape[1:]  # Required by BoundaryAttack

# Sample 200 points for faster attack
print("🔍 Sampling 200 data points for evaluation...")
sample_df = X.sample(n=200, random_state=42)
X_sample = sample_df
y_sample = y.loc[sample_df.index]

results = {}

# Define a helper to wrap attack steps
def run_attack(name, clf, filename):
    print(f"⚔️  Running BoundaryAttack on {name}...")
    attack = BoundaryAttack(estimator=clf, targeted=False, max_iter=50)
    X_adv = attack.generate(x=X_sample.to_numpy())
    y_pred = np.argmax(clf.predict(X_adv), axis=1)
    acc = float(accuracy_score(y_sample, y_pred))
    print(f"✅ Accuracy on adversarial {name}: {acc:.4f}")
    pd.DataFrame(X_adv, columns=X.columns).to_csv(f"data/X_adv_{filename}.csv", index=False)
    return acc, y_pred

# Run attacks
acc_lr, pred_lr = run_attack("Logistic Regression", clf_lr, "lr")
acc_rf, pred_rf = run_attack("Random Forest", clf_rf, "rf")
acc_xgb, pred_xgb = run_attack("XGBoost", clf_xgb, "xgb")

# Save predictions to data folder
print("💾 Saving adversarial predictions to data/...")
pd.DataFrame({
    'true_label': y_sample.values,
    'pred_lr': pred_lr,
    'pred_rf': pred_rf,
    'pred_xgb': pred_xgb
}).to_csv("data/adversarial_predictions.csv", index=False)

# Save metrics to metrics folder
print("💾 Saving adversarial metrics to metrics/...")
with open("metrics/adversarial_metrics.json", "w") as f:
    json.dump({
        'lr_accuracy_adv': acc_lr,
        'rf_accuracy_adv': acc_rf,
        'xgb_accuracy_adv': acc_xgb
    }, f, indent=2)

print("\n🎉 Done! Adversarial evaluation using BoundaryAttack completed.")