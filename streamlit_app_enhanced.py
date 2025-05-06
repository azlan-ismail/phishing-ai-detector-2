
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title='Phishing Detection App', layout='wide')

@st.cache_data
def load_data():
    df = pd.read_csv('data/phishing.csv')
    X = df.drop('Result', axis=1)
    y = df['Result'].replace({1: 1, -1: 0})
    return X, y

@st.cache_resource
def load_models():
    model_lr = joblib.load('models/phishing_model_lr.pkl')
    model_rf = joblib.load('models/phishing_model_rf.pkl')
    model_xgb = joblib.load('models/phishing_model_xgb.pkl')
    return model_lr, model_rf, model_xgb

X, y = load_data()
model_lr, model_rf, model_xgb = load_models()
feature_names = X.columns.tolist()

tab1, tab2 = st.tabs(["🔍 Prediction & Explanation", "🎯 Threshold Tuning"])

with tab1:
    st.header("🔍 Prediction & Explanation")

    input_mode = st.radio("Select Input Mode", ("Form Input", "Upload CSV"))

    if input_mode == "Form Input":
        st.markdown("### 🔢 Enter Feature Values")
        input_data = {}
        cols = st.columns(3)
        for idx, feature in enumerate(feature_names):
            with cols[idx % 3]:
                input_data[feature] = st.number_input(feature, value=int(X[feature].mode()[0]), step=1, format="%d")
        input_df = pd.DataFrame([input_data])

    else:
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)
            st.write("✅ Uploaded Data Preview:")
            st.dataframe(input_df.head())
        else:
            input_df = None

    if input_df is not None:
        st.markdown("### 📈 Model Predictions")
        proba_lr = model_lr.predict_proba(input_df)[:, 1][0]
        proba_rf = model_rf.predict_proba(input_df)[:, 1][0]
        proba_xgb = model_xgb.predict_proba(input_df)[:, 1][0]

        st.metric("Logistic Regression", f"{proba_lr:.3f} (Phishing)" if proba_lr >= 0.5 else f"{proba_lr:.3f} (Legit)")
        st.metric("Random Forest", f"{proba_rf:.3f} (Phishing)" if proba_rf >= 0.5 else f"{proba_rf:.3f} (Legit)")
        st.metric("XGBoost", f"{proba_xgb:.3f} (Phishing)" if proba_xgb >= 0.5 else f"{proba_xgb:.3f} (Legit)")

        st.markdown("### 🔍 SHAP Explanation")
        shap_model_option = st.selectbox("Select model for SHAP explanation", 
                                         ["Logistic Regression", "Random Forest", "XGBoost"])

        shap_model = model_lr if shap_model_option == "Logistic Regression" else (
                     model_rf if shap_model_option == "Random Forest" else model_xgb)

        explainer = shap.Explainer(shap_model.predict_proba, X)
        shap_values = explainer(input_df)
        shap_single = shap.Explanation(
            values=shap_values.values[0][:, 1],
            base_values=shap_values.base_values[0][1],
            data=shap_values.data[0],
            feature_names=shap_values.feature_names
        )
        fig = shap.plots.waterfall(shap_single, show=False)
        st.pyplot(fig, use_container_width=True)

with tab2:
    st.header("🎯 Threshold Tuning for Phishing Detection")
    st.markdown("""
    Adjust the **decision threshold** below to explore how classification performance changes
    for each model. This is crucial in real-world cybersecurity systems where trade-offs between
    catching threats and avoiding false positives must be carefully balanced.
    """)

    threshold = st.slider("Select Decision Threshold", 0.0, 1.0, 0.5, 0.01)

    def evaluate(model, name):
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)
        return {
            "Model": name,
            "Accuracy": accuracy_score(y, preds),
            "Precision": precision_score(y, preds, zero_division=0),
            "Recall": recall_score(y, preds, zero_division=0),
            "F1 Score": f1_score(y, preds, zero_division=0)
        }

    results = pd.DataFrame([
        evaluate(model_lr, 'Logistic Regression'),
        evaluate(model_rf, 'Random Forest'),
        evaluate(model_xgb, 'XGBoost')
    ])

    st.subheader(f'📊 Evaluation Results at Threshold = {threshold:.2f}')
    st.dataframe(results.style.format(precision=3), use_container_width=True)

    metric_to_plot = st.selectbox("Metric to Visualize", ["Accuracy", "Precision", "Recall", "F1 Score"])
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results, x="Model", y=metric_to_plot)
    plt.title(f"{metric_to_plot} at Threshold {threshold:.2f}")
    plt.ylabel(metric_to_plot)
    plt.xlabel("Model")
    st.pyplot(plt)
