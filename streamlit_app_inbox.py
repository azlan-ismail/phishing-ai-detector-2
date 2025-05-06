# This script uses Streamlit and explainability libraries to visualize phishing email detection.
# NOTE: This script requires the 'streamlit', 'shap', and 'lime' packages to be installed in your environment.
# To run: `streamlit run streamlit_app_inbox_rewritten.py`

try:
    import streamlit as st
except ModuleNotFoundError:
    raise ImportError("The 'streamlit' package is required. Install it using 'pip install streamlit'")

import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import re

# Load models
model_lr = joblib.load("models/phishing_model_lr.pkl")
model_rf = joblib.load("models/phishing_model_rf.pkl")
model_xgb = joblib.load("models/phishing_model_xgb.pkl")

# Feature names
real_feature_names = [f"feature_{i}" for i in range(1, 31)]
xgb_feature_names = [
    'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
    'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
    'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
    'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
    'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
    'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
    'Statistical_report'
]

feature_name_mapping = {f"feature_{i+1}": xgb_feature_names[i] for i in range(30)}
readable_feature_names = [feature_name_mapping.get(f, f) for f in real_feature_names]

# Simulated inbox
data = [
    {"Email ID": 1, "From": "support@bank.com", "Subject": "Account alert!", "Preview": "Your account is locked due to suspicious activity. Please verify immediately."},
    {"Email ID": 2, "From": "newsletter@shop.com", "Subject": "Latest deals", "Preview": "Check out our 50% discounts on electronics this weekend only."},
    {"Email ID": 3, "From": "it@company.com", "Subject": "Password expiry", "Preview": "Update your password within 24 hours to avoid service interruption."},
    {"Email ID": 4, "From": "admin@securepay.com", "Subject": "Transaction blocked", "Preview": "Unusual login detected. Click here to verify your identity."},
    {"Email ID": 5, "From": "info@health.net", "Subject": "COVID updates", "Preview": "New health guidelines issued by MOH. Stay informed and safe."},
    {"Email ID": 6, "From": "careers@jobs.com", "Subject": "Job interview invitation", "Preview": "We reviewed your resume and would like to schedule an interview."},
]
inbox_df = pd.DataFrame(data)

# Feature extraction
def extract_features(preview, sender="", subject=""):
    f = []
    f.append(1 if re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', preview) else 0)
    f.append(1 if len(preview) > 50 and 'http' in preview else 0)
    f.append(1 if "bit.ly" in preview or "tinyurl.com" in preview else 0)
    f.append(1 if '@' in preview or re.search(r'(gmail|yahoo|outlook)', sender.lower()) else 0)
    f.append(1 if re.search(r'https?://[^\s]*-[^\s]*\.', preview) or '-' in sender else 0)
    f.append(1 if preview.count('.') > 2 else 0)
    f.append(1 if "https://" in preview else 0)
    f.append(1 if preview.count("https") > 1 else 0)
    f.append(1 if re.search(r'https?://', preview) else 0)
    f.append(1 if "click here" in preview.lower() else 0)
    f.append(1 if re.search(r"(verify|urgent|immediately|suspend|expire)", preview.lower()) else 0)
    f.append(1 if re.search(r"(iframe|javascript|onmouseover)", preview.lower()) else 0)
    f.append(1 if ("bank" in sender.lower() and re.search(r"(gmail|yahoo|outlook)", sender.lower())) else 0)
    f.append(1 if re.search(r"(blocked|account|verify|suspend)", subject.lower()) else 0)
    while len(f) < 30: f.append(0)
    return f

mock_features = {
    row["Email ID"]: extract_features(row["Preview"], row["From"], row["Subject"])
    for _, row in inbox_df.iterrows()
}

# Streamlit UI
st.title("📥 Simulated Email Inbox - Phishing Detection")
model_option = st.selectbox("🔧 Choose model", ["Logistic Regression", "Random Forest", "XGBoost"])
model = model_lr if model_option == "Logistic Regression" else (model_rf if model_option == "Random Forest" else model_xgb)

selected_subject = st.selectbox("📧 Choose an email to inspect", inbox_df["Subject"])
selected_email = inbox_df[inbox_df["Subject"] == selected_subject].iloc[0]
st.markdown(f"**From:** {selected_email['From']}")
st.markdown(f"**Subject:** {selected_email['Subject']}")
st.markdown(f"**Content:** {selected_email['Preview']}")

threshold = st.slider("⚙️ Set prediction threshold", 0.0, 1.0, 0.5, step=0.01)
explanation_method = st.radio("🧠 Choose explanation method", ["SHAP", "LIME"])

columns_used = xgb_feature_names if model_option == "XGBoost" else real_feature_names
plot_names = xgb_feature_names if model_option == "XGBoost" else readable_feature_names
X_all = pd.DataFrame(list(mock_features.values()), columns=columns_used)
email_vector = pd.DataFrame([mock_features[selected_email["Email ID"]]], columns=columns_used)

# Prediction
proba = model.predict_proba(email_vector)[0, 1]
prediction = "Phishing" if proba >= threshold else "Legitimate"
st.metric("Prediction", f"{prediction} ({proba:.3f})", delta=f"Threshold: {threshold:.2f}")

if explanation_method == "SHAP":
    st.subheader("🔍 SHAP Explanation")
    explainer = shap.Explainer(model.predict_proba, X_all)
    shap_values = explainer(email_vector)
    shap_single = shap.Explanation(
        values=shap_values.values[0][:, 1],
        base_values=shap_values.base_values[0][1],
        data=shap_values.data[0],
        feature_names=plot_names
    )
    fig = shap.plots.waterfall(shap_single, show=False)
    st.pyplot(fig, use_container_width=True)

    shap_series = pd.Series(shap_single.values, index=shap_single.feature_names)
    # shap_series_filtered removed; filtering now applied directly to pos and neg
    pos = shap_series[shap_series > 0.01].sort_values(ascending=False).head(3)
    neg = shap_series[shap_series < -0.01].sort_values().head(2)

    def fmt(f): return ', '.join([f"**{k} ({v:+.3f})**" for k, v in f.items()])
    
    
    st.subheader("🗣️ SHAP Explanation Summary")

    if prediction == "Phishing":
        if not pos.empty:
            st.markdown(f"{fmt(pos)} increased phishing risk.")
        else:
            st.markdown("This email was classified as phishing, but no dominant features were found increasing the risk.")
        if not neg.empty:
            st.markdown(f"{fmt(neg)} helped reduce the risk.")
    else:
        if not neg.empty:
            st.markdown(f"{fmt(neg)} reduced phishing risk.")
        else:
            st.markdown("This email was classified as legitimate, but no dominant features were found reducing the risk.")
        if not pos.empty:
            st.markdown(f"{fmt(pos)} slightly increased the risk.")


else:
    st.subheader("🔍 LIME Explanation")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_all.values,
        feature_names=plot_names,
        class_names=["Legitimate", "Phishing"],
        mode="classification"
    )
    lime_exp = lime_explainer.explain_instance(
        data_row=email_vector.values[0],
        predict_fn=model.predict_proba,
        num_features=5
    )
    fig = lime_exp.as_pyplot_figure()
    st.pyplot(fig, use_container_width=True)

    st.subheader("🗣️ LIME Explanation Summary")
    lime_pairs = dict(lime_exp.as_list(label=1))  # uses readable names already from plot_names
    top = sorted(lime_pairs.items(), key=lambda x: -abs(x[1]))[:3]
    for feature, impact in top:
        st.markdown(f"- **{feature}** contributed {impact:+.3f} to phishing risk")
