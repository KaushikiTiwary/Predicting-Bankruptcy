import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('lda_model.pkl')
scaler = joblib.load('scaler.pkl')

# Altman-style LDA weights (Healthcare)
z_weights = {
    'X1': -0.140792,
    'X2':  0.140466,
    'X3':  0.055419,
    'X4':  0.075092,
    'X5':  0.121032
}

st.set_page_config(page_title="Healthcare Bankruptcy Predictor", layout="centered")
st.title("🏥 Healthcare Bankruptcy Risk Predictor")
st.markdown("Enter **raw financial inputs** below to compute both a Machine Learning-based bankruptcy risk and a custom Z-Score.")

# Basic Company Info
tic = st.text_input("📌 Company Ticker (tic)", value="HEALTH123")
fyear = st.number_input("📅 Financial Year", value=2024, step=1)
industry = st.text_input("🏭 Industry", value="Healthcare")

st.subheader("📊 Enter Raw Financial Details")

# Financial Inputs (Based on the image you shared)
assets_current_total = st.number_input("Assets Current Total", value=0.0)
liabilities_current_total = st.number_input("Liabilities Current Total", value=0.0)
total_assets = st.number_input("Total Assets", value=1.0)  # to avoid div by zero
shareholders_equity = st.number_input("Shareholders' Equity", value=0.0)
ebit = st.number_input("Earnings Before Interest & Tax (EBIT)", value=0.0)
total_sales = st.number_input("Total Sales", value=0.0)
long_term_debt = st.number_input("Long Term Debt", value=1.0)
stock_price = st.number_input("Stock Price", value=0.0)
shares_outstanding = st.number_input("Shares Outstanding", value=0.0)

# Risk Classification Function
def get_risk(prob):
    if prob < 0.49:
        return "🔴 Very High Risk"
    elif prob < 0.50:
        return "🟠 High Risk"
    elif prob <= 0.51:
        return "🟡 Medium Risk"
    else:
        return "🟢 Very Low Risk"

# Calculate on button click
if st.button("🔍 Predict Bankruptcy Risk"):
    # 1. Compute Altman-style Ratios (X1–X5)
    x1 = (assets_current_total - liabilities_current_total) / total_assets
    x2 = shareholders_equity / total_assets
    x3 = ebit / total_assets
    x4 = (stock_price * shares_outstanding) / long_term_debt
    x5 = total_sales / total_assets

    # 2. Z-Score (LDA-style) calculation
    z_score = (
        z_weights['X1'] * x1 +
        z_weights['X2'] * x2 +
        z_weights['X3'] * x3 +
        z_weights['X4'] * x4 +
        z_weights['X5'] * x5
    )

    # 3. ML Probability using model
    input_df = pd.DataFrame([[x1, x2, x3, x4, x5]], columns=['X1', 'X2', 'X3', 'X4', 'X5'])
    scaled_input = scaler.transform(input_df)
    ml_prob = model.predict_proba(scaled_input)[0, 1]
    ml_risk = get_risk(ml_prob)

    # 4. Display
    st.success("✅ Prediction & Z-Score Computed")

    st.write("### 🔢 Computed Altman Ratios")
    st.json({
        "X1 (Working Capital / Total Assets)": round(x1, 4),
        "X2 (Retained Earnings / Total Assets)": round(x2, 4),
        "X3 (EBIT / Total Assets)": round(x3, 4),
        "X4 (MVE / Total Liabilities)": round(x4, 4),
        "X5 (Sales / Total Assets)": round(x5, 4),
    })

    st.write("### 📊 Altman-Style Z-Score")
    st.metric(label="Z-Score", value=f"{z_score:.4f}")

    st.write("### 🤖 ML-Based Bankruptcy Risk")
    st.metric(label="Bankruptcy Probability", value=f"{ml_prob:.4f}")
    st.markdown(f"**Risk Level**: {ml_risk}")

    # Final Table
    result_df = pd.DataFrame([{
        "tic": tic, "fyear": fyear, "industry": industry,
        "X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5,
        "Z_Score": z_score,
        "ML_Probability": ml_prob,
        "Risk_Level": ml_risk
    }])
    st.write("### 🧾 Full Computation Summary")
    st.dataframe(result_df)
