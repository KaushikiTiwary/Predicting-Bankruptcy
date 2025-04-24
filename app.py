import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('lda_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Manual Bankruptcy Predictor", layout="centered")
st.title("🧠 Manual Bankruptcy Risk Predictor")
st.markdown("Enter company financial details below to estimate bankruptcy risk.")

# Input fields
tic = st.text_input("Enter Company Ticker (tic)", value="XYZ")
fyear = st.number_input("Enter Financial Year (fyear)", value=2024, step=1)
industry = st.text_input("Enter Industry", value="Unknown")

x1 = st.number_input("X1: Working Capital / Total Assets", format="%.6f")
x2 = st.number_input("X2: Retained Earnings / Total Assets", format="%.6f")
x3 = st.number_input("X3: EBIT / Total Assets", format="%.6f")
x4 = st.number_input("X4: Market Value of Equity / Total Liabilities", format="%.6f")
x5 = st.number_input("X5: Sales / Total Assets", format="%.6f")

# Risk classifier function
def get_risk(prob):
    if prob < 0.49:
        return "🔴 Very High Risk"
    elif prob < 0.50:
        return "🟠 High Risk"
    elif prob <= 0.51:
        return "🟡 Medium Risk"
    else:
        return "🟢 Very Low Risk"

# Prediction button
if st.button("🔍 Predict Bankruptcy Risk"):
    input_df = pd.DataFrame([[x1, x2, x3, x4, x5]], columns=['X1', 'X2', 'X3', 'X4', 'X5'])
    scaled_input = scaler.transform(input_df)
    prob = model.predict_proba(scaled_input)[0, 1]
    risk_label = get_risk(prob)

    result_df = pd.DataFrame([{
        "tic": tic, "fyear": fyear, "industry": industry,
        "lda_probability": prob, "risk_zone": risk_label
    }])

    st.success("✅ Prediction Complete")
    st.dataframe(result_df)
