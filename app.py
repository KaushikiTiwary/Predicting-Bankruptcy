import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('lda_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define Z-score weights for each industry
z_weights_dict = {
    'Healthcare': {
        'X1': -0.140792,
        'X2':  0.140466,
        'X3':  0.055419,
        'X4':  0.075092,
        'X5':  0.121032
    },
    'Tech': {
        'X1': -0.461234,
        'X2':  0.538634,
        'X3':  0.043953,
        'X4':  0.119705,
        'X5':  0.172948
    }
}

# Set page layout
st.set_page_config(page_title="Bankruptcy Risk Predictor", layout="centered")
st.title("📊 Industry-Specific Bankruptcy Risk Predictor")
st.markdown("Use Altman-style Z-Score + ML model to predict risk for **Healthcare** or **Tech** companies.")

# Select Industry
industry = st.selectbox("🏭 Select Industry", options=["Healthcare", "Tech"])
z_weights = z_weights_dict[industry]

# Display Risk Buckets (same for now)
with st.expander("📌 Risk Bucket Thresholds"):
    st.markdown("""
    - 🔴 **Very High Risk**: < `0.49`  
    - 🟧 **High Risk**: `0.49 – 0.50`  
    - 🟨 **Medium Risk**: `0.50 – 0.50`  
    - 🟩 **Very Low Risk**: > `0.50`
    """)

# Company Info
tic = st.text_input("🏷️ Company Ticker (tic)", value="ABC123")
fyear = st.number_input("📅 Financial Year", value=2024, step=1)

# Financial Inputs
st.subheader("💰 Enter Financial Inputs")
assets_current_total = st.number_input("Assets Current Total", value=0.0)
liabilities_current_total = st.number_input("Liabilities Current Total", value=0.0)
total_assets = st.number_input("Total Assets", value=1.0)
retained_earnings = st.number_input("Retained Earnings", value=0.0)
ebit = st.number_input("EBIT (Earnings Before Interest & Tax)", value=0.0)
total_sales = st.number_input("Total Sales", value=0.0)
total_liabilities = st.number_input("Total Liabilities", value=1.0)
stock_price = st.number_input("Stock Price", value=0.0)
shares_outstanding = st.number_input("Shares Outstanding", value=0.0)

# Risk Classification Function
def get_risk(prob):
    if prob < 0.49:
        return "🔴 Very High Risk"
    elif prob < 0.50:
        return "🟧 High Risk"
    elif prob <= 0.50:
        return "🟨 Medium Risk"
    else:
        return "🟩 Very Low Risk"

# Predict button
if st.button("🔍 Predict Bankruptcy Risk"):
    x1 = (assets_current_total - liabilities_current_total) / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = (stock_price * shares_outstanding) / total_liabilities
    x5 = total_sales / total_assets

    z_score = (
        z_weights['X1'] * x1 +
        z_weights['X2'] * x2 +
        z_weights['X3'] * x3 +
        z_weights['X4'] * x4 +
        z_weights['X5'] * x5
    )

    input_df = pd.DataFrame([[x1, x2, x3, x4, x5]], columns=['X1', 'X2', 'X3', 'X4', 'X5'])
    scaled_input = scaler.transform(input_df)
    ml_prob = model.predict_proba(scaled_input)[0, 1]
    ml_risk = get_risk(ml_prob)

    st.success("✅ Prediction Complete")
    st.write("### 📉 Z-Score Components")
    st.json({
        "X1 (Working Capital / Total Assets)": round(x1, 4),
        "X2 (Retained Earnings / Total Assets)": round(x2, 4),
        "X3 (EBIT / Total Assets)": round(x3, 4),
        "X4 (Market Value of Equity / Total Liabilities)": round(x4, 4),
        "X5 (Sales / Total Assets)": round(x5, 4),
    })

    st.metric("📊 Z-Score", f"{z_score:.4f}")
    st.metric("🤖 ML Probability", f"{ml_prob:.4f}")
    st.markdown(f"**📌 Risk Zone**: {ml_risk}")

    result_df = pd.DataFrame([{
        "tic": tic, "fyear": fyear, "industry": industry,
        "X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5,
        "Z_Score": z_score,
        "ML_Probability": ml_prob,
        "Risk_Level": ml_risk
    }])
    st.dataframe(result_df)

