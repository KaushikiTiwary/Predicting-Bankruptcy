import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('lda_model.pkl')
scaler = joblib.load('scaler.pkl')

# Altman-style LDA weights for healthcare companies
z_weights = {
    'X1': -0.140792,
    'X2':  0.140466,
    'X3':  0.055419,
    'X4':  0.075092,
    'X5':  0.121032
}

st.set_page_config(page_title="Healthcare Bankruptcy Predictor", layout="centered")
st.title("🏥 Healthcare Bankruptcy Risk Predictor (Z-Score + ML)")
st.markdown("Enter raw financial details to compute bankruptcy risk using both ML and Altman-style Z-Score.")

# Company meta
tic = st.text_input("Company Ticker (tic)", value="HEALTH123")
fyear = st.number_input("Financial Year", value=2024, step=1)
industry = st.text_input("Industry", value="Healthcare")

# Raw inputs
st.subheader("📊 Financial Values")

working_capital = st.number_input("Working Capital", value=0.0)
total_assets = st.number_input("Total Assets", value=1.0)  # avoid division by zero
retained_earnings = st.number_input("Retained Earnings", value=0.0)
ebit = st.number_input("EBIT", value=0.0)
market_value_equity = st.number_input("Market Value of Equity", value=0.0)
total_liabilities = st.number_input("Total Liabilities", value=1.0)
sales = st.number_input("Sales", value=0.0)

# Risk labeling
def get_risk(prob):
    if prob < 0.49:
        return "🔴 Very High Risk"
    elif prob < 0.50:
        return "🟠 High Risk"
    elif prob <= 0.51:
        return "🟡 Medium Risk"
    else:
        return "🟢 Very Low Risk"

# Predict and score
if st.button("🔍 Predict Risk & Compute Z-Score"):
    # Calculate ratios
    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_value_equity / total_liabilities
    x5 = sales / total_assets

    # LDA-based Z-score calculation
    z_score = (
        z_weights['X1'] * x1 +
        z_weights['X2'] * x2 +
        z_weights['X3'] * x3 +
        z_weights['X4'] * x4 +
        z_weights['X5'] * x5
    )

    # ML Model prediction
    input_df = pd.DataFrame([[x1, x2, x3, x4, x5]], columns=['X1', 'X2', 'X3', 'X4', 'X5'])
    scaled_input = scaler.transform(input_df)
    ml_prob = model.predict_proba(scaled_input)[0, 1]
    ml_risk = get_risk(ml_prob)

    # Display results
    st.success("✅ Prediction & Z-Score Complete")
    st.write("### 🔢 Computed Ratios")
    st.json({
        "X1": round(x1, 4),
        "X2": round(x2, 4),
        "X3": round(x3, 4),
        "X4": round(x4, 4),
        "X5": round(x5, 4),
    })

    st.write("### 🧠 LDA Z-Score")
    st.metric(label="Z-Score (Custom Weights)", value=f"{z_score:.4f}")

    st.write("### 📈 ML Bankruptcy Probability")
    st.metric(label="Predicted Probability", value=f"{ml_prob:.4f}")
    st.markdown(f"*Risk Zone*: {ml_risk}")

    # Combined output table
    result_df = pd.DataFrame([{
        "tic": tic, "fyear": fyear, "industry": industry,
        "X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5,
        "LDA_Z_Score": z_score,
        "ML_Probability": ml_prob,
        "Risk_Zone": ml_risk
    }])
    st.dataframe(result_df)
