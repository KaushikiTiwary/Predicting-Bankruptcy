import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('lda_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🧠 Bankruptcy Risk Predictor")

uploaded_file = st.file_uploader("📄 Upload a CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### Input Data Preview", df.head())

    # Extract only the X1 to X5 columns
    X = df[['X1', 'X2', 'X3', 'X4', 'X5']]
    X_scaled = scaler.transform(X)

    # Predict
    probs = model.predict_proba(X_scaled)[:, 1]
    df['lda_probability'] = probs

    # Risk classification thresholds (from your model)
    q25, q50, q75 = 0.49, 0.50, 0.51

    def classify(prob):
        if prob < q25:
            return "🔴 Very High Risk"
        elif prob < q50:
            return "🟧 High Risk"
        elif prob < q75:
            return "🟨 Medium Risk"
        else:
            return "🟩 Very Low Risk"

    df['risk_zone'] = df['lda_probability'].apply(classify)

    st.write("### 🧾 Bankruptcy Risk Results")
    st.dataframe(df[['tic', 'fyear', 'lda_probability', 'risk_zone']])
