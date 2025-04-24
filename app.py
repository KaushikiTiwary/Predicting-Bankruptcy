import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('lda_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🧠 Bankruptcy Risk Predictor")
st.markdown("Upload your dataset and visualize the bankruptcy probability over time for each company.")

uploaded_file = st.file_uploader("📄 Upload a CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 🧾 Input Data Preview")
    st.dataframe(df.head())

    # Explain features
    with st.expander("ℹ️ What do the variables X1 to X5 mean?"):
        st.markdown("""
        - **X1**: Working Capital / Total Assets  
        - **X2**: Retained Earnings / Total Assets  
        - **X3**: EBIT / Total Assets  
        - **X4**: Market Value of Equity / Total Liabilities  
        - **X5**: Sales / Total Assets
        """)

    # Predict bankruptcy probabilities
    features = df[['X1', 'X2', 'X3', 'X4', 'X5']]
    features_scaled = scaler.transform(features)
    df['lda_probability'] = model.predict_proba(features_scaled)[:, 1]

    # Risk classification
    def get_risk(prob):
        if prob < 0.49:
            return "🔴 Very High Risk"
        elif prob < 0.50:
            return "🟠 High Risk"
        elif prob <= 0.51:
            return "🟡 Medium Risk"
        else:
            return "🟢 Very Low Risk"

    df['risk_zone'] = df['lda_probability'].apply(get_risk)

    # Show results table
    st.write("### 📊 Bankruptcy Risk Results")
    st.dataframe(df[['tic', 'fyear', 'industry', 'lda_probability', 'risk_zone']])

    # Save download
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Risk Predictions", csv, "bankruptcy_results.csv", "text/csv")

    # ------------------- INTERACTIVE FILTERS + LINE CHART -------------------
    st.write("## 📈 Visualize Bankruptcy Trend")

    # Dropdown filters
    industry_options = sorted(df['industry'].dropna().unique())
    selected_industry = st.selectbox("Filter by Industry", industry_options)

    tic_options = sorted(df[df['industry'] == selected_industry]['tic'].unique())
    selected_tic = st.selectbox("Choose Company (Ticker)", tic_options)

    # Filter for selected company
    company_df = df[(df['tic'] == selected_tic)].sort_values('fyear')

    st.write(f"### Risk Data for `{selected_tic}`")
    st.dataframe(company_df[['fyear', 'lda_probability', 'risk_zone']])

    # Plot using Matplotlib (to match your original chart)
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(company_df['fyear'], company_df['lda_probability'], marker='o', linestyle='-')
    ax.set_xlabel("Financial Year")
    ax.set_ylabel("Probability of Bankruptcy")
    ax.set_title(f"📉 Risk Evolution for {selected_tic} (Unseen Company)")

    # Add thresholds
    ax.axhline(0.49, color='red', linestyle='--', label='Very High Risk Threshold')
    ax.axhline(0.50, color='orange', linestyle='--', label='High Risk Threshold')
    ax.axhline(0.51, color='green', linestyle='--', label='Very Low Risk Threshold')

    ax.legend()
    st.pyplot(fig)
