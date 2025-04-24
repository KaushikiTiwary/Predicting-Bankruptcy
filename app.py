import streamlit as st
import pandas as pd
import joblib
import altair as alt

# Load model and scaler
model = joblib.load('lda_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🧠 Bankruptcy Risk Predictor")

uploaded_file = st.file_uploader("📄 Upload a CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### Input Data Preview", df.head())

    # Show variable descriptions
    with st.expander("ℹ️ What do the variables X1 to X5 mean?"):
        st.markdown("""
        - *X1*: Working Capital / Total Assets  
        - *X2*: Retained Earnings / Total Assets  
        - *X3*: EBIT / Total Assets  
        - *X4*: Market Value of Equity / Total Liabilities  
        - *X5*: Sales / Total Assets
        """)

    # Extract only the X1 to X5 columns
    X = df[['X1', 'X2', 'X3', 'X4', 'X5']]
    X_scaled = scaler.transform(X)

    # Predict
    probs = model.predict_proba(X_scaled)[:, 1]
    df['lda_probability'] = probs

    # Risk classification thresholds
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

    # Show full results
    st.write("### 🧾 Bankruptcy Risk Results")
    st.dataframe(df[['tic', 'fyear', 'lda_probability', 'risk_zone']])

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name='bankruptcy_risk_results.csv',
        mime='text/csv'
    )

    # Company-specific risk trend
    st.write("### 📊 View Risk Trend for a Specific Company")
    selected_tic = st.selectbox("Choose a company (ticker)", sorted(df['tic'].unique()))
    company_df = df[df['tic'] == selected_tic]

    st.write(f"#### Risk Data for {selected_tic}")
    st.dataframe(company_df[['fyear', 'lda_probability', 'risk_zone']])

    chart = alt.Chart(company_df).mark_line(point=True).encode(
        x='fyear:O',
        y='lda_probability:Q',
        tooltip=['fyear', 'lda_probability', 'risk_zone']
    ).properties(
        title=f"📉 Bankruptcy Risk Trend for {selected_tic}"
    )

    st.altair_chart(chart, use_container_width=True)
