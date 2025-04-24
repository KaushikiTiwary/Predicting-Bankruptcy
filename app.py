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

    # Extract and scale features
    X = df[['X1', 'X2', 'X3', 'X4', 'X5']]
    X_scaled = scaler.transform(X)

    # Predict probabilities
    probs = model.predict_proba(X_scaled)[:, 1]
    df['lda_probability'] = probs

    # Classification logic
    def classify_zone(prob):
        if prob < 0.49:
            return "🔴 Very High Risk"
        elif prob < 0.50:
            return "🟠 High Risk"
        elif prob <= 0.50:
            return "🟡 Medium Risk"
        else:
            return "🟢 Very Low Risk"

    df['risk_zone'] = df['lda_probability'].apply(classify_zone)

    # Display results
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

    # Risk trend for specific company
    st.write("### 📊 View Risk Trend for a Specific Company")
    selected_tic = st.selectbox("Choose a company (ticker)", sorted(df['tic'].unique()))
    company_df = df[df['tic'] == selected_tic].copy()

    st.write(f"#### Risk Data for {selected_tic}")
    st.dataframe(company_df[['fyear', 'lda_probability', 'risk_zone']])

    # Define risk level order for y-axis
    risk_order = ["🔴 Very High Risk", "🟠 High Risk", "🟡 Medium Risk", "🟢 Very Low Risk"]
    company_df['risk_zone'] = pd.Categorical(company_df['risk_zone'], categories=risk_order, ordered=True)

    # Categorical line chart
    chart = alt.Chart(company_df).mark_line(point=True).encode(
        x=alt.X('fyear:O', title='Fiscal Year'),
        y=alt.Y('risk_zone:N', sort=risk_order, title='Risk Level'),
        color=alt.Color('risk_zone:N', scale=alt.Scale(scheme='redyellowgreen')),
        tooltip=['fyear', 'lda_probability', 'risk_zone']
    ).properties(
        title=f"📉 Risk Category Trend for {selected_tic}"
    )

    st.altair_chart(chart, use_container_width=True)

