import streamlit as st
import pandas as pd
import joblib
import altair as alt

# Set page config
st.set_page_config(page_title="Bankruptcy Predictor", layout="wide")

# Load model and scaler
model = joblib.load('lda_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🧠 Bankruptcy Risk Predictor")
st.write("### 🔍 Filter & Visualize Bankruptcy Risk")

# File upload
uploaded_file = st.file_uploader("\ud83d\udcc4 Upload a CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preview
    st.write("### 📃 Input Data Preview")
    st.dataframe(df.head())

    # Explanations
    with st.expander("ℹ️ What do the variables X1 to X5 mean?"):
        st.markdown("""
        - **X1**: Working Capital / Total Assets  
        - **X2**: Retained Earnings / Total Assets  
        - **X3**: EBIT / Total Assets  
        - **X4**: Market Value of Equity / Total Liabilities  
        - **X5**: Sales / Total Assets
        """)

    # Filter X variables and scale
    X = df[['X1', 'X2', 'X3', 'X4', 'X5']]
    X_scaled = scaler.transform(X)

    # Predict probabilities
    df['lda_probability'] = model.predict_proba(X_scaled)[:, 1]

    # Risk thresholds
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

    # Show result
    st.write("### 📜 Bankruptcy Risk Results")
    st.dataframe(df[['tic', 'fyear', 'lda_probability', 'risk_zone']])

    # Download option
    csv = df.to_csv(index=False)
    st.download_button("\ud83d\udce5 Download Results as CSV", csv, file_name="bankruptcy_risk_results.csv", mime="text/csv")

    # Select for visualization
    st.write("### \ud83d\udcca Company Risk Trend Viewer")
    col1, col2 = st.columns(2)

    with col1:
        selected_tic = st.selectbox("Choose a Ticker (Company)", sorted(df['tic'].unique()))
    with col2:
        selected_industry = st.selectbox("Choose an Industry", sorted(df['industry'].dropna().unique()))

    filtered_df = df[(df['tic'] == selected_tic) & (df['industry'] == selected_industry)]

    if not filtered_df.empty:
        st.write(f"#### Risk Trend for {selected_tic} in {selected_industry}")
        st.dataframe(filtered_df[['fyear', 'lda_probability', 'risk_zone']])

        # Altair chart
        chart = alt.Chart(filtered_df).mark_line(point=True).encode(
            x='fyear:O',
            y='lda_probability:Q',
            tooltip=['fyear', 'lda_probability', 'risk_zone']
        ).properties(
            title=f"\ud83d\udcc9 Bankruptcy Risk Trend for {selected_tic}"
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No data found for selected filters. Try changing the ticker or industry.")
