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

    st.write("### 📋 Input Data Preview")
    st.dataframe(df.head())

    # Explain ratios
    with st.expander("ℹ️ What do the variables X1 to X5 mean?"):
        st.markdown("""
        - **X1**: Working Capital / Total Assets  
        - **X2**: Retained Earnings / Total Assets  
        - **X3**: EBIT / Total Assets  
        - **X4**: Market Value of Equity / Total Liabilities  
        - **X5**: Sales / Total Assets
        """)

    # Predict probability
    X = df[['X1', 'X2', 'X3', 'X4', 'X5']]
    X_scaled = scaler.transform(X)
    df['lda_probability'] = model.predict_proba(X_scaled)[:, 1]

    # Risk classification
    def classify(prob):
        if prob < 0.49:
            return "🔴 Very High Risk"
        elif prob < 0.50:
            return "🟠 High Risk"
        elif prob <= 0.51:
            return "🟡 Medium Risk"
        else:
            return "🟢 Very Low Risk"

    df['risk_zone'] = df['lda_probability'].apply(classify)

    # -----------------------------
    # 📊 Filter & Trend Section
    st.markdown("### 🔎 Filter & Visualize Bankruptcy Risk")

    # Dropdowns
    selected_industry = st.selectbox("Choose an Industry", sorted(df['industry'].dropna().unique()))
    filtered_by_industry = df[df['industry'] == selected_industry]

    selected_tic = st.selectbox("Choose a Company (Ticker)", sorted(filtered_by_industry['tic'].unique()))
    company_df = filtered_by_industry[filtered_by_industry['tic'] == selected_tic].copy()

    st.write(f"#### Company: `{selected_tic}` | Industry: `{selected_industry}`")
    st.dataframe(company_df[['fyear', 'lda_probability', 'risk_zone']])

    # Altair chart with thresholds
    thresholds = {
        "🔴 Very High Risk": (0.0, 0.49),
        "🟠 High Risk": (0.49, 0.50),
        "🟡 Medium Risk": (0.50, 0.51),
        "🟢 Very Low Risk": (0.51, 1.0),
    }

    # Bands
    bands = []
    for label, (low, high) in thresholds.items():
        bands.append(
            alt.Chart(pd.DataFrame({
                'y': [low], 'y2': [high], 'risk': [label]
            })).mark_rect(opacity=0.2).encode(
                y='y:Q',
                y2='y2:Q',
                color=alt.Color('risk:N', scale=alt.Scale(
                    domain=list(thresholds.keys()),
                    range=['red', 'orange', 'gold', 'green']
                ))
            )
        )

    line = alt.Chart(company_df).mark_line(point=True).encode(
        x=alt.X('fyear:O', title='Fiscal Year'),
        y=alt.Y('lda_probability:Q', title='Probability of Bankruptcy'),
        tooltip=['fyear', 'lda_probability', 'risk_zone']
    )

    final_chart = alt.layer(*bands, line).properties(
        title=f"📈 Risk Evolution for {selected_tic} (Unseen Company)"
    ).configure_title(fontSize=16)

    st.altair_chart(final_chart, use_container_width=True)

    # Download button
    csv = df.to_csv(index=False)
    st.download_button("📥 Download Full Predictions", data=csv, file_name='bankruptcy_risk_results.csv')
