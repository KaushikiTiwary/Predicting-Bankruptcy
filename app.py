import streamlit as st
import pandas as pd
import joblib
import altair as alt

# Load model and scaler
model = joblib.load('lda_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🧠 Bankruptcy Risk Predictor")

# File uploader for CSV
uploaded_file = st.file_uploader("📄 Upload a CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Display data preview
    st.write("### Input Data Preview", df.head())

    # Variable descriptions
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

    # Risk zone classification logic
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

    # Dropdowns for Company and Year
    st.write("### 🔎 Filter Risk Trend")
    selected_tic = st.selectbox("Choose a company (ticker)", sorted(df['tic'].unique()))
    selected_year = st.selectbox("Choose a fiscal year", sorted(df['fyear'].unique()))

    filtered_df = df[(df['tic'] == selected_tic) & (df['fyear'] == selected_year)]

    st.write(f"#### Risk Data for {selected_tic} in {selected_year}")
    st.dataframe(filtered_df[['fyear', 'lda_probability', 'risk_zone']])

    # Define risk order
    risk_order = ["🔴 Very High Risk", "🟠 High Risk", "🟡 Medium Risk", "🟢 Very Low Risk"]
    df['risk_zone'] = pd.Categorical(df['risk_zone'], categories=risk_order, ordered=True)

    # Filtered chart across years (optional: drop year filter for trend)
    trend_df = df[df['tic'] == selected_tic].copy()

    # Define risk thresholds
    thresholds = {
        "🔴 Very High Risk": (0.0, 0.49),
        "🟠 High Risk": (0.49, 0.50),
        "🟡 Medium Risk": (0.50, 0.51),
        "🟢 Very Low Risk": (0.51, 1.0),
    }

    # Create background colored bands for each risk zone
    bands = []
    for label, (low, high) in thresholds.items():
        bands.append(
            alt.Chart(pd.DataFrame({
                'y': [low],
                'y2': [high],
                'risk': [label]
            })).mark_rect(opacity=0.2).encode(
                y='y:Q',
                y2='y2:Q',
                color=alt.Color('risk:N',
                                scale=alt.Scale(domain=list(thresholds.keys()),
                                                range=['red', 'orange', 'gold', 'green']),
                                legend=alt.Legend(orient="top", title="Risk Zone"))
            )
        )

    # Main line chart
    line = alt.Chart(trend_df).mark_line(point=True).encode(
        x=alt.X('fyear:O', title='Fiscal Year'),
        y=alt.Y('lda_probability:Q', title='LDA Probability'),
        tooltip=['fyear', 'lda_probability', 'risk_zone']
    )

    # Combine bands + line
    final_chart = alt.layer(*bands, line).properties(
        title=f"📈 Bankruptcy Probability & Risk Zones for {selected_tic}"
    ).configure_title(
        fontSize=16,
        anchor='start'
    )

    st.altair_chart(final_chart, use_container_width=True)
