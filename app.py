# Filter: Industry → Ticker → Year (optional)
st.write("### 🔍 Filter & Visualize Bankruptcy Risk")

selected_industry = st.selectbox("Select Industry", sorted(df['industry'].unique()))

filtered_df = df[df['industry'] == selected_industry]

selected_tic = st.selectbox("Select Company (Ticker)", sorted(filtered_df['tic'].unique()))
company_df = filtered_df[filtered_df['tic'] == selected_tic]

years = sorted(company_df['fyear'].unique())
selected_year_range = st.slider("Select Year Range", min_value=min(years), max_value=max(years), value=(min(years), max(years)))

company_df = company_df[(company_df['fyear'] >= selected_year_range[0]) & (company_df['fyear'] <= selected_year_range[1])]

st.write(f"### 🏢 Company: `{selected_tic}` | 🏭 Industry: `{selected_industry}`")
st.dataframe(company_df[['fyear', 'lda_probability', 'risk_zone']])

chart = alt.Chart(company_df).mark_line(point=True).encode(
    x=alt.X('fyear:O', title="Financial Year"),
    y=alt.Y('lda_probability:Q', title="Probability of Bankruptcy"),
    tooltip=['fyear', 'lda_probability', 'risk_zone']
).properties(
    title=f"📉 Bankruptcy Trend for {selected_tic}"
)

st.altair_chart(chart, use_container_width=True)
