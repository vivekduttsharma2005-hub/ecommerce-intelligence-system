import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="E-commerce Intelligence", layout="wide", page_icon="🛒")

# ──────────────────────────────────────────────
# STYLES
# ──────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #e9ecef;
    text-align: center;
}
.metric-label { font-size: 13px; color: #6c757d; margin-bottom: 4px; }
.metric-value { font-size: 26px; font-weight: 600; color: #212529; }
.section-header { font-size: 18px; font-weight: 600; color: #212529; margin: 1.5rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
@st.cache_data

def load_data():
    df = pd.read_csv("data_sample.csv", encoding="ISO-8859-1")

    # Clean data
    df = df.dropna(subset=["CustomerID", "Description"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # Feature engineering
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Fix date parsing (VERY IMPORTANT)
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        dayfirst=True,
        errors="coerce"   # prevents crash
    )

    # Drop invalid dates
    df = df.dropna(subset=["InvoiceDate"])

    # Convert CustomerID
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

    return df

df = load_data()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
st.sidebar.title("E-commerce Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "👥 Customer Segmentation",
    "📈 Demand Forecasting",
    "🤖 Recommendations"
])

# ──────────────────────────────────────────────
# DATE FILTER (shared across pages)
# ──────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Date range**")
min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()
start_date, end_date = st.sidebar.date_input(
    "Select range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
mask = (df["InvoiceDate"].dt.date >= start_date) & (df["InvoiceDate"].dt.date <= end_date)
dff = df[mask]

if dff.empty:
    st.warning("No data in selected date range.")
    st.stop()

# ══════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Sales Dashboard")

    total_revenue = dff["TotalPrice"].sum()
    total_orders  = dff["InvoiceNo"].nunique()
    total_customers = dff["CustomerID"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Total Revenue", "Total Orders", "Unique Customers", "Avg Order Value"],
        [f"£{total_revenue:,.0f}", f"{total_orders:,}", f"{total_customers:,}", f"£{avg_order_value:,.2f}"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Monthly Revenue</div>", unsafe_allow_html=True)
    monthly = (
        dff.set_index("InvoiceDate")
        .resample("ME")["TotalPrice"]
        .sum()
        .reset_index()
    )
    fig_rev = px.area(
        monthly, x="InvoiceDate", y="TotalPrice",
        labels={"InvoiceDate": "Month", "TotalPrice": "Revenue (£)"},
        color_discrete_sequence=["#4361ee"]
    )
    fig_rev.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
    st.plotly_chart(fig_rev, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-header'>Top 10 Products by Revenue</div>", unsafe_allow_html=True)
        top_products = (
            dff.groupby("Description")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig_prod = px.bar(
            top_products, x="TotalPrice", y="Description", orientation="h",
            labels={"TotalPrice": "Revenue (£)", "Description": ""},
            color_discrete_sequence=["#7209b7"]
        )
        fig_prod.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=360, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_prod, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-header'>Revenue by Country</div>", unsafe_allow_html=True)
        top_countries = (
            dff.groupby("Country")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig_country = px.bar(
            top_countries, x="Country", y="TotalPrice",
            labels={"TotalPrice": "Revenue (£)", "Country": ""},
            color_discrete_sequence=["#f72585"]
        )
        fig_country.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=360)
        st.plotly_chart(fig_country, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 2 — CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════
elif page == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation (RFM + KMeans)")

    snapshot_date = dff["InvoiceDate"].max()

    rfm = dff.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum")
    ).reset_index()

    # Fix: normalize before KMeans
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    n_clusters = st.slider("Number of customer segments", 2, 6, 4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled).astype(str)

    cluster_labels = {str(i): f"Segment {i+1}" for i in range(n_clusters)}
    rfm["Segment"] = rfm["Cluster"].map(cluster_labels)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Recency vs Monetary</div>", unsafe_allow_html=True)
        fig_scatter = px.scatter(
            rfm, x="Recency", y="Monetary", color="Segment",
            hover_data=["CustomerID", "Frequency"],
            labels={"Recency": "Days since last purchase", "Monetary": "Total spend (£)"},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_scatter.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Frequency vs Monetary</div>", unsafe_allow_html=True)
        fig_scatter2 = px.scatter(
            rfm, x="Frequency", y="Monetary", color="Segment",
            hover_data=["CustomerID", "Recency"],
            labels={"Frequency": "Number of orders", "Monetary": "Total spend (£)"},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_scatter2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380)
        st.plotly_chart(fig_scatter2, use_container_width=True)

    st.markdown("<div class='section-header'>Segment Summary</div>", unsafe_allow_html=True)
    summary = rfm.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean")
    ).round(1).reset_index()
    st.dataframe(summary, use_container_width=True)

    with st.expander("View full RFM table"):
        st.dataframe(rfm.sort_values("Monetary", ascending=False), use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 3 — DEMAND FORECASTING
# ══════════════════════════════════════════════
elif page == "📈 Demand Forecasting":
    st.title("📈 Demand Forecasting")

    # Prepare time series
    ts = (
        dff.set_index("InvoiceDate")
        .resample("D")["TotalPrice"]
        .sum()
        .reset_index()
    )
    ts.columns = ["ds", "y"]
    ts = ts[ts["y"] > 0]

    horizon = st.slider("Forecast horizon (days)", 7, 90, 30)

    # TRY PROPHET FIRST
    try:
        from prophet import Prophet

        with st.spinner("Training Prophet model..."):
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.90
            )
            model.fit(ts)
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts["ds"], y=ts["y"],
            name="Actual", line=dict(color="#4361ee", width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat"],
            name="Forecast", line=dict(color="#f72585", width=2, dash="dash")
        ))

        st.plotly_chart(fig, use_container_width=True)

        st.success("Using Prophet model ✅")

    except Exception as e:
        st.warning("Prophet not supported in this environment ⚠️")
        
        # FALLBACK: MOVING AVERAGE FORECAST
        ts["Forecast"] = ts["y"].rolling(window=7).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts["ds"], y=ts["y"],
            name="Actual"
        ))
        fig.add_trace(go.Scatter(
            x=ts["ds"], y=ts["Forecast"],
            name="Forecast (7-day avg)",
            line=dict(dash="dash")
        ))

        st.plotly_chart(fig, use_container_width=True)

        st.info("Fallback model used: Moving Average")
# ══════════════════════════════════════════════
# PAGE 4 — RECOMMENDATIONS
# ══════════════════════════════════════════════
elif page == "🤖 Recommendations":
    st.title("🤖 Product Recommendations (Collaborative Filtering)")

    @st.cache_data
    def build_recommendation_model(data):
        # Build user-item matrix (rows = customers, cols = products)
        user_item = (
            data.groupby(["CustomerID", "Description"])["Quantity"]
            .sum()
            .unstack(fill_value=0)
        )
        # Compute cosine similarity between customers
        similarity = cosine_similarity(user_item)
        sim_df = pd.DataFrame(
            similarity,
            index=user_item.index,
            columns=user_item.index
        )
        return user_item, sim_df

    # Limit to top products for performance
    top_n_products = 200
    top_products = dff["Description"].value_counts().head(top_n_products).index
    dff_filtered = dff[dff["Description"].isin(top_products)]

    with st.spinner("Building recommendation model..."):
        user_item_matrix, similarity_matrix = build_recommendation_model(dff_filtered)

    customer_ids = sorted(user_item_matrix.index.tolist())
    selected_customer = st.selectbox("Select a customer ID", customer_ids)

    n_recommendations = st.slider("Number of recommendations", 3, 15, 5)

    def get_recommendations(customer_id, user_item, sim_matrix, n=5):
        if customer_id not in sim_matrix.index:
            return pd.Series(dtype=float)

        # Find most similar customers (excluding self)
        similar_customers = (
            sim_matrix[customer_id]
            .drop(customer_id)
            .sort_values(ascending=False)
            .head(10)
        )

        # Products already bought by this customer
        already_bought = set(user_item.loc[customer_id][user_item.loc[customer_id] > 0].index)

        # Weighted score from similar customers
        scores = {}
        for sim_customer, similarity_score in similar_customers.items():
            for product, qty in user_item.loc[sim_customer].items():
                if qty > 0 and product not in already_bought:
                    scores[product] = scores.get(product, 0) + similarity_score * qty

        return pd.Series(scores).sort_values(ascending=False).head(n)

    recommendations = get_recommendations(
        selected_customer, user_item_matrix, similarity_matrix, n_recommendations
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='section-header'>Recommended products</div>", unsafe_allow_html=True)
        if recommendations.empty:
            st.info("Not enough data to generate recommendations for this customer.")
        else:
            rec_df = recommendations.reset_index()
            rec_df.columns = ["Product", "Score"]
            rec_df["Score"] = (rec_df["Score"] / rec_df["Score"].max() * 100).round(1)
            for _, row in rec_df.iterrows():
                bar_width = int(row["Score"])
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="font-size:13px;font-weight:500;color:var(--text-color);margin-bottom:3px;">{row['Product']}</div>
                    <div style="background:#e9ecef;border-radius:6px;height:8px;width:100%;">
                        <div style="background:#4361ee;border-radius:6px;height:8px;width:{bar_width}%;"></div>
                    </div>
                    <div style="font-size:11px;color:#6c757d;margin-top:2px;">Relevance: {row['Score']}%</div>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-header'>This customer's purchase history</div>", unsafe_allow_html=True)
        customer_history = (
            dff[dff["CustomerID"] == selected_customer]
            .groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig_hist = px.bar(
            customer_history, x="Quantity", y="Description", orientation="h",
            labels={"Quantity": "Units bought", "Description": ""},
            color_discrete_sequence=["#7209b7"]
        )
        fig_hist.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=360,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_hist, use_container_width=True)
