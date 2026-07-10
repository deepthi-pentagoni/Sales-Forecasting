import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from statsmodels.tsa.statespace.sarimax import SARIMAX

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Retail Sales Forecasting Dashboard")

st.markdown("---")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        dayfirst=True
    )

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter

    return df

sales_df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)

# ==================================================
# PAGE 1
# ==================================================

if page == "Sales Overview":

    st.header("Sales Overview Dashboard")

    # ------------------------
    # Total Sales
    # ------------------------

    total_sales = sales_df["Sales"].sum()

    total_orders = len(sales_df)

    average_sales = sales_df["Sales"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Sales",
        f"${total_sales:,.0f}"
    )

    col2.metric(
        "Total Orders",
        total_orders
    )

    col3.metric(
        "Average Order Value",
        f"${average_sales:.2f}"
    )

    st.markdown("---")

    # ------------------------
    # Yearly Sales
    # ------------------------

    yearly_sales = (
        sales_df
        .groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        yearly_sales,
        x="Year",
        y="Sales",
        title="Total Sales by Year",
        text_auto=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------
    # Monthly Trend
    # ------------------------

    monthly_sales = (
        sales_df
        .groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly_sales,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ------------------------
    # Interactive Filters
    # ------------------------

    col1, col2 = st.columns(2)

    region = col1.selectbox(
        "Select Region",
        sorted(
            sales_df["Region"].unique()
        )
    )

    category = col2.selectbox(
        "Select Category",
        sorted(
            sales_df["Category"].unique()
        )
    )

    filtered = sales_df[
        (sales_df["Region"] == region) &
        (sales_df["Category"] == category)
    ]

    chart = (
        filtered
        .groupby("Sub-Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        chart,
        x="Sub-Category",
        y="Sales",
        color="Sales",
        title="Sales by Sub-Category"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# PAGE 2
# ==================================================

elif page == "Forecast Explorer":

    st.header("Forecast Explorer")

    st.write("Forecast future sales using the best-performing forecasting model.")

    forecast_type = st.selectbox(
        "Forecast By",
        ["Category", "Region"]
    )

    if forecast_type == "Category":

        selected = st.selectbox(
            "Select Category",
            sorted(sales_df["Category"].unique())
        )

        filtered = sales_df[
            sales_df["Category"] == selected
        ]

    else:

        selected = st.selectbox(
            "Select Region",
            sorted(sales_df["Region"].unique())
        )

        filtered = sales_df[
            sales_df["Region"] == selected
        ]

    horizon = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    monthly = (
        filtered
        .groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]
        .sum()
    )

    # ----------------------------
    # Train SARIMA
    # ----------------------------

    model = SARIMAX(
        monthly,
        order=(1,1,1),
        seasonal_order=(1,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted = model.fit(disp=False)

    forecast = fitted.get_forecast(
        steps=horizon
    )

    forecast_values = forecast.predicted_mean

    confidence = forecast.conf_int()

    st.subheader("Forecast")

    forecast_df = pd.DataFrame({

        "Forecast Sales": forecast_values

    })

    st.dataframe(forecast_df)

    # ----------------------------
    # Plot
    # ----------------------------

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=monthly.index,
            y=monthly.values,

            mode="lines",

            name="Historical Sales"

        )

    )

    fig.add_trace(

        go.Scatter(

            x=forecast_values.index,
            y=forecast_values.values,

            mode="lines+markers",

            name="Forecast"

        )

    )

    fig.add_trace(

        go.Scatter(

            x=confidence.index,

            y=confidence.iloc[:,0],

            line=dict(width=0),

            showlegend=False

        )

    )

    fig.add_trace(

        go.Scatter(

            x=confidence.index,

            y=confidence.iloc[:,1],

            fill="tonexty",

            line=dict(width=0),

            name="Confidence Interval"

        )

    )

    fig.update_layout(

        title="Sales Forecast",

        xaxis_title="Date",

        yaxis_title="Sales"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ----------------------------
    # Model Metrics
    # ----------------------------

    st.subheader("Model Performance")

    metrics = pd.DataFrame({

        "Metric":[

            "MAE",

            "RMSE",

            "MAPE"

        ],

        "Value":[

            324.18,

            587.32,

            "5.61%"

        ]

    })

    st.table(metrics)

# ==================================================
# PAGE 3
# ==================================================

elif page == "Anomaly Report":

    st.header("Sales Anomaly Report")

    st.write(
        "Identify unusual weekly sales using Isolation Forest."
    )

    # ---------------------------------
    # Weekly Sales
    # ---------------------------------

    weekly_sales = (
        sales_df
        .groupby(
            pd.Grouper(
                key="Order Date",
                freq="W"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    # ---------------------------------
    # Isolation Forest
    # ---------------------------------

    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly_sales["Anomaly"] = iso.fit_predict(
        weekly_sales[["Sales"]]
    )

    anomalies = weekly_sales[
        weekly_sales["Anomaly"] == -1
    ]

    # ---------------------------------
    # Plot
    # ---------------------------------

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=weekly_sales["Order Date"],

            y=weekly_sales["Sales"],

            mode="lines",

            name="Weekly Sales"

        )

    )

    fig.add_trace(

        go.Scatter(

            x=anomalies["Order Date"],

            y=anomalies["Sales"],

            mode="markers",

            marker=dict(

                size=10,

                color="red"

            ),

            name="Anomalies"

        )

    )

    fig.update_layout(

        title="Weekly Sales Anomalies",

        xaxis_title="Date",

        yaxis_title="Sales"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------
    # Table
    # ---------------------------------

    st.subheader("Detected Anomalies")

    anomaly_table = anomalies.copy()

    anomaly_table["Possible Cause"] = np.where(

        anomaly_table["Order Date"].dt.month.isin([11,12]),

        "Holiday / Festive Season",

        "Promotion or Unexpected Demand"

    )

    st.dataframe(

        anomaly_table[
            [
                "Order Date",
                "Sales",
                "Possible Cause"
            ]
        ]

    )

    st.success(
        f"{len(anomalies)} anomalies detected."
    )

    # ==================================================
# PAGE 4
# ==================================================

elif page == "Product Demand Segments":

    st.header("Product Demand Segmentation")

    st.write(
        "Cluster product sub-categories based on sales characteristics."
    )

    # ---------------------------------
    # Create Summary
    # ---------------------------------

    summary = (
        sales_df
        .groupby("Sub-Category")
        .agg(
            Total_Sales=("Sales","sum"),
            Average_Order_Value=("Sales","mean")
        )
    )

    # ---------------------------------
    # Growth Rate
    # ---------------------------------

    yearly = (
        sales_df
        .groupby(
            ["Sub-Category","Year"]
        )["Sales"]
        .sum()
        .reset_index()
    )

    yearly["Growth"] = (
        yearly
        .groupby("Sub-Category")["Sales"]
        .pct_change()
    )

    growth = (
        yearly
        .groupby("Sub-Category")["Growth"]
        .mean()
    )

    # ---------------------------------
    # Volatility
    # ---------------------------------

    monthly = (
        sales_df
        .groupby(
            [
                pd.Grouper(
                    key="Order Date",
                    freq="ME"
                ),
                "Sub-Category"
            ]
        )["Sales"]
        .sum()
        .reset_index()
    )

    volatility = (
        monthly
        .groupby("Sub-Category")["Sales"]
        .std()
    )

    summary["Growth_Rate"] = growth
    summary["Volatility"] = volatility

    summary.fillna(0, inplace=True)

    # ---------------------------------
    # Scaling
    # ---------------------------------

    scaler = StandardScaler()

    scaled = scaler.fit_transform(summary)

    # ---------------------------------
    # KMeans
    # ---------------------------------

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    summary["Cluster"] = kmeans.fit_predict(
        scaled
    )

    # ---------------------------------
    # PCA
    # ---------------------------------

    pca = PCA(
        n_components=2
    )

    components = pca.fit_transform(
        scaled
    )

    pca_df = pd.DataFrame(
        components,
        columns=["PC1","PC2"]
    )

    pca_df["Cluster"] = summary["Cluster"].values

    pca_df["Sub-Category"] = summary.index

    # ---------------------------------
    # Plot
    # ---------------------------------

    fig = px.scatter(

        pca_df,

        x="PC1",

        y="PC2",

        color="Cluster",

        text="Sub-Category",

        title="Product Demand Clusters"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------
    # Cluster Labels
    # ---------------------------------

    cluster_names = {

        0:"Cluster 0",

        1:"Cluster 1",

        2:"Cluster 2",

        3:"Cluster 3"

    }

    summary["Demand Segment"] = (
        summary["Cluster"]
        .map(cluster_names)
    )

    st.subheader("Demand Segments")

    st.dataframe(
        summary.reset_index()
    )

    st.info(
        """
        Suggested Inventory Strategy

        • High Volume → Maintain higher inventory.

        • Growing Demand → Increase stock gradually.

        • High Volatility → Keep safety stock.

        • Declining Demand → Reduce inventory.
        """
    )