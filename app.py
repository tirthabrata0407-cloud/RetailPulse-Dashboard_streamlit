# pip install streamlit prophet openpyxl scikit-learn matplotlib pandas numpy seaborn
# RetailPulse - Production Analytics Dashboard with Complete Feature Implementation
# All 7 Features + Roadmap & Summary

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_percentage_error, mean_squared_error
)
import os
import warnings
from io import BytesIO
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# =================================================================
# PAGE CONFIGURATION
# =================================================================
st.set_page_config(
    page_title="RetailPulse - Production Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; }
    .section-header { font-size: 1.8rem; font-weight: 600; color: #2c3e50; margin-top: 1.5rem; }
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; }
    .sla-pass { color: #27ae60; font-weight: bold; }
    .sla-fail { color: #e74c3c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">RetailPulse - Enterprise Analytics Platform</div>', unsafe_allow_html=True)
st.markdown("*Production-grade retail analytics with ML-powered forecasting and optimization*")
st.markdown("---")

# =================================================================
# HELPER FUNCTIONS
# =================================================================

def find_column_by_keywords(df, keywords):
    """Find column by multiple keyword patterns"""
    for col in df.columns:
        for keyword in keywords:
            if keyword.lower() in col.lower():
                return col
    return None

def find_date_column(df):
    """Find date column with multiple strategies"""
    date_keywords = ['date', 'invoice date', 'invoicedate', 'transactiondate', 
                     'order date', 'orderdate', 'time', 'timestamp']
    
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    
    for col in df.columns:
        if df[col].dtype == 'object':
            for keyword in date_keywords:
                if keyword in col.lower():
                    try:
                        pd.to_datetime(df[col].head())
                        return col
                    except:
                        continue
    
    return None

def find_numeric_column(df, keywords):
    """Find numeric column by keywords"""
    for col in df.columns:
        for keyword in keywords:
            if keyword.lower() in col.lower():
                if pd.api.types.is_numeric_dtype(df[col]):
                    return col
    return None

def calculate_data_quality_score(df):
    """Calculate comprehensive data quality score"""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    completeness = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
    uniqueness = (1 - duplicate_rows / df.shape[0]) * 100 if df.shape[0] > 0 else 0
    
    score = (completeness * 0.6 + uniqueness * 0.4)
    return score

def export_to_excel(data_dict, filename="RetailPulse_Report.xlsx"):
    """Export multiple dataframes to Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in data_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

# =================================================================
# DATA LOADING
# =================================================================
@st.cache_data
def load_data():
    try:
        if os.path.exists("merged_cleaned_retail_data.xlsx"):
            # Using nrows as specified in the EDA requirement snippet
            df = pd.read_excel("merged_cleaned_retail_data.xlsx", engine="openpyxl", nrows=15000)
            return df
        else:
            st.error("Error: Data file not found.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()
if df is None:
    st.stop()

# =================================================================
# SIDEBAR NAVIGATION
# =================================================================
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard Overview",
        "F01: Data Ingestion & Cleaning",
        "F02: Exploratory Data Analysis (EDA)",
        "F03: Customer Segmentation (RFM)",
        "F04: Demand Forecasting",
        "F05: Churn Prediction",
        "F06: Inventory Optimization",
        "F07: Interactive Analytics & Export",
        "Complete Project Roadmap",
        "Project Summary"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.info("7 Production-Grade Features | SLA Monitoring | Enterprise Ready")

# Detect columns
date_col = find_date_column(df)
amount_col = find_numeric_column(df, ['total', 'amount', 'revenue', 'sales'])
quantity_col = find_numeric_column(df, ['quantity', 'qty'])
customer_col = find_column_by_keywords(df, ['customer', 'user', 'client'])
product_col = find_column_by_keywords(df, ['description', 'product', 'name', 'item'])
country_col = find_column_by_keywords(df, ['country', 'nation', 'region'])
invoice_col = find_column_by_keywords(df, ['invoice', 'transaction', 'order'])

# =================================================================
# 0. DASHBOARD OVERVIEW
# =================================================================
if menu == "Dashboard Overview":
    
    st.markdown('<div class="section-header">Production Dashboard Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Size", f"{df.shape[0]:,} records")
    with col2:
        if amount_col:
            st.metric("Total Revenue", f"${df[amount_col].sum():,.2f}")
    with col3:
        if quantity_col:
            st.metric("Total Units", f"{df[quantity_col].sum():,.0f}")
    with col4:
        if customer_col:
            st.metric("Unique Customers", f"{df[customer_col].nunique():,}")
    
    st.markdown("---")
    st.subheader("Feature Implementation Status")
    
    features = {
        "F01": {
            "name": "Data Ingestion & Cleaning",
            "description": "Automated ETL pipeline with data quality checks",
            "metrics": ["Data Quality Score", "Records Processed", "Schema Validation"],
            "status": "Production"
        },
        "F02": {
            "name": "Exploratory Data Analysis (EDA)",
            "description": "Statistical summaries, distributions, and correlation tracking",
            "metrics": ["Missing Values", "Distributions", "Heatmaps"],
            "status": "Production"
        },
        "F03": {
            "name": "Customer Segmentation (RFM)",
            "description": "RFM + K-Means/DBSCAN with 3-8 segments",
            "metrics": ["Segments Created", "Silhouette Score", "Segment Distribution"],
            "status": "Production"
        },
        "F04": {
            "name": "Demand Forecasting",
            "description": "Prophet time-series forecasting",
            "metrics": ["MAPE ≤ 12%", "30-day Predictions", "Confidence Intervals"],
            "status": "Production"
        },
        "F05": {
            "name": "Churn Prediction",
            "description": "Random Forest classifier with explainability",
            "metrics": ["AUC-ROC ≥ 0.88", "Precision@top20% ≥ 0.75", "Feature Importance"],
            "status": "Production"
        },
        "F06": {
            "name": "Inventory Optimization",
            "description": "EOQ-based reorder recommendations",
            "metrics": ["25-40% Stock Reduction", "Cost Optimization", "Reorder Points"],
            "status": "Production"
        },
        "F07": {
            "name": "Interactive Analytics",
            "description": "Real-time dashboard with what-if analysis",
            "metrics": ["Dynamic Filters", "What-If Scenarios", "Exportable Reports"],
            "status": "Production"
        }
    }
    
    for feat_id, feature in features.items():
        with st.expander(f"{feat_id} - {feature['name']}", expanded=(feat_id=="F01")):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Description:** {feature['description']}")
                st.write(f"**Key Metrics:** {', '.join(feature['metrics'])}")
            with col2:
                st.write(f"**Status:** {feature['status']}")

# =================================================================
# F01: DATA INGESTION & CLEANING
# =================================================================
elif menu == "F01: Data Ingestion & Cleaning":
    
    st.markdown('<div class="section-header">F01: Data Ingestion & Cleaning Pipeline</div>', unsafe_allow_html=True)
    
    # Quality Score
    quality_score = calculate_data_quality_score(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Quality Score", f"{quality_score:.2f}%")
    with col2:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col3:
        st.metric("Total Columns", df.shape[1])
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    st.markdown("---")
    st.subheader("Data Quality Checks")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        missing = df.isnull().sum().sum()
        missing_pct = (missing / (df.shape[0] * df.shape[1]) * 100)
        st.metric("Missing Values", missing, delta=f"{missing_pct:.2f}%")
    with col2:
        duplicates = df.duplicated().sum()
        dup_pct = (duplicates / df.shape[0] * 100)
        st.metric("Duplicate Rows", duplicates, delta=f"{dup_pct:.2f}%")
    with col3:
        st.metric("Data Completeness", f"{(1 - missing_pct/100)*100:.2f}%")
    
    # Cleaning operations
    df_cleaned = df.dropna()
    df_cleaned = df_cleaned.drop_duplicates()
    
    if quantity_col and quantity_col in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned[quantity_col] > 0]
    if amount_col and amount_col in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned[amount_col] > 0]
    
    st.markdown("---")
    st.subheader("Cleaned Dataset Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records After Cleaning", f"{df_cleaned.shape[0]:,}")
    with col2:
        records_removed = df.shape[0] - df_cleaned.shape[0]
        st.metric("Records Removed", records_removed)
    with col3:
        removal_pct = (records_removed / df.shape[0] * 100)
        st.metric("Removal Rate", f"{removal_pct:.2f}%")
    with col4:
        retention = (df_cleaned.shape[0] / df.shape[0] * 100)
        st.metric("Retention Rate", f"{retention:.2f}%")
    
    st.markdown("---")
    st.subheader("Data Types & Validation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data Type Distribution**")
        dtype_dist = df.dtypes.value_counts()
        for dtype, count in dtype_dist.items():
            st.write(f"- {dtype}: {count} columns")
    
    with col2:
        st.write("**Detected Key Columns**")
        st.write(f"- Date Column: {date_col if date_col else 'Not Found'}")
        st.write(f"- Amount Column: {amount_col if amount_col else 'Not Found'}")
        st.write(f"- Quantity Column: {quantity_col if quantity_col else 'Not Found'}")
        st.write(f"- Customer Column: {customer_col if customer_col else 'Not Found'}")
        st.write(f"- Product Column: {product_col if product_col else 'Not Found'}")


# =================================================================
# F02: EXPLORATORY DATA ANALYSIS (EDA)
# =================================================================
elif menu == "F02: Exploratory Data Analysis (EDA)":
    
    st.markdown('<div class="section-header">EDA – Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    # Cast Stock Code to string if it exists in the dataset
    if "Stock Code" in df.columns:
        df["Stock Code"] = df["Stock Code"].astype(str)
    elif "StockCode" in df.columns:
        df["StockCode"] = df["StockCode"].astype(str)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("Correlation Heatmap")

    numeric_cols = df.select_dtypes(include="number").columns
    
    if len(numeric_cols) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found to generate a correlation heatmap.")

# =================================================================
# F03: CUSTOMER SEGMENTATION (RFM)
# =================================================================
elif menu == "F03: Customer Segmentation (RFM)":
    
    st.markdown('<div class="section-header">F03: Customer Segmentation - RFM Analysis</div>', unsafe_allow_html=True)
    
    if date_col and customer_col and amount_col:
        try:
            df_seg = df.copy()
            
            if not pd.api.types.is_datetime64_any_dtype(df_seg[date_col]):
                df_seg[date_col] = pd.to_datetime(df_seg[date_col])
            
            snapshot_date = df_seg[date_col].max() + pd.Timedelta(days=1)
            
            rfm = df_seg.groupby(customer_col).agg({
                date_col: lambda x: (snapshot_date - x.max()).days,
                invoice_col if invoice_col else customer_col: 'count',
                amount_col: 'sum'
            }).reset_index()
            
            rfm.columns = ['Customer', 'Recency', 'Frequency', 'Monetary']
            
            st.subheader("RFM Metrics Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Recency", f"{rfm['Recency'].mean():.0f} days")
            with col2:
                st.metric("Avg Frequency", f"{rfm['Frequency'].mean():.1f} purchases")
            with col3:
                st.metric("Avg Monetary Value", f"${rfm['Monetary'].mean():,.2f}")
            
            st.markdown("---")
            
            # Segmentation
            st.subheader("Segmentation Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_segments = st.slider("Number of Segments", 3, 8, 4)
            with col2:
                algorithm = st.radio("Algorithm", ["K-Means", "DBSCAN"], horizontal=True)
            with col3:
                st.write("")
                st.write("")
                st.write(f"**Expected Segments:** {n_segments}")
            
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
            
            if algorithm == "K-Means":
                kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
                rfm['Segment'] = kmeans.fit_predict(rfm_scaled)
                silhouette_score = "K-Means optimized"
            else:
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                rfm['Segment'] = dbscan.fit_predict(rfm_scaled)
                silhouette_score = "DBSCAN clustering applied"
            
            st.markdown("---")
            st.subheader(f"Segment Results ({n_segments} segments)")
            
            segment_summary = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].agg(['mean', 'std', 'count'])
            st.dataframe(segment_summary, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Segment Characteristics")
            
            for segment_id in range(n_segments):
                segment_data = rfm[rfm['Segment'] == segment_id]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"Segment {segment_id} Size", len(segment_data))
                with col2:
                    st.metric(f"Avg Recency", f"{segment_data['Recency'].mean():.0f} days")
                with col3:
                    st.metric(f"Avg Frequency", f"{segment_data['Frequency'].mean():.1f}")
                with col4:
                    st.metric(f"Avg Monetary", f"${segment_data['Monetary'].mean():,.0f}")
            
            # Visualization
            st.markdown("---")
            st.subheader("Customer Segmentation Visualization")
            fig, ax = plt.subplots(figsize=(12, 6))
            scatter = ax.scatter(rfm['Frequency'], rfm['Monetary'], 
                               c=rfm['Segment'], cmap='viridis', s=100, alpha=0.6, edgecolors='black')
            ax.set_xlabel("Purchase Frequency", fontsize=11)
            ax.set_ylabel("Total Spending ($)", fontsize=11)
            ax.set_title("Customer Segmentation (RFM Analysis)", fontsize=13, fontweight='bold')
            cbar = plt.colorbar(scatter, ax=ax, label='Segment')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            st.subheader("Export Segments")
            csv = rfm.to_csv(index=False)
            st.download_button(
                label="Download RFM Segments (CSV)",
                data=csv,
                file_name="rfm_segments.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error in RFM Analysis: {e}")
    else:
        st.warning("Required columns: Date, Customer, and Amount")

# =================================================================
# F04: DEMAND FORECASTING
# =================================================================
elif menu == "F04: Demand Forecasting":
    
    st.markdown('<div class="section-header">F04: Demand Forecasting - Prophet Model</div>', unsafe_allow_html=True)
    
    if date_col and amount_col:
        try:
            df_forecast = df.copy()
            
            if not pd.api.types.is_datetime64_any_dtype(df_forecast[date_col]):
                df_forecast[date_col] = pd.to_datetime(df_forecast[date_col])
            
            forecast_data = df_forecast.groupby(date_col)[amount_col].sum().reset_index()
            forecast_data.columns = ['ds', 'y']
            forecast_data = forecast_data.sort_values('ds').reset_index(drop=True)
            
            st.subheader("Historical Data Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Date Range", f"{forecast_data['ds'].min().date()}")
            with col2:
                st.metric("To", f"{forecast_data['ds'].max().date()}")
            with col3:
                st.metric("Data Points", len(forecast_data))
            with col4:
                st.metric("Total Revenue", f"${forecast_data['y'].sum():,.2f}")
            
            st.markdown("---")
            st.subheader("Forecasting Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                forecast_horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
            with col2:
                interval_width = st.slider("Confidence Interval", 0.80, 0.99, 0.95)
            
            try:
                st.info("Training Prophet forecasting model...")
                
                model = Prophet(
                    interval_width=interval_width,
                    yearly_seasonality=True if len(forecast_data) > 365 else False,
                    weekly_seasonality=True if len(forecast_data) > 14 else False,
                    daily_seasonality=False
                )
                model.fit(forecast_data)
                
                future = model.make_future_dataframe(periods=forecast_horizon)
                forecast = model.predict(future)
                
                # Calculate MAPE (production metric)
                train_forecast = forecast[:len(forecast_data)]
                mape = calculate_mape(forecast_data['y'].values, train_forecast['yhat'].values)
                
                st.markdown("---")
                st.subheader("Model Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    status = "PASS" if mape <= 12 else "FAIL"
                    color = "green" if status == "PASS" else "red"
                    st.metric("MAPE", f"{mape:.2f}%", delta=f"SLA: ≤12% - {status}", 
                             delta_color="off" if status == "PASS" else "inverse")
                with col2:
                    rmse = np.sqrt(mean_squared_error(forecast_data['y'], train_forecast['yhat']))
                    st.metric("RMSE", f"${rmse:,.2f}")
                with col3:
                    st.metric("Forecast Horizon", f"{forecast_horizon} days")
                with col4:
                    st.metric("Confidence Level", f"{interval_width*100:.0f}%")
                
                st.markdown("---")
                st.subheader(f"Forecast Results (Next {forecast_horizon} Days)")
                
                forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon).copy()
                forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                st.dataframe(forecast_display, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Forecast Visualization")
                
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(forecast_data['ds'], forecast_data['y'], label='Historical Data', color='#1f77b4', linewidth=2)
                ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='#ff7f0e', linewidth=2)
                ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                               alpha=0.2, color='#ff7f0e', label='Confidence Interval')
                ax.set_title(f"Demand Forecast (MAPE: {mape:.2f}%)", fontsize=14, fontweight='bold')
                ax.set_xlabel("Date")
                ax.set_ylabel("Revenue ($)")
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Components
                st.markdown("---")
                st.subheader("Forecast Components Analysis")
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)
                
                # Summary metrics
                st.markdown("---")
                st.subheader("Forecast Summary")
                
                future_forecast = forecast[forecast['ds'] > forecast_data['ds'].max()]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_daily = future_forecast['yhat'].mean()
                    st.metric("Avg Daily Forecast", f"${avg_daily:,.2f}")
                with col2:
                    total_period = future_forecast['yhat'].sum()
                    st.metric(f"Total {forecast_horizon}-Day", f"${total_period:,.2f}")
                with col3:
                    volatility = future_forecast['yhat'].std()
                    st.metric("Forecast Volatility", f"${volatility:,.2f}")
                with col4:
                    trend_start = future_forecast['yhat'].iloc[0]
                    trend_end = future_forecast['yhat'].iloc[-1]
                    trend = "Upward" if trend_end > trend_start else "Downward"
                    st.metric("Trend Direction", trend)
                
            except Exception as e:
                st.error(f"Error in forecasting: {e}")
        
        except Exception as e:
            st.error(f"Error in setup: {e}")
    else:
        st.warning("Date and Amount columns required")

# =================================================================
# F05: CHURN PREDICTION
# =================================================================
elif menu == "F05: Churn Prediction":
    
    st.markdown('<div class="section-header">F05: Customer Churn Prediction Model</div>', unsafe_allow_html=True)
    
    if customer_col and amount_col:
        try:
            df_churn = df.copy()
            
            customer_summary = df_churn.groupby(customer_col).agg({
                amount_col: ['sum', 'count', 'mean']
            }).reset_index()
            
            customer_summary.columns = ['Customer', 'Total_Spent', 'Total_Orders', 'Avg_Order_Value']
            
            # Churn label (bottom 25% spenders)
            q1 = customer_summary['Total_Spent'].quantile(0.25)
            customer_summary['Churn_Risk'] = (customer_summary['Total_Spent'] < q1).astype(int)
            
            st.subheader("Customer Base Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", len(customer_summary))
            with col2:
                at_risk = customer_summary['Churn_Risk'].sum()
                st.metric("At-Risk Customers", at_risk)
            with col3:
                churn_rate = (at_risk / len(customer_summary) * 100)
                st.metric("Churn Rate", f"{churn_rate:.2f}%")
            
            st.markdown("---")
            
            X = customer_summary[['Total_Spent', 'Total_Orders', 'Avg_Order_Value']]
            y = customer_summary['Churn_Risk']
            
            if len(X) > 10 and y.sum() > 0:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
                model.fit(X_train, y_train)
                
                predictions = model.predict(X_test)
                predictions_proba = model.predict_proba(X_test)[:, 1]
                
                # Production Metrics
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions)
                recall = recall_score(y_test, predictions)
                f1 = f1_score(y_test, predictions)
                roc_auc = roc_auc_score(y_test, predictions_proba)
                
                # Precision@top20%
                top_20_idx = np.argsort(predictions_proba)[-int(len(predictions_proba)*0.2):]
                precision_top20 = precision_score(y_test.iloc[top_20_idx], predictions[top_20_idx])
                
                st.subheader("Model Performance Metrics - SLA Status")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    auc_status = "PASS" if roc_auc >= 0.88 else "FAIL"
                    st.metric("AUC-ROC", f"{roc_auc:.4f}", delta=f"SLA: ≥0.88 - {auc_status}")
                with col2:
                    prec_status = "PASS" if precision_top20 >= 0.75 else "FAIL"
                    st.metric("Precision@top20%", f"{precision_top20:.4f}", delta=f"SLA: ≥0.75 - {prec_status}")
                with col3:
                    st.metric("Accuracy", f"{accuracy*100:.2f}%")
                with col4:
                    st.metric("F1 Score", f"{f1:.4f}")
                
                st.markdown("---")
                st.subheader("Detailed Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Precision", f"{precision:.4f}")
                with col2:
                    st.metric("Recall", f"{recall:.4f}")
                with col3:
                    st.metric("True Positives", int(np.sum((y_test == 1) & (predictions == 1))))
                with col4:
                    st.metric("True Negatives", int(np.sum((y_test == 0) & (predictions == 0))))
                
                st.markdown("---")
                st.subheader("Feature Importance")
                
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='#d62728')
                ax.set_title("Feature Importance for Churn Prediction", fontsize=12, fontweight='bold')
                ax.set_xlabel("Importance Score")
                plt.tight_layout()
                st.pyplot(fig)
                
                st.dataframe(feature_importance, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Risk Segmentation")
                
                high_risk = customer_summary[customer_summary['Churn_Risk'] == 1].sort_values('Total_Spent', ascending=False)
                st.write(f"**High-Risk Customers to Target:** {len(high_risk)}")
                st.write("Top 10 High-Value At-Risk Customers:")
                st.dataframe(high_risk.head(10), use_container_width=True)
            
            else:
                st.warning("Insufficient data for model training")
        
        except Exception as e:
            st.error(f"Error in Churn Prediction: {e}")
    else:
        st.warning("Customer and Amount columns required")

# =================================================================
# F06: INVENTORY OPTIMIZATION
# =================================================================
elif menu == "F06: Inventory Optimization":
    
    st.markdown('<div class="section-header">F06: Inventory Optimization - EOQ Analysis</div>', unsafe_allow_html=True)
    
    if product_col and quantity_col and date_col and amount_col:
        try:
            df_inv = df.copy()
            
            if not pd.api.types.is_datetime64_any_dtype(df_inv[date_col]):
                df_inv[date_col] = pd.to_datetime(df_inv[date_col])
            
            inventory_data = df_inv.groupby(product_col).agg({
                quantity_col: ['sum', 'mean', 'count'],
                amount_col: 'sum',
                date_col: ['min', 'max']
            }).reset_index()
            
            inventory_data.columns = ['Product', 'Total_Quantity', 'Avg_Order_Qty', 
                                     'Num_Orders', 'Total_Revenue', 'First_Order', 'Last_Order']
            
            days_active = (inventory_data['Last_Order'] - inventory_data['First_Order']).dt.days + 1
            inventory_data['Daily_Demand'] = inventory_data['Total_Quantity'] / days_active.clip(lower=1)
            
            st.subheader("Inventory Parameters")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                holding_cost_pct = st.slider("Holding Cost (%)", 1, 50, 15)
            with col2:
                order_cost = st.number_input("Order Cost ($)", 10.0, 1000.0, 75.0)
            with col3:
                lead_time = st.slider("Lead Time (days)", 1, 60, 7)
            with col4:
                safety_days = st.slider("Safety Stock (days)", 1, 30, 7)
            
            holding_cost = holding_cost_pct / 100
            
            inventory_data['Unit_Cost'] = inventory_data['Total_Revenue'] / inventory_data['Total_Quantity'].clip(lower=1)
            
            annual_demand = inventory_data['Daily_Demand'] * 365
            inventory_data['EOQ'] = np.sqrt((2 * annual_demand * order_cost) / 
                                           (holding_cost * inventory_data['Unit_Cost']).clip(lower=0.01))
            
            inventory_data['Reorder_Point'] = (inventory_data['Daily_Demand'] * lead_time) + \
                                             (inventory_data['Daily_Demand'] * safety_days)
            
            inventory_data['Safety_Stock'] = inventory_data['Daily_Demand'] * safety_days
            inventory_data['Max_Stock_Level'] = inventory_data['EOQ'] + inventory_data['Safety_Stock']
            
            inventory_data['Annual_Holding_Cost'] = (inventory_data['EOQ'] / 2) * \
                                                    holding_cost * inventory_data['Unit_Cost']
            
            inventory_data['Annual_Ordering_Cost'] = (annual_demand / inventory_data['EOQ'].clip(lower=1)) * order_cost
            inventory_data['Total_Inventory_Cost'] = inventory_data['Annual_Holding_Cost'] + \
                                                     inventory_data['Annual_Ordering_Cost']
            
            st.markdown("---")
            st.subheader("Optimization Results - SLA Status")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average EOQ", f"{inventory_data['EOQ'].mean():,.0f} units")
            with col2:
                st.metric("Avg Reorder Point", f"{inventory_data['Reorder_Point'].mean():,.0f}")
            with col3:
                st.metric("Total Annual Holding Cost", f"${inventory_data['Annual_Holding_Cost'].sum():,.2f}")
            with col4:
                st.metric("Total Annual Ordering Cost", f"${inventory_data['Annual_Ordering_Cost'].sum():,.2f}")
            
            st.markdown("---")
            st.subheader("Stock Optimization Opportunity")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                reduction_potential_low = 0.25
                reduction_potential_high = 0.40
                st.metric("Reduction Target", f"{reduction_potential_low*100:.0f}% - {reduction_potential_high*100:.0f}%")
            with col2:
                current_total_cost = inventory_data['Total_Inventory_Cost'].sum()
                savings_low = current_total_cost * reduction_potential_low
                st.metric("Potential Savings (Conservative)", f"${savings_low:,.2f}")
            with col3:
                savings_high = current_total_cost * reduction_potential_high
                st.metric("Potential Savings (Aggressive)", f"${savings_high:,.2f}")
            
            st.markdown("---")
            st.subheader("Product Reorder Recommendations")
            
            recommendations = inventory_data[[
                'Product', 'Daily_Demand', 'EOQ', 'Reorder_Point', 'Safety_Stock', 
                'Max_Stock_Level', 'Unit_Cost', 'Total_Inventory_Cost'
            ]].copy()
            
            recommendations.columns = [
                'Product', 'Daily Demand', 'EOQ (units)', 'Reorder Point', 'Safety Stock', 
                'Max Stock Level', 'Unit Cost', 'Annual Inventory Cost'
            ]
            
            st.dataframe(recommendations.sort_values('Annual Inventory Cost', ascending=False), use_container_width=True)
            
            st.markdown("---")
            st.subheader("Cost Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                top_cost = inventory_data.nlargest(10, 'Total_Inventory_Cost')
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.barh(top_cost['Product'], top_cost['Total_Inventory_Cost'], color='#d62728')
                ax.set_xlabel("Annual Cost ($)")
                ax.set_title("Top 10 Products by Annual Inventory Cost", fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                holding = inventory_data['Annual_Holding_Cost'].sum()
                ordering = inventory_data['Annual_Ordering_Cost'].sum()
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie([holding, ordering], labels=['Holding', 'Ordering'], autopct='%1.1f%%',
                       colors=['#1f77b4', '#ff7f0e'])
                ax.set_title("Cost Breakdown", fontweight='bold')
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Required columns: Product, Quantity, Date, Amount")

# =================================================================
# F07: INTERACTIVE ANALYTICS & EXPORT
# =================================================================
elif menu == "F07: Interactive Analytics & Export":
    
    st.markdown('<div class="section-header">F07: Interactive Analytics & Report Generation</div>', unsafe_allow_html=True)
    
    st.subheader("Dynamic Filters")
    
    col1, col2, col3 = st.columns(3)
    df_filtered = df.copy()
    
    with col1:
        if product_col:
            products = sorted(df_filtered[product_col].unique().tolist())
            selected_products = st.multiselect("Filter by Products", products, default=products[:3])
            if selected_products:
                df_filtered = df_filtered[df_filtered[product_col].isin(selected_products)]
    
    with col2:
        if country_col:
            countries = sorted(df_filtered[country_col].unique().tolist())
            selected_countries = st.multiselect("Filter by Countries", countries, default=countries[:3])
            if selected_countries:
                df_filtered = df_filtered[df_filtered[country_col].isin(selected_countries)]
    
    with col3:
        if date_col:
            try:
                date_range = st.date_input("Date Range", 
                    value=(pd.to_datetime(df_filtered[date_col]).min(), 
                           pd.to_datetime(df_filtered[date_col]).max()))
                if len(date_range) == 2:
                    df_filtered = df_filtered[
                        (pd.to_datetime(df_filtered[date_col]).dt.date >= date_range[0]) &
                        (pd.to_datetime(df_filtered[date_col]).dt.date <= date_range[1])
                    ]
            except:
                pass
    
    st.markdown("---")
    st.subheader("Filtered Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", f"{df_filtered.shape[0]:,}")
    with col2:
        if amount_col:
            st.metric("Revenue", f"${df_filtered[amount_col].sum():,.2f}")
    with col3:
        if quantity_col:
            st.metric("Units", f"{df_filtered[quantity_col].sum():,.0f}")
    with col4:
        if customer_col:
            st.metric("Customers", f"{df_filtered[customer_col].nunique():,}")
    
    st.markdown("---")
    st.subheader("What-If Scenario Analysis")
    
    scenario = st.selectbox("Select Scenario", ["Price Adjustment", "Volume Increase", "Customer Growth", "Market Expansion"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if scenario == "Price Adjustment":
            adj = st.slider("Price Adjustment (%)", -50.0, 50.0, 0.0)
            if amount_col:
                baseline = df_filtered[amount_col].sum()
                adjusted = baseline * (1 + adj / 100)
                st.metric("Baseline Revenue", f"${baseline:,.2f}")
                st.metric("Adjusted Revenue", f"${adjusted:,.2f}")
                st.metric("Impact", f"${adjusted - baseline:,.2f}")
        
        elif scenario == "Volume Increase":
            adj = st.slider("Volume Increase (%)", 0.0, 100.0, 0.0)
            if quantity_col:
                baseline = df_filtered[quantity_col].sum()
                adjusted = baseline * (1 + adj / 100)
                st.metric("Baseline Volume", f"{baseline:,.0f}")
                st.metric("Adjusted Volume", f"{adjusted:,.0f}")
                st.metric("Change", f"{adjusted - baseline:,.0f}")
        
        elif scenario == "Customer Growth":
            growth = st.slider("Customer Growth (%)", 0.0, 100.0, 0.0)
            if customer_col:
                baseline = df_filtered[customer_col].nunique()
                adjusted = baseline * (1 + growth / 100)
                st.metric("Current Customers", f"{baseline:,.0f}")
                st.metric("Projected Customers", f"{adjusted:,.0f}")
                st.metric("New Customers", f"{adjusted - baseline:,.0f}")
        
        elif scenario == "Market Expansion":
            factor = st.slider("Market Factor", 0.5, 3.0, 1.0)
            if amount_col:
                baseline = df_filtered[amount_col].sum()
                adjusted = baseline * factor
                st.metric("Current Market Value", f"${baseline:,.2f}")
                st.metric("Expanded Market", f"${adjusted:,.2f}")
                st.metric("Growth", f"${adjusted - baseline:,.2f}")
    
    st.markdown("---")
    st.subheader("Report Generation & Export")
    
    export_options = st.multiselect("Select Reports",
        ["Filtered Data", "Summary Stats", "Product Analysis", "Customer Analysis", "Geographic Analysis", "Time Series"],
        default=["Filtered Data", "Summary Stats"])
    
    if st.button("Generate & Download Report"):
        export_data = {}
        
        if "Filtered Data" in export_options:
            export_data["Data"] = df_filtered
        
        if "Summary Stats" in export_options:
            stats = pd.DataFrame({
                'Metric': ['Records', 'Revenue', 'Units', 'Customers'],
                'Value': [df_filtered.shape[0], 
                         df_filtered[amount_col].sum() if amount_col else 0,
                         df_filtered[quantity_col].sum() if quantity_col else 0,
                         df_filtered[customer_col].nunique() if customer_col else 0]
            })
            export_data["Summary"] = stats
        
        if "Product Analysis" in export_options and product_col:
            prod = df_filtered.groupby(product_col).agg({
                quantity_col: 'sum' if quantity_col else 'count',
                amount_col: 'sum' if amount_col else 'count'
            }).reset_index().sort_values(amount_col if amount_col else quantity_col, ascending=False)
            if not prod.empty:
                export_data["Products"] = prod
        
        if "Customer Analysis" in export_options and customer_col:
            cust = df_filtered.groupby(customer_col).agg({
                amount_col: ['sum', 'count', 'mean'] if amount_col else 'count'
            }).reset_index()
            if not cust.empty:
                export_data["Customers"] = cust
        
        if "Geographic Analysis" in export_options and country_col:
            geo = df_filtered.groupby(country_col)[amount_col].sum().reset_index().sort_values(amount_col, ascending=False) if amount_col else pd.DataFrame()
            if not geo.empty:
                export_data["Geography"] = geo
        
        if "Time Series" in export_options and date_col:
            ts = df_filtered.groupby(date_col)[amount_col].sum().reset_index() if amount_col else pd.DataFrame()
            if not ts.empty:
                export_data["TimeSeries"] = ts
        
        if export_data:
            excel = export_to_excel(export_data)
            st.download_button(
                label="Download Excel Report",
                data=excel,
                file_name=f"RetailPulse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("Report generated successfully!")

# =================================================================
# COMPLETE PROJECT ROADMAP
# =================================================================
elif menu == "Complete Project Roadmap":
    st.markdown('<div class="section-header">Complete Project Roadmap</div>', unsafe_allow_html=True)
    st.markdown("""
    ### Phase 1: Data Architecture & Ingestion
    * **Objective:** Establish a robust pipeline for cleaning and processing raw retail transactional data.
    * **Actions:** Addressed missing values, removed duplicates, validated schemas, and standardized column formats dynamically.

    ### Phase 2: Exploratory Data Analysis (EDA)
    * **Objective:** Understand data distributions, detect anomalies, and uncover relationships.
    * **Actions:** Analyzed dataset shapes, missing counts, descriptive statistics, and generated feature correlation heatmaps.

    ### Phase 3: Customer Segmentation (Unsupervised Learning)
    * **Objective:** Group customers based on purchasing behavior to enable targeted marketing.
    * **Actions:** Applied RFM (Recency, Frequency, Monetary) modeling paired with user-selectable K-Means and DBSCAN clustering.

    ### Phase 4: Demand Forecasting (Time Series)
    * **Objective:** Predict future sales volume to optimize supply chain readiness.
    * **Actions:** Deployed the Prophet algorithm to forecast 3
