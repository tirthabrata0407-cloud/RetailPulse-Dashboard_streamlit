# pip install streamlit prophet openpyxl scikit-learn matplotlib pandas numpy
# Complete RetailPulse Streamlit Dashboard Code
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="RetailPulse Dashboard",
    layout="wide"
)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("📊 RetailPulse - AI Powered Retail Analytics Dashboard")
st.markdown("---")

# -------------------------------------------------
# HELPER FUNCTIONS FOR COLUMN DETECTION
# -------------------------------------------------

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
    
    # Strategy 1: Look for datetime columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    
    # Strategy 2: Look for object columns with date keywords
    for col in df.columns:
        if df[col].dtype == 'object':
            for keyword in date_keywords:
                if keyword in col.lower():
                    try:
                        pd.to_datetime(df[col].head())
                        return col
                    except:
                        continue
    
    # Strategy 3: Try converting any object column that looks like a date
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col], errors='coerce').notna().sum() > len(df) * 0.8
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

def get_column_suggestions(df, purpose):
    """Provide column suggestions"""
    available_cols = df.columns.tolist()
    st.info(f"📋 Available columns: {', '.join(available_cols)}")

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
@st.cache_data
def load_data():
    try:
        if os.path.exists("merged_cleaned_retail_data.xlsx"):
            df = pd.read_excel("merged_cleaned_retail_data.xlsx")
            return df
        else:
            st.error("❌ Data file 'merged_cleaned_retail_data.xlsx' not found.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
st.sidebar.title("📋 Navigation")
menu = st.sidebar.radio(
    "Select Analysis",
    [
        "Dataset Overview",
        "Data Cleaning",
        "Feature Engineering",
        "EDA",
        "Customer Segmentation",
        "Demand Forecasting",
        "Churn Prediction"
    ]
)

# -------------------------------------------------
# DETECT KEY COLUMNS ONCE
# -------------------------------------------------
date_col = find_date_column(df)
amount_col = find_numeric_column(df, ['total', 'amount', 'revenue', 'sales'])
quantity_col = find_numeric_column(df, ['quantity', 'qty'])
customer_col = find_column_by_keywords(df, ['customer', 'user', 'client'])
product_col = find_column_by_keywords(df, ['description', 'product', 'name', 'item'])
country_col = find_column_by_keywords(df, ['country', 'nation', 'region'])
invoice_col = find_column_by_keywords(df, ['invoice', 'transaction', 'order'])

# =================================================
# 1. DATASET OVERVIEW
# =================================================
if menu == "Dataset Overview":

    st.header("📁 Dataset Overview")

    st.subheader("First 5 Rows")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Shape")
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")

    with col2:
        st.subheader("Data Types")
        st.write(df.dtypes)

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.dataframe(missing[missing > 0])
    else:
        st.success("✅ No missing values!")

    st.subheader("Detected Key Columns")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Date Column:** {date_col if date_col else '❌ Not found'}")
    with col2:
        st.write(f"**Amount Column:** {amount_col if amount_col else '❌ Not found'}")
    with col3:
        st.write(f"**Customer Column:** {customer_col if customer_col else '❌ Not found'}")

# =================================================
# 2. DATA CLEANING
# =================================================
elif menu == "Data Cleaning":

    st.header("🧹 Data Cleaning")

    st.subheader("Original Shape")
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

    # Remove missing values
    df_cleaned = df.dropna()

    # Remove duplicates
    df_cleaned = df_cleaned.drop_duplicates()

    # Remove negative quantity
    if quantity_col and quantity_col in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned[quantity_col] > 0]
        st.info(f"✅ Removed negative quantities from '{quantity_col}'")

    # Remove negative price/amount
    if amount_col and amount_col in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned[amount_col] > 0]
        st.info(f"✅ Removed negative amounts from '{amount_col}'")

    st.subheader("Cleaned Dataset Shape")
    st.write(f"**Rows:** {df_cleaned.shape[0]} | **Columns:** {df_cleaned.shape[1]}")

    st.subheader("Missing Values After Cleaning")
    missing_cleaned = df_cleaned.isnull().sum()
    if missing_cleaned.sum() > 0:
        st.dataframe(missing_cleaned[missing_cleaned > 0])
    else:
        st.success("✅ No missing values after cleaning!")

    st.success("✅ Data Cleaning Completed Successfully")

# =================================================
# 3. FEATURE ENGINEERING
# =================================================
elif menu == "Feature Engineering":

    st.header("⚙️ Feature Engineering")

    df_feat = df.copy()
    
    if date_col:
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df_feat[date_col]):
                df_feat[date_col] = pd.to_datetime(df_feat[date_col])
            
            # Date Features
            df_feat['Year'] = df_feat[date_col].dt.year
            df_feat['Month'] = df_feat[date_col].dt.month
            df_feat['Day'] = df_feat[date_col].dt.day
            df_feat['DayOfWeek'] = df_feat[date_col].dt.day_name()
            df_feat['Quarter'] = df_feat[date_col].dt.quarter

            st.subheader("Date Features Added")
            st.dataframe(df_feat[[date_col, 'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter']].head(), use_container_width=True)
            
            st.success("✅ Date features created successfully!")
        except Exception as e:
            st.error(f"Error processing date column: {e}")
    else:
        st.warning("⚠️ No date column found in the dataset")
        get_column_suggestions(df, "date")

    # Total Revenue
    if amount_col:
        st.subheader("Total Revenue")
        total_revenue = df_feat[amount_col].sum()
        st.metric("Revenue", f"${total_revenue:,.2f}")

    # Quantity Statistics
    if quantity_col:
        st.subheader("Quantity Statistics")
        st.metric("Total Quantity Sold", f"{df_feat[quantity_col].sum():,.0f}")

# =================================================
# 4. EDA
# =================================================
elif menu == "EDA":

    st.header("📈 Exploratory Data Analysis")

    df_eda = df.copy()
    
    if date_col and amount_col:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df_eda[date_col]):
                df_eda[date_col] = pd.to_datetime(df_eda[date_col])
            
            # Monthly Sales Trend
            st.subheader("Monthly Sales Trend")
            df_eda['YearMonth'] = df_eda[date_col].dt.to_period('M')
            monthly_sales = df_eda.groupby('YearMonth')[amount_col].sum()

            fig, ax = plt.subplots(figsize=(12, 5))
            monthly_sales.plot(kind='line', marker='o', ax=ax, linewidth=2, markersize=8)
            ax.set_title(f"Monthly {amount_col} Trend", fontsize=14, fontweight='bold')
            ax.set_xlabel("Month")
            ax.set_ylabel(amount_col)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error in Monthly Sales: {e}")
    else:
        st.warning(f"⚠️ Missing date column: {date_col} or amount column: {amount_col}")

    # Top Products
    if product_col and quantity_col:
        try:
            st.subheader("Top 10 Selling Products")
            top_products = df_eda.groupby(product_col)[quantity_col].sum().sort_values(ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(12, 6))
            top_products.plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title("Top Selling Products", fontsize=14, fontweight='bold')
            ax.set_xlabel("Quantity")
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error in Top Products: {e}")
    else:
        st.warning(f"⚠️ Missing product column: {product_col} or quantity column: {quantity_col}")

    # Country Sales
    if country_col and amount_col:
        try:
            st.subheader("Top Countries by Revenue")
            country_sales = df_eda.groupby(country_col)[amount_col].sum().sort_values(ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(12, 6))
            country_sales.plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_title("Top Countries by Revenue", fontsize=14, fontweight='bold')
            ax.set_xlabel("Country")
            ax.set_ylabel("Revenue")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error in Country Sales: {e}")
    else:
        st.warning(f"⚠️ Missing country column: {country_col} or amount column: {amount_col}")

# =================================================
# 5. CUSTOMER SEGMENTATION
# =================================================
elif menu == "Customer Segmentation":

    st.header("👥 Customer Segmentation (RFM Analysis)")

    if date_col and customer_col and amount_col:
        try:
            df_seg = df.copy()
            
            if not pd.api.types.is_datetime64_any_dtype(df_seg[date_col]):
                df_seg[date_col] = pd.to_datetime(df_seg[date_col])
            
            snapshot_date = df_seg[date_col].max() + pd.Timedelta(days=1)

            # RFM Calculation
            rfm = df_seg.groupby(customer_col).agg({
                date_col: lambda x: (snapshot_date - x.max()).days,
                invoice_col if invoice_col else customer_col: 'count',
                amount_col: 'sum'
            }).reset_index()

            rfm.columns = ['Customer', 'Recency', 'Frequency', 'Monetary']

            st.subheader("RFM Analysis Table")
            st.dataframe(rfm.head(10), use_container_width=True)

            # Scaling
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

            # KMeans
            kmeans = KMeans(n_clusters=4, random_state=42)
            rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

            st.subheader("Cluster Summary")
            st.dataframe(rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean(), use_container_width=True)

            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(rfm['Frequency'], rfm['Monetary'], 
                               c=rfm['Cluster'], cmap='viridis', s=100, alpha=0.6)
            ax.set_title("Customer Segmentation (RFM)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Monetary")
            plt.colorbar(scatter, ax=ax, label='Cluster')
            plt.tight_layout()
            st.pyplot(fig)

            st.success("✅ Customer Segmentation Completed")

        except Exception as e:
            st.error(f"Error in RFM Analysis: {e}")
    else:
        st.warning("⚠️ Missing required columns for RFM analysis")
        st.info(f"Need: Date ({date_col}), Customer ({customer_col}), Amount ({amount_col})")

# =================================================
# 6. DEMAND FORECASTING
# =================================================
elif menu == "Demand Forecasting":

    st.header("📉 Demand Forecasting")

    if date_col and amount_col:
        try:
            df_forecast = df.copy()
            
            if not pd.api.types.is_datetime64_any_dtype(df_forecast[date_col]):
                df_forecast[date_col] = pd.to_datetime(df_forecast[date_col])
            
            forecast_data = df_forecast.groupby(date_col)[amount_col].sum().reset_index()
            forecast_data.columns = ['ds', 'y']
            forecast_data = forecast_data.sort_values('ds').reset_index(drop=True)

            st.subheader("Forecast Data")
            st.dataframe(forecast_data.head(10), use_container_width=True)

            try:
                # Prophet Model
                model = Prophet(interval_width=0.95, yearly_seasonality=True)
                model.fit(forecast_data)
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)

                st.subheader("Forecast Results (Next 30 Days)")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30), use_container_width=True)

                # Forecast Plot
                fig1 = model.plot(forecast)
                plt.title(f"Demand Forecast for {amount_col}", fontsize=14, fontweight='bold')
                st.pyplot(fig1)

                # Components Plot
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)

                st.success("✅ Demand Forecasting Completed")

            except Exception as e:
                st.error(f"Error in Prophet forecasting: {e}")
                st.info("Make sure you have at least 2 data points for forecasting")

        except Exception as e:
            st.error(f"Error in Forecasting Setup: {e}")
    else:
        st.warning(f"⚠️ Missing date column: {date_col} or amount column: {amount_col}")

# =================================================
# 7. CHURN PREDICTION
# =================================================
elif menu == "Churn Prediction":

    st.header("⚠️ Customer Churn Prediction")

    if customer_col and amount_col:
        try:
            df_churn = df.copy()
            
            # Create customer summary
            customer_summary = df_churn.groupby(customer_col).agg({
                amount_col: ['sum', 'count', 'mean']
            }).reset_index()

            customer_summary.columns = ['Customer', 'Total_Spent', 'Total_Orders', 'Avg_Order_Value']

            # Create churn label (customers with low activity = potential churn)
            q1 = customer_summary['Total_Spent'].quantile(0.25)
            customer_summary['Churn_Risk'] = (customer_summary['Total_Spent'] < q1).astype(int)

            st.subheader("Customer Summary")
            st.dataframe(customer_summary.head(10), use_container_width=True)

            # Prepare data for modeling
            X = customer_summary[['Total_Spent', 'Total_Orders', 'Avg_Order_Value']]
            y = customer_summary['Churn_Risk']

            if len(X) > 10 and y.sum() > 0:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
                with col2:
                    st.metric("Total Customers", len(customer_summary))
                with col3:
                    st.metric("At-Risk Customers", y.sum())

                # Feature Importance
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
                ax.set_title("Feature Importance for Churn Prediction", fontsize=14, fontweight='bold')
                ax.set_xlabel("Importance")
                plt.tight_layout()
                st.pyplot(fig)

                # Churn Distribution
                st.subheader("Churn Risk Distribution")
                churn_dist = customer_summary['Churn_Risk'].value_counts()

                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#2ecc71', '#e74c3c']
                churn_dist.plot(kind='bar', ax=ax, color=colors)
                ax.set_title("Customer Churn Risk Distribution", fontsize=14, fontweight='bold')
                ax.set_xlabel("Churn Risk (0=Low, 1=High)")
                ax.set_ylabel("Number of Customers")
                ax.set_xticklabels(['Low Risk', 'High Risk'], rotation=0)
                plt.tight_layout()
                st.pyplot(fig)

                st.success("✅ Churn Prediction Analysis Completed")

            else:
                st.warning("⚠️ Insufficient data for churn prediction model")

        except Exception as e:
            st.error(f"Error in Churn Prediction: {e}")
            st.info("Make sure you have customer and amount columns")
    else:
        st.warning(f"⚠️ Missing customer column: {customer_col} or amount column: {amount_col}")

st.markdown("---")
st.markdown("**📊 RetailPulse Dashboard** | Powered by Streamlit & Machine Learning")
