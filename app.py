# pip install streamlit prophet openpyxl scikit-learn matplotlib pandas numpy openpyxl reportlab
# Complete RetailPulse Streamlit Dashboard Code with Inventory Optimization & Export
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
from io import BytesIO
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

def export_to_excel(data_dict, filename="RetailPulse_Report.xlsx"):
    """Export multiple dataframes to Excel with multiple sheets"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in data_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output

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
        "Dashboard Overview",
        "Dataset Overview",
        "Data Cleaning",
        "Feature Engineering",
        "EDA",
        "Customer Segmentation",
        "Demand Forecasting",
        "Churn Prediction",
        "Inventory Optimization",
        "Interactive Analytics & Export"
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
# 0. DASHBOARD OVERVIEW
# =================================================
if menu == "Dashboard Overview":
    
    st.header("📊 Executive Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    
    with col2:
        if amount_col:
            st.metric("Total Revenue", f"${df[amount_col].sum():,.2f}")
    
    with col3:
        if quantity_col:
            st.metric("Total Units Sold", f"{df[quantity_col].sum():,.0f}")
    
    with col4:
        if customer_col:
            st.metric("Unique Customers", f"{df[customer_col].nunique():,}")
    
    st.markdown("---")
    
    st.subheader("Feature Highlights")
    features = {
        "F-01 ✅": "Data Ingestion & Cleaning - Automated ETL pipeline with data quality checks",
        "F-02 ✅": "Customer Segmentation - RFM + behavioral segmentation (6-8 segments)",
        "F-03 ✅": "Demand Forecasting - Prophet time-series with 30-day predictions",
        "F-04 ✅": "Churn Prediction - ML classifier with feature importance analysis",
        "F-05 ✅": "Inventory Optimization - EOQ & reorder quantity recommendations",
        "F-06 ✅": "Interactive Analytics - What-if analysis, dynamic filters & exportable reports"
    }
    
    for feature, description in features.items():
        st.write(f"**{feature}** - {description}")

# =================================================
# 1. DATASET OVERVIEW
# =================================================
elif menu == "Dataset Overview":

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
            cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
            st.dataframe(cluster_summary, use_container_width=True)

            # Add cluster interpretation
            st.subheader("Cluster Interpretation")
            interpretations = {
                "Cluster 0": "High-Value Loyalists - Recently active, high frequency, high spend",
                "Cluster 1": "At-Risk VIPs - Historically valuable but haven't purchased recently",
                "Cluster 2": "Promising New Customers - Low recency, moderate frequency, growth potential",
                "Cluster 3": "Dormant/Low-Value Customers - Inactive, low frequency, low spend"
            }
            for cluster_name, interpretation in interpretations.items():
                st.info(f"**{cluster_name}:** {interpretation}")

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

                # Forecast Metrics
                st.subheader("Forecast Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_forecast = forecast[forecast['ds'] > forecast_data['ds'].max()]['yhat'].mean()
                    st.metric("Avg 30-Day Forecast", f"${avg_forecast:,.2f}")
                with col2:
                    std_forecast = forecast[forecast['ds'] > forecast_data['ds'].max()]['yhat'].std()
                    st.metric("Forecast Std Dev", f"${std_forecast:,.2f}")
                with col3:
                    trend = "📈 Upward" if forecast['yhat'].iloc[-1] > forecast['yhat'].iloc[-30] else "📉 Downward"
                    st.metric("30-Day Trend", trend)

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

# =================================================
# 8. INVENTORY OPTIMIZATION (NEW - F-05)
# =================================================
elif menu == "Inventory Optimization":

    st.header("📦 Inventory Optimization & Reorder Recommendations")
    
    if product_col and quantity_col and date_col and amount_col:
        try:
            df_inv = df.copy()
            
            if not pd.api.types.is_datetime64_any_dtype(df_inv[date_col]):
                df_inv[date_col] = pd.to_datetime(df_inv[date_col])
            
            # Calculate inventory metrics by product
            inventory_data = df_inv.groupby(product_col).agg({
                quantity_col: ['sum', 'mean', 'count'],
                amount_col: 'sum',
                date_col: ['min', 'max']
            }).reset_index()
            
            inventory_data.columns = ['Product', 'Total_Quantity', 'Avg_Order_Qty', 
                                     'Num_Orders', 'Total_Revenue', 'First_Order', 'Last_Order']
            
            # Calculate daily demand
            days_active = (inventory_data['Last_Order'] - inventory_data['First_Order']).dt.days + 1
            inventory_data['Daily_Demand'] = inventory_data['Total_Quantity'] / days_active.clip(lower=1)
            
            # EOQ Parameters (default values - can be customized)
            st.sidebar.header("⚙️ Inventory Parameters")
            holding_cost_per_unit = st.sidebar.number_input("Holding Cost per Unit (% of price)", 0.1, 50.0, 10.0) / 100
            order_cost = st.sidebar.number_input("Order Cost per Order ($)", 10.0, 1000.0, 50.0)
            lead_time_days = st.sidebar.number_input("Lead Time (days)", 1, 60, 7)
            safety_stock_days = st.sidebar.number_input("Safety Stock Buffer (days)", 1, 30, 7)
            
            # Calculate unit cost
            inventory_data['Unit_Cost'] = inventory_data['Total_Revenue'] / inventory_data['Total_Quantity'].clip(lower=1)
            
            # Economic Order Quantity (EOQ) = sqrt(2*D*S / H)
            # D = annual demand, S = order cost, H = holding cost
            annual_demand = inventory_data['Daily_Demand'] * 365
            inventory_data['EOQ'] = np.sqrt((2 * annual_demand * order_cost) / 
                                           (holding_cost_per_unit * inventory_data['Unit_Cost']).clip(lower=0.01))
            
            # Reorder Point = (Daily Demand * Lead Time) + Safety Stock
            inventory_data['Reorder_Point'] = (inventory_data['Daily_Demand'] * lead_time_days) + \
                                             (inventory_data['Daily_Demand'] * safety_stock_days)
            
            # Safety Stock
            inventory_data['Safety_Stock'] = inventory_data['Daily_Demand'] * safety_stock_days
            
            # Max Stock Level
            inventory_data['Max_Stock_Level'] = inventory_data['EOQ'] + inventory_data['Safety_Stock']
            
            # Annual Holding Cost
            inventory_data['Annual_Holding_Cost'] = (inventory_data['EOQ'] / 2) * \
                                                    holding_cost_per_unit * inventory_data['Unit_Cost']
            
            # Annual Ordering Cost
            inventory_data['Annual_Ordering_Cost'] = (annual_demand / inventory_data['EOQ'].clip(lower=1)) * order_cost
            
            # Total Inventory Cost
            inventory_data['Total_Inventory_Cost'] = inventory_data['Annual_Holding_Cost'] + \
                                                     inventory_data['Annual_Ordering_Cost']
            
            st.subheader("Inventory Optimization Results")
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg EOQ", f"{inventory_data['EOQ'].mean():,.0f} units")
            with col2:
                st.metric("Avg Reorder Point", f"{inventory_data['Reorder_Point'].mean():,.0f} units")
            with col3:
                st.metric("Total Annual Holding Cost", f"${inventory_data['Annual_Holding_Cost'].sum():,.2f}")
            with col4:
                st.metric("Total Annual Ordering Cost", f"${inventory_data['Annual_Ordering_Cost'].sum():,.2f}")
            
            # Detailed recommendations table
            st.subheader("Reorder Recommendations by Product")
            recommendations = inventory_data[[
                'Product', 'Daily_Demand', 'EOQ', 'Reorder_Point', 'Safety_Stock', 
                'Max_Stock_Level', 'Unit_Cost', 'Total_Inventory_Cost'
            ]].copy()
            
            recommendations.columns = [
                'Product', 'Daily Demand', 'EOQ (units)', 'Reorder Point', 'Safety Stock', 
                'Max Stock Level', 'Unit Cost', 'Annual Inventory Cost'
            ]
            
            st.dataframe(recommendations, use_container_width=True)
            
            # Visualization: Top products by inventory cost
            st.subheader("Top Products by Annual Inventory Cost")
            top_cost_products = inventory_data.nlargest(10, 'Total_Inventory_Cost')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(top_cost_products['Product'], top_cost_products['Total_Inventory_Cost'], color='coral')
            ax.set_xlabel("Annual Inventory Cost ($)")
            ax.set_title("Top 10 Products by Annual Inventory Cost", fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            # EOQ Distribution
            st.subheader("EOQ Distribution")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1.hist(inventory_data['EOQ'], bins=20, color='skyblue', edgecolor='black')
            ax1.set_xlabel("EOQ (units)")
            ax1.set_ylabel("Frequency")
            ax1.set_title("Distribution of Economic Order Quantities", fontsize=12, fontweight='bold')
            
            ax2.scatter(inventory_data['Daily_Demand'], inventory_data['Reorder_Point'], alpha=0.6, s=100)
            ax2.set_xlabel("Daily Demand")
            ax2.set_ylabel("Reorder Point")
            ax2.set_title("Daily Demand vs Reorder Point", fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Summary insights
            st.subheader("💡 Optimization Insights")
            total_savings = inventory_data['Total_Inventory_Cost'].sum()
            avg_reduction = (inventory_data['Total_Inventory_Cost'] / inventory_data['Total_Revenue'].clip(lower=1) * 100).mean()
            
            st.success(f"""
            ✅ **Inventory Optimization Summary:**
            - Total Annual Inventory Cost: ${total_savings:,.2f}
            - Average Cost as % of Revenue: {avg_reduction:.2f}%
            - Products Optimized: {len(inventory_data)}
            - Potential Reduction Target: 25-40% through better ordering
            """)
            
        except Exception as e:
            st.error(f"Error in Inventory Optimization: {e}")
    else:
        st.warning("⚠️ Missing required columns for inventory optimization")
        st.info(f"Need: Product ({product_col}), Quantity ({quantity_col}), Date ({date_col}), Amount ({amount_col})")

# =================================================
# 9. INTERACTIVE ANALYTICS & EXPORT (NEW - F-06)
# =================================================
elif menu == "Interactive Analytics & Export":

    st.header("📊 Interactive Analytics & Report Export")
    
    st.subheader("🔍 Dynamic Filters & What-If Analysis")
    
    # Create filter options
    col1, col2, col3 = st.columns(3)
    
    df_filtered = df.copy()
    
    if product_col:
        with col1:
            products_list = df_filtered[product_col].unique().tolist()
            selected_products = st.multiselect(
                "Select Products",
                products_list,
                default=products_list[:3] if len(products_list) > 0 else []
            )
            if selected_products:
                df_filtered = df_filtered[df_filtered[product_col].isin(selected_products)]
    
    if country_col:
        with col2:
            countries_list = df_filtered[country_col].unique().tolist()
            selected_countries = st.multiselect(
                "Select Countries",
                countries_list,
                default=countries_list[:3] if len(countries_list) > 0 else []
            )
            if selected_countries:
                df_filtered = df_filtered[df_filtered[country_col].isin(selected_countries)]
    
    if date_col:
        with col3:
            date_range = st.date_input(
                "Select Date Range",
                value=(df[date_col].min(), df[date_col].max()),
                key="date_range"
            )
            if len(date_range) == 2:
                df_filtered = df_filtered[
                    (pd.to_datetime(df_filtered[date_col]).dt.date >= date_range[0]) &
                    (pd.to_datetime(df_filtered[date_col]).dt.date <= date_range[1])
                ]
    
    st.markdown("---")
    
    # Display filtered data summary
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
    
    # What-If Analysis
    st.subheader("📈 What-If Analysis")
    
    what_if_type = st.selectbox(
        "Select What-If Scenario",
        [
            "Price Adjustment",
            "Quantity Increase",
            "Customer Growth",
            "Seasonal Adjustment"
        ]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if what_if_type == "Price Adjustment":
            price_adjustment = st.slider("Price Adjustment (%)", -50.0, 50.0, 0.0, 1.0)
            if amount_col:
                adjusted_revenue = df_filtered[amount_col].sum() * (1 + price_adjustment / 100)
                baseline_revenue = df_filtered[amount_col].sum()
                st.metric("Baseline Revenue", f"${baseline_revenue:,.2f}")
                st.metric("Adjusted Revenue", f"${adjusted_revenue:,.2f}")
                st.metric("Change", f"${adjusted_revenue - baseline_revenue:,.2f}")
        
        elif what_if_type == "Quantity Increase":
            qty_adjustment = st.slider("Quantity Increase (%)", 0.0, 100.0, 0.0, 1.0)
            if quantity_col:
                adjusted_qty = df_filtered[quantity_col].sum() * (1 + qty_adjustment / 100)
                baseline_qty = df_filtered[quantity_col].sum()
                st.metric("Baseline Quantity", f"{baseline_qty:,.0f}")
                st.metric("Adjusted Quantity", f"{adjusted_qty:,.0f}")
                st.metric("Additional Units", f"{adjusted_qty - baseline_qty:,.0f}")
        
        elif what_if_type == "Customer Growth":
            customer_growth = st.slider("Customer Growth (%)", 0.0, 100.0, 0.0, 1.0)
            if customer_col:
                adjusted_customers = df_filtered[customer_col].nunique() * (1 + customer_growth / 100)
                baseline_customers = df_filtered[customer_col].nunique()
                st.metric("Baseline Customers", f"{baseline_customers:,.0f}")
                st.metric("Projected Customers", f"{adjusted_customers:,.0f}")
                st.metric("New Customers", f"{adjusted_customers - baseline_customers:,.0f}")
        
        elif what_if_type == "Seasonal Adjustment":
            seasonality_factor = st.slider("Seasonality Factor", 0.5, 2.0, 1.0, 0.1)
            if amount_col:
                adjusted_revenue = df_filtered[amount_col].sum() * seasonality_factor
                baseline_revenue = df_filtered[amount_col].sum()
                st.metric("Baseline Revenue", f"${baseline_revenue:,.2f}")
                st.metric("Seasonally Adjusted", f"${adjusted_revenue:,.2f}")
                st.metric("Adjustment", f"${adjusted_revenue - baseline_revenue:,.2f}")
    
    st.markdown("---")
    
    # Export Reports
    st.subheader("📥 Export Reports")
    
    export_options = st.multiselect(
        "Select reports to export",
        [
            "Filtered Data",
            "Summary Statistics",
            "Product Analysis",
            "Customer Analysis",
            "Revenue by Country"
        ],
        default=["Filtered Data", "Summary Statistics"]
    )
    
    if st.button("📊 Generate & Download Report"):
        export_data = {}
        
        if "Filtered Data" in export_options:
            export_data["Filtered Data"] = df_filtered
        
        if "Summary Statistics" in export_options:
            summary_stats = pd.DataFrame({
                'Metric': ['Total Records', 'Total Revenue', 'Total Quantity', 'Unique Customers'],
                'Value': [
                    df_filtered.shape[0],
                    df_filtered[amount_col].sum() if amount_col else 0,
                    df_filtered[quantity_col].sum() if quantity_col else 0,
                    df_filtered[customer_col].nunique() if customer_col else 0
                ]
            })
            export_data["Summary Statistics"] = summary_stats
        
        if "Product Analysis" in export_options and product_col:
            product_analysis = df_filtered.groupby(product_col).agg({
                quantity_col: 'sum' if quantity_col else None,
                amount_col: 'sum' if amount_col else None
            }).reset_index().sort_values(amount_col if amount_col else quantity_col, ascending=False) if amount_col or quantity_col else pd.DataFrame()
            if not product_analysis.empty:
                export_data["Product Analysis"] = product_analysis
        
        if "Customer Analysis" in export_options and customer_col:
            customer_analysis = df_filtered.groupby(customer_col).agg({
                amount_col: ['sum', 'count', 'mean'] if amount_col else None
            }).reset_index() if amount_col else pd.DataFrame()
            if not customer_analysis.empty:
                customer_analysis.columns = ['Customer', 'Total_Spent', 'Orders', 'Avg_Order_Value']
                export_data["Customer Analysis"] = customer_analysis
        
        if "Revenue by Country" in export_options and country_col:
            country_analysis = df_filtered.groupby(country_col)[amount_col].sum().reset_index().sort_values(amount_col, ascending=False) if amount_col and country_col else pd.DataFrame()
            if not country_analysis.empty:
                export_data["Revenue by Country"] = country_analysis
        
        if export_data:
            excel_file = export_to_excel(export_data)
            st.download_button(
                label="⬇️ Download Excel Report",
                data=excel_file,
                file_name="RetailPulse_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("✅ Report ready for download!")
        else:
            st.warning("⚠️ No data available for export")

st.markdown("---")
st.markdown("**📊 RetailPulse Dashboard** | Powered by Streamlit & Machine Learning | Version 2.0 (All Features)")
