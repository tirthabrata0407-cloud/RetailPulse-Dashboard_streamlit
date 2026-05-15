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
# LOAD DATASET
# -------------------------------------------------
@st.cache_data

def load_data():
    df = pd.read_excel("merged_cleaned_retail_data.xlsx")
    return df


df = load_data()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
menu = st.sidebar.selectbox(
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

# =================================================
# 1. DATASET OVERVIEW
# =================================================
if menu == "Dataset Overview":

    st.header("📁 Dataset Overview")

    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Column Names")
    st.write(df.columns)

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# =================================================
# 2. DATA CLEANING
# =================================================
elif menu == "Data Cleaning":

    st.header("🧹 Data Cleaning")

    st.subheader("Original Shape")
    st.write(df.shape)

    # Remove missing values
    df_cleaned = df.dropna()

    # Remove duplicates
    df_cleaned = df_cleaned.drop_duplicates()

    # Remove negative quantity
    df_cleaned = df_cleaned[df_cleaned['Quantity'] > 0]

    # Remove negative price
    df_cleaned = df_cleaned[df_cleaned['Price'] > 0]

    st.subheader("Cleaned Dataset Shape")
    st.write(df_cleaned.shape)

    st.subheader("Missing Values After Cleaning")
    st.write(df_cleaned.isnull().sum())

    st.success("Data Cleaning Completed Successfully")

# =================================================
# 3. FEATURE ENGINEERING
# =================================================
elif menu == "Feature Engineering":

    st.header("⚙️ Feature Engineering")

    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])

    # Date Features
    df['Year'] = df['Invoice Date'].dt.year
    df['Month'] = df['Invoice Date'].dt.month
    df['Day'] = df['Invoice Date'].dt.day
    df['Weekday'] = df['Invoice Date'].dt.day_name()

    st.subheader("Date Features Added")

    st.dataframe(
        df[['Invoice Date', 'Year', 'Month', 'Day', 'Weekday']].head()
    )

    st.subheader("Total Revenue")
    st.metric("Revenue", round(df['Total_Amount'].sum(), 2))

    st.subheader("Total Profit")
    st.metric("Profit", round(df['Profit'].sum(), 2))

# =================================================
# 4. EDA
# =================================================
elif menu == "EDA":

    st.header("📈 Exploratory Data Analysis")

    # ---------------------------------------------
    # Monthly Sales
    # ---------------------------------------------
    st.subheader("Monthly Sales Trend")

    monthly_sales = df.groupby('Month')['Total_Amount'].sum()

    fig, ax = plt.subplots(figsize=(10,5))

    monthly_sales.plot(kind='line', marker='o', ax=ax)

    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")

    st.pyplot(fig)

    # ---------------------------------------------
    # Top Products
    # ---------------------------------------------
    st.subheader("Top 10 Selling Products")

    top_products = df.groupby('Description')['Quantity'].sum() \
                     .sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12,6))

    top_products.plot(kind='bar', ax=ax)

    ax.set_title("Top Selling Products")
    ax.set_xlabel("Products")
    ax.set_ylabel("Quantity")

    plt.xticks(rotation=90)

    st.pyplot(fig)

    # ---------------------------------------------
    # Country Sales
    # ---------------------------------------------
    st.subheader("Top Countries by Revenue")

    country_sales = df.groupby('Country')['Total_Amount'].sum() \
                      .sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10,5))

    country_sales.plot(kind='bar', ax=ax)

    ax.set_title("Top Countries")
    ax.set_xlabel("Country")
    ax.set_ylabel("Revenue")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # ---------------------------------------------
    # Customer Type Analysis
    # ---------------------------------------------
    st.subheader("Customer Type Revenue")

    customer_type = df.groupby('Customer_Type')['Total_Amount'].sum()

    fig, ax = plt.subplots()

    customer_type.plot(kind='pie', autopct='%1.1f%%', ax=ax)

    ax.set_ylabel("")

    st.pyplot(fig)

# =================================================
# 5. CUSTOMER SEGMENTATION
# =================================================
elif menu == "Customer Segmentation":

    st.header("👥 Customer Segmentation (RFM Analysis)")

    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])

    snapshot_date = df['Invoice Date'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('Customer ID').agg({

        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,

        'Invoice': 'nunique',

        'Total_Amount': 'sum'

    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    st.subheader("RFM Table")
    st.dataframe(rfm.head())

    # Scaling
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)

    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    st.subheader("Cluster Summary")
    st.write(rfm.groupby('Cluster').mean())

    # Visualization
    fig, ax = plt.subplots(figsize=(10,6))

    scatter = ax.scatter(
        rfm['Frequency'],
        rfm['Monetary'],
        c=rfm['Cluster']
    )

    ax.set_title("Customer Segmentation")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Monetary")

    st.pyplot(fig)

# =================================================
# 6. DEMAND FORECASTING
# =================================================
elif menu == "Demand Forecasting":

    st.header("📉 Demand Forecasting")

    forecast_data = df.groupby('Invoice Date')['Total_Amount'].sum().reset_index()

    forecast_data.columns = ['ds', 'y']

    st.subheader("Forecast Data")
    st.dataframe(forecast_data.head())

    # Prophet Model
    model = Prophet()

    model.fit(forecast_data)

    future = model.make_future_dataframe(periods=30)

    forecast = model.predict(future)

    st.subheader("Forecast Results")

    st.dataframe(forecast[['ds', 'yhat']].tail())

    # Forecast Plot
    fig1 = model.plot(forecast)

    st.pyplot(fig1)

# =================================================
# 7. CHURN PREDICTION
# =================================================
elif menu == "Churn Prediction":

    st.header("⚠️ Churn Prediction")

    customer_data = df.groupby('Customer ID').agg({

        'Total_Amount': 'sum',

        'Invoice': 'nunique',

        'Churn': 'max'

    }).reset_index()

    customer_data.columns = [
        'Customer_ID',
        'Total_Spent',
        'Total_Orders',
        'Churn'
    ]

    X = customer_data[['Total_Spent', 'Total_Orders']]

    y = customer_data['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    st.subheader("Model Accuracy")

    st.metric("Accuracy", round(accuracy * 100, 2))

    # Churn Distribution
    st.subheader("Churn Distribution")

    churn_counts = customer_data['Churn'].value_counts()

    fig, ax = plt.subplots()

    churn_counts.plot(kind='bar', ax=ax)

    ax.set_title("Customer Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    st.success("Churn Prediction Completed")
