# RetailPulse – AI Customer Analytics & Demand Forecasting

🚀 **Project Overview**

RetailPulse is an AI-powered retail analytics platform developed using Python, Streamlit, and Machine Learning techniques. The project helps businesses analyze customer behavior, monitor sales performance, identify churn risks, gain inventory insights, and forecast future sales through an interactive dashboard.

The platform combines Data Analytics, Machine Learning, Visualization, and Deployment concepts into a single end-to-end retail intelligence solution.

---

## 🌐 Live Deployment

🔗 **Streamlit App**
```
[ADD YOUR STREAMLIT APP LINK HERE]
https://your-app-name.streamlit.app/
```

🔗 **GitHub Repository**
```
https://github.com/tirthabrata0407-cloud/streamlit_template
```

---

## ✨ Features

✅ **Exploratory Data Analysis (EDA)** - Statistical summaries, distributions, and correlation tracking

✅ **Sales Performance Dashboard** - Real-time revenue and units monitoring

✅ **Customer Segmentation (RFM)** - RFM + K-Means/DBSCAN with 3-8 segments

✅ **Demand Forecasting** - Prophet time-series forecasting with confidence intervals

✅ **Churn Prediction** - Random Forest classifier with explainability

✅ **Inventory Optimization** - EOQ-based reorder recommendations

✅ **Interactive Analytics & Export** - Dynamic filters, what-if scenarios, and exportable reports

✅ **Data Ingestion & Cleaning** - Automated ETL pipeline with data quality checks

✅ **Production-Grade SLA Monitoring** - Enterprise-ready metrics tracking

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Programming** | Python 3.9+ |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib |
| **Machine Learning** | Scikit-learn, Prophet |
| **Dashboard** | Streamlit |
| **Deployment** | Streamlit Cloud, Docker |
| **CI/CD** | GitHub Actions |
| **Data Format** | Excel (openpyxl) |

---

## 📁 Project Structure

```
streamlit_template/
│
├── app.py                          # Main Streamlit application
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies (pinned versions)
├── Dockerfile                       # Docker containerization
├── .dockerignore                    # Docker build optimization
│
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI/CD pipeline
│
├── merged_cleaned_retail_data.xlsx  # Sample retail dataset (15,000 records)
│
└── outputs/                         # Generated reports and exports
```

---

## 📊 Dashboard Modules (7 Production-Grade Features)

### **F01: Data Ingestion & Cleaning**
- Automated ETL pipeline with data quality checks
- Data Quality Score calculation
- Missing value & duplicate detection
- Schema validation
- Data type detection

### **F02: Exploratory Data Analysis (EDA)**
- Dataset preview and shape analysis
- Missing value analysis
- Summary statistics
- Daily sales trends visualization
- Correlation heatmap analysis

### **F03: Customer Segmentation (RFM)**
- RFM (Recency, Frequency, Monetary) modeling
- K-Means and DBSCAN clustering
- 3-8 configurable segments
- Segment characteristics analysis
- Customer segmentation visualization

### **F04: Demand Forecasting**
- Prophet time-series forecasting
- 7-90 day forecast horizons
- MAPE ≤ 12% SLA monitoring
- Confidence interval visualization
- Forecast components analysis

### **F05: Churn Prediction**
- Random Forest classifier
- AUC-ROC ≥ 0.88 SLA tracking
- Precision@top20% ≥ 0.75 metric
- Feature importance analysis
- High-value at-risk customer identification

### **F06: Inventory Optimization**
- Economic Order Quantity (EOQ) calculations
- 25-40% stock reduction potential
- Reorder point recommendations
- Safety stock analysis
- Annual cost optimization

### **F07: Interactive Analytics & Export**
- Dynamic multi-filter capabilities
- What-if scenario analysis
- Real-time report generation
- Excel export functionality
- Comprehensive analytics dashboards

---

## 📦 Installation

### Clone Repository
```bash
git clone https://github.com/tirthabrata0407-cloud/streamlit_template.git
cd streamlit_template
```

### Install Requirements
```bash
pip install -r requirements.txt
```

### Run Streamlit Application
```bash
streamlit run app.py
```

The application will be available at: `http://localhost:8501`

---

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t retailpulse:latest .
```

### Run Docker Container
```bash
docker run -p 8501:8501 retailpulse:latest
```

### Verify Container Health
```bash
docker ps
curl http://localhost:8501/_stcore/health
```

### Docker Compose
Create a `docker-compose.yml` file:
```yaml
version: '3.8'
services:
  retailpulse:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./merged_cleaned_retail_data.xlsx:/app/merged_cleaned_retail_data.xlsx
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

Then run:
```bash
docker-compose up
```

---

## ⚙️ CI/CD Pipeline with GitHub Actions

The project includes automated CI/CD pipeline (`.github/workflows/ci.yml`) that:

✅ **Validates Python syntax** - Checks for syntax errors
✅ **Verifies imports** - Ensures all dependencies are available
✅ **Runs linting** - Code quality checks with flake8
✅ **Builds Docker image** - Creates container image
✅ **Tests Docker image** - Validates image works correctly
✅ **Checks project structure** - Verifies all required files exist
✅ **Triggers automatically** - On every push to main/develop branches

### View Pipeline Status
```
Go to: https://github.com/tirthabrata0407-cloud/streamlit_template/actions
```

### Pipeline Triggers
The CI/CD pipeline runs automatically on:
- ✅ Push to `main` branch
- ✅ Push to `develop` branch
- ✅ Pull requests to `main` or `develop`

---

## 🎯 Key Metrics & SLA Monitoring

| Feature | SLA Target | Status |
|---------|-----------|--------|
| **Data Quality Score** | > 85% | Production |
| **Forecast Accuracy (MAPE)** | ≤ 12% | Production |
| **Churn Model (AUC-ROC)** | ≥ 0.88 | Production |
| **Churn Precision@20%** | ≥ 0.75 | Production |
| **Inventory Reduction** | 25-40% | Production |
| **Data Ingestion** | < 5 seconds | Production |
| **Docker Build Time** | < 2 minutes | Production |

---

## 🧮 Machine Learning Models

### **1. RFM Segmentation**
- **Algorithm:** K-Means & DBSCAN clustering
- **Features:** Recency, Frequency, Monetary value
- **Output:** Customer segments with actionable insights

### **2. Demand Forecasting**
- **Algorithm:** Facebook Prophet
- **Scope:** 30-day default, configurable up to 90 days
- **Metrics:** MAPE, RMSE, Confidence Intervals

### **3. Churn Prediction**
- **Algorithm:** Random Forest Classifier
- **Features:** Total Spent, Total Orders, Average Order Value
- **Output:** Churn risk scores and feature importance

### **4. Inventory Optimization**
- **Method:** Economic Order Quantity (EOQ)
- **Calculations:** Reorder points, safety stock, holding costs
- **Output:** Optimization recommendations

---

## 💡 Usage Guide

### **Dashboard Navigation**

1. **Dashboard Overview** - High-level metrics and feature status
2. **Data Ingestion & Cleaning** - View data quality metrics
3. **EDA** - Explore data distributions and trends
4. **Customer Segmentation** - Analyze customer clusters
5. **Demand Forecasting** - View sales predictions
6. **Churn Prediction** - Identify at-risk customers
7. **Inventory Optimization** - Get reorder recommendations
8. **Interactive Analytics** - Dynamic filters and what-if analysis
9. **Project Roadmap** - Implementation phases
10. **Project Summary** - Executive overview

---

## 📈 Data Requirements

The application expects an Excel file (`merged_cleaned_retail_data.xlsx`) with the following structure:

- **Date Column:** Transaction date (keywords: "Date", "InvoiceDate", "OrderDate")
- **Amount Column:** Revenue/Sales (keywords: "Total", "Amount", "Revenue", "Sales")
- **Quantity Column:** Order quantity (keywords: "Quantity", "Qty")
- **Customer Column:** Customer identifier (keywords: "Customer", "User", "Client")
- **Product Column:** Product description (keywords: "Description", "Product", "Name", "Item")
- **Country Column:** Geographic location (keywords: "Country", "Nation", "Region")
- **Invoice Column:** Transaction ID (keywords: "Invoice", "Transaction", "Order")

---

## 🎓 Key Learnings

✅ Data Cleaning & EDA best practices

✅ Machine Learning workflow implementation

✅ Interactive Dashboard development with Streamlit

✅ Time-series forecasting with Prophet

✅ Classification models and explainability

✅ Prescriptive analytics (EOQ optimization)

✅ Production-level SLA monitoring

✅ Real-time reporting and exports

✅ Docker containerization for deployment

✅ CI/CD pipeline automation with GitHub Actions

✅ Infrastructure as Code principles

---

## 🚧 Challenges Faced & Solutions

| Challenge | Solution |
|-----------|----------|
| Handling large retail datasets | Optimized data loading with caching |
| Dynamic column detection | Multi-keyword matching strategy |
| Memory optimization | Efficient data filtering and aggregation |
| Forecasting accuracy | Prophet with seasonal decomposition |
| Model interpretability | Feature importance visualization |
| Report generation | Automated Excel export with multiple sheets |
| Deployment compatibility | Docker containerization with health checks |
| Build automation | GitHub Actions CI/CD pipeline |
| Environment consistency | Version pinning in requirements.txt |

---

## 🔮 Future Enhancements

🚀 Real-time data ingestion from databases

🚀 Advanced forecasting models (LSTM, XGBoost)

🚀 Cloud database integration (PostgreSQL, MongoDB)

🚀 User authentication system

🚀 Automated ETL pipelines

🚀 API endpoints for programmatic access

🚀 Mobile-responsive dashboard

🚀 Real-time notifications and alerts

🚀 Kubernetes deployment configuration

🚀 Advanced monitoring with Prometheus & Grafana

---
---

## 📝 Requirements

See `requirements.txt` for complete dependencies with pinned versions:

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.7.0
prophet>=1.1.0
openpyxl>=3.10.0
scikit-learn>=1.2.0
```

### Installation from requirements.txt
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### Customizable Parameters

- **Forecasting Horizon:** 7-90 days
- **Confidence Intervals:** 80-99%
- **Segmentation:** 3-8 clusters
- **Holding Costs:** Configurable percentage
- **Lead Time:** Adjustable days
- **Safety Stock:** Configurable buffer days
- **Streamlit Theme:** Customizable colors (see `.streamlit/config.toml`)
- **Docker Port:** Default 8501 (configurable)

### Streamlit Configuration Files

#### `.streamlit/config.toml` - Deployed on Streamlit Cloud
```toml
[server]
port = 8501
headless = true
runOnSave = true

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
```

---

---

## ✅ Project Status

✅ Completed End-to-End Retail Analytics Platform

✅ Production-Grade Features (7/7)

✅ SLA Monitoring Enabled

✅ Docker Containerization Ready

✅ CI/CD Pipeline Automated

✅ GitHub Actions Configured

✅ Streamlit Configuration Optimized

✅ Enterprise Ready

✅ Dashboard Ready

✅ Portfolio Ready

---

---

## 👨‍💻 Author

**Tirtha Brata Das**
Data Science and Machine Learning enthusiast with expertise in building end-to-end analytics solutions and production-grade applications.

- 📧 **Email:** tirthabrata0407@gmail.com
- 🔗 **GitHub:** https://github.com/tirthabrata0407-cloud

**RetailPulse v2.0** | 7 Production Features | SLA Monitoring | Enterprise Ready | Docker & CI/CD Enabled

*Platform Status: Active & Maintained*
*Last Updated: 2026-05-29*
*Version: 2.0 (Production)*
