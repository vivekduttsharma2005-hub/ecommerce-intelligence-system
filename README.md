# ecommerce-intelligence-system
End-to-end E-commerce Intelligence System with dashboarding, customer segmentation (RFM + KMeans), recommendation engine, and demand forecasting using Streamlit.
# 🛒 E-commerce Intelligence System

## 🚀 Project Overview

This project is an **end-to-end Data Science application** that analyzes e-commerce data and provides actionable business insights. It combines **data analysis, machine learning, and interactive visualization** into a single dashboard.

---

## 🎯 Key Features

### 📊 Sales Dashboard

* Total Revenue, Orders, Customers
* Monthly revenue trends
* Top products & countries analysis

### 👥 Customer Segmentation

* RFM (Recency, Frequency, Monetary) analysis
* K-Means clustering
* Customer behavior visualization

### 📈 Demand Forecasting

* Time series forecasting
* Prophet-based model (with fallback support)
* Future revenue prediction

### 🤖 Recommendation System

* Collaborative filtering
* Product recommendations for each customer
* Purchase history insights

---

## 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy** (Data Processing)
* **Scikit-learn** (Machine Learning)
* **Prophet / Fallback Model** (Forecasting)
* **Plotly** (Visualization)
* **Streamlit** (Web App)

---

## 📂 Project Structure

```
project/
│── app.py              # Main Streamlit application
│── data.csv            # Dataset
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
│── LICENSE             # MIT License
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/ecommerce-intelligence-system.git
cd ecommerce-intelligence-system
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

👉 https://ecommerce-intelligence-system-jfvh949jro3qo6v7se3wzn.streamlit.app/

---

## 📊 Dataset

* Online Retail Dataset
* Contains transaction-level data:

  * CustomerID
  * InvoiceDate
  * Product details
  * Quantity & Price

---

## 💡 Key Insights

* Identifies high-value customers
* Tracks sales performance over time
* Predicts future demand trends
* Recommends products to improve sales

---

## 🧠 Future Improvements

* Add ARIMA / LSTM forecasting
* Real-time data integration
* User authentication system
* Advanced recommendation models

---

## 👨‍💻 Author

**Vivek Dutt Sharma**

---

## 📄 License

This project is licensed under the MIT License.
