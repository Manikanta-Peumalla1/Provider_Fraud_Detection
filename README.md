# 🏥 Provider Fraud Detection using Machine Learning

## 📌 Overview
Healthcare fraud is a major challenge for insurance providers, leading to billions of dollars in losses every year. Fraudulent claims can include unnecessary treatments, inflated billing, or services not rendered.

This project focuses on building a **Machine Learning model to identify potentially fraudulent healthcare providers** using claims data.

---

## 🎯 Objective
- Predict whether a healthcare provider is **fraudulent or not**
- Identify **key features contributing to fraud**
- Enable **early detection and investigation of suspicious claims**

---

## 📊 Dataset
The dataset consists of multiple components:

- **Inpatient Claims Data**
- **Outpatient Claims Data**
- **Beneficiary Details**
- **Provider Information**

These datasets are merged and transformed to create a **provider-level dataset** for modeling.

---

## 🧠 Problem Type
- **Binary Classification**
  - Target: Fraud / Not Fraud

---

## ⚙️ Project Workflow

### 1. Data Preprocessing
- Handling missing values
- Date feature engineering (hospital stay duration)
- Data cleaning and transformations

### 2. Feature Engineering
- Aggregated provider-level features
- Claim counts, reimbursement averages
- Disease-based features (Diabetes, Cancer, etc.)
- Hospital stay statistics

### 3. Exploratory Data Analysis (EDA)
- Distribution of claims
- Fraud vs non-fraud comparison
- Feature correlations

### 4. Handling Imbalanced Data
- Fraud cases are rare → class imbalance handled using:
  - SMOTE / Resampling techniques

### 5. Model Building
Implemented multiple ML models:
- Logistic Regression
- Random Forest
- XGBoost / Gradient Boosting

### 6. Model Evaluation
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC

---

## 📈 Key Features Used
- Claim Count (IP/OP)
- Average Claim Amount
- Deductible Amounts
- Number of Days in Hospital
- Chronic Disease Indicators
- Beneficiary Counts

---

## 📊 Results
- Achieved strong performance in identifying fraudulent providers
- Improved recall to minimize **false negatives** (critical in fraud detection)
- Identified important features contributing to fraud detection

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Manikanta-Peumalla1/Provider_Fraud_Detection.git

# Navigate to project folder
cd Provider_Fraud_Detection

# Install dependencies
pip install -r requirements.txt

# Run the notebook / script