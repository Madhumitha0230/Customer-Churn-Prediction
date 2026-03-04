# 📊 Customer Churn Prediction using Machine Learning

## 🚀 Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses.  
This project predicts whether a customer will leave a service based on historical data using Machine Learning classification techniques.

The goal is to help businesses identify high-risk customers and take preventive actions to reduce revenue loss.

---

## 🎯 Objective

- Predict whether a customer will churn (Yes/No)
- Analyze customer behavior patterns
- Identify key factors influencing churn
- Compare multiple classification models

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📂 Dataset

- Telco Customer Churn Dataset
- 7,000+ customer records
- Features include:
  - Tenure
  - Monthly Charges
  - Total Charges
  - Contract Type
  - Internet Service
  - Payment Method
  - Churn (Target Variable)

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Removed unnecessary columns
- Converted TotalCharges to numeric
- Handled missing values
- Encoded categorical variables
- Scaled features (for Logistic Regression)

### 2️⃣ Model Building
Two classification models were implemented:

- Logistic Regression
- Random Forest Classifier

### 3️⃣ Model Evaluation
- Accuracy Score
- Classification Report
- Confusion Matrix
- ROC Curve
- AUC Score
- Feature Importance Analysis

---

## 📈 Results

| Model | Accuracy | AUC Score |
|-------|----------|-----------|
| Logistic Regression | ~80% | ~0.84 |
| Random Forest | ~85% | ~0.88 |

Random Forest performed better and identified the most influential features contributing to customer churn.

---

## 🔍 Key Insights

- Contract type significantly affects churn probability.
- Customers with month-to-month contracts are more likely to churn.
- Higher monthly charges increase churn risk.
- Shorter tenure customers are more likely to leave.

---

## 📊 Visualizations Included

- Confusion Matrix Heatmap
- Feature Importance Graph
- ROC Curve Comparison

---

## 📁 Project Structure
Customer-Churn-Prediction/
│
├── churn_model.py
├── churn.csv
├── README.md
├── .gitignore


---

## 💡 Future Improvements

- Hyperparameter tuning
- Deployment using Streamlit
- Model saving using joblib
- Real-time churn prediction app

---

## 👩‍💻 Author

**Madhumitha L**

Machine Learning & Data Science Enthusiast  
Internship Project Submission  

---

## ⭐ If You Found This Useful

Give this repository a ⭐ on GitHub!
