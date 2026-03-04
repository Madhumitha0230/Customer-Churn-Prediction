# 📊 Customer Churn Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Classification-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## 🚀 Project Overview

Customer churn is a major challenge for subscription-based businesses. Losing customers directly impacts revenue and growth.

This project builds Machine Learning classification models to predict whether a customer is likely to leave a service based on historical behavioral and billing data.

The goal is to help businesses proactively identify high-risk customers and implement retention strategies.

---

## 🎯 Business Problem

Companies lose significant revenue due to customer churn.

Instead of reacting after customers leave, businesses can use predictive analytics to:

- Identify customers at high risk
- Offer personalized retention plans
- Reduce churn rate
- Increase customer lifetime value

---

## 📂 Dataset Information

Dataset: **Telco Customer Churn Dataset**

- 7,000+ customer records
- 20+ features
- Target Variable: `Churn (Yes/No)`

### Important Features:

- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Internet Service
- Payment Method
- Online Security
- Tech Support

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing

- Removed unnecessary columns (CustomerID)
- Converted `TotalCharges` to numeric
- Handled missing values
- Encoded categorical variables using Label Encoding
- Split dataset into training & testing sets (80-20)
- Applied feature scaling (for Logistic Regression)

---

### 2️⃣ Model Building

Two classification models were implemented:

#### 🔹 Logistic Regression
- Baseline classification model
- Simple and interpretable
- Good for linear decision boundaries

#### 🔹 Random Forest Classifier
- Ensemble learning algorithm
- Handles non-linearity
- Higher accuracy and better generalization

---

## 📊 Model Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report
- ROC Curve
- AUC Score
- Feature Importance

---

## 📈 Results

| Model | Accuracy | AUC Score |
|--------|----------|-----------|
| Logistic Regression | ~80% | 0.84 |
| Random Forest | ~85% | 0.88 |

✅ **Random Forest performed better and achieved higher AUC.**

---

## 📉 ROC Curve Analysis

The ROC Curve comparison shows that Random Forest has a larger area under the curve (AUC = 0.88), indicating better classification performance compared to Logistic Regression.

---

## 🔍 Feature Importance (Random Forest)

Top influential features contributing to churn:

1. Contract Type
2. Tenure
3. Monthly Charges
4. Total Charges
5. Internet Service

These insights help businesses understand what drives customer churn.

---

## 💡 Key Business Insights

✔ Customers with month-to-month contracts are more likely to churn  
✔ Customers with high monthly charges show higher churn risk  
✔ Short-tenure customers are more likely to leave  
✔ Long-term contract customers are more stable  

---

## 🧠 Business Impact

This model enables:

- Early churn detection
- Targeted marketing strategies
- Revenue optimization
- Improved customer retention strategy

---

## 📁 Project Structure
Customer-Churn-Prediction/
│
├── churn_model.py
├── churn.csv
├── requirements.txt
├── README.md
└── .gitignore

---

## ▶️ How to Run This Project

### Step 1: Clone Repository
git clone https://github.com/Madhumitha0230/Customer-Churn-Prediction.git

### Step 2: Navigate to Folder

### Step 3: Install Dependencies

### Step 4: Run Model

---

## 🚀 Future Improvements

- Hyperparameter tuning (GridSearchCV)
- Model saving using joblib
- Streamlit web deployment
- Real-time churn prediction system
- Model explainability using SHAP

---

## 📌 Internship Submission

This project was developed as part of my Machine Learning Internship.

It demonstrates:

- Data preprocessing skills
- Feature engineering
- Classification model implementation
- Model evaluation techniques
- Business insight generation

---

## 👩‍💻 Author

**Madhumitha L**  
Machine Learning & Data Science Enthusiast  
GitHub: https://github.com/Madhumitha0230

---

## ⭐ Support

If you found this project useful, please give it a ⭐ on GitHub!
