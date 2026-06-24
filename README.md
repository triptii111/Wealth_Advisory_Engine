# Investment Recommendation System using Machine Learning

An end-to-end predictive classification pipeline that maps personal financial metrics to optimal asset investment strategies (Stocks, ETFs, Crypto, Bonds, Mutual Funds, Fixed Deposits).

---

## 📌 Project Overview

| Section | Key Technical Details |
| :--- | :--- |
| **Objective** | To construct an end-to-end machine learning pipeline that profiles users based on financial data and accurately classifies them into tailored investment strategies. |
| **Approach** | • Cleaned and preprocessed synthetic transactional/personal finance records by handling missing entries with median/mode imputation and removing data anomalies via the **Interquartile Range (IQR)** method.<br>• Engineered **10 advanced financial features** (e.g., *Savings Ratio*, *Debt Ratio*, *Disposable Income*, *Emergency Fund Months*, and a composite *Financial Health Score*) to introduce robust domain-specific variance.<br>• Programmed a customized multi-class rule engine to establish a high-fidelity synthetic ground truth target mimicking automated advisory algorithms.<br>• Evaluated **5 classification models** (Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, and **XGBoost**) using stratified splits and `StandardScaler`. <br>• Optimised the top framework using **GridSearchCV** and serialized the trained artifacts (`joblib`) for terminal deployment. |
| **Results** | • Identified driving financial indicators via **Global Feature Importance Analysis**, mapping exactly how variance in debt and savings ratios dictates allocation.<br>• Achieved optimal multi-class accuracy, precision, and recall scores across all engineered investment categories with solid model generalization on unseen test datasets. |

---

## 🛠️ Tech Stack & Dependencies

*   **Language:** Python
*   **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn, Joblib

---

## 🚀 Pipeline Structure

1. **Part 1–4: Data Exploration:** Structural validation, profiling individual unique counts, and missing value verification.
2. **Part 5: Exploratory Data Analysis (EDA):** Univariate distributions, bivariate scatter plots, segmentation box plots, and correlation matrix visualizations.
3. **Part 6: Data Cleaning:** Median/mode imputation and outlier extraction using statistical boundaries.
4. **Part 7: Feature Engineering:** Development of economic financial performance tracking ratios.
5. **Part 8: Rule-Based Target Creation:** Simulating asset programmatic portfolio matching logic.
6. **Part 9: Preprocessing:** Categorical label encoding, stratified data split, and feature normalization.
7. **Part 10–12: Modeling & Optimization:** Comparative evaluation across 5 algorithmic structures followed by hyperparameter tuning via `GridSearchCV`.
8. **Part 13: Model Serialization:** Exporting model binaries and encoders into standalone `.pkl` production files.

---

## 📦 Saved Artifacts

The pipeline automatically serializes the following components for terminal predictions:
*   `investment_model.pkl` (The final trained, hyperparameter-tuned model)
*   `scaler.pkl` (The trained `StandardScaler` instance)
*   `target_encoder.pkl` (The label encoder for output recommendations)
*   `feature_encoders.pkl` (Dictionary containing individual encoders for categorical inputs)
