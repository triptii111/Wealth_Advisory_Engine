# Wealth Advisory Engine

An end-to-end machine learning project that recommends the best investment option
(Stocks, Mutual Fund, Fixed Deposit, Cryptocurrency, or Government Bond) based on a person's financial profile.

---

## Project Overview

| Section | Details |
| --- | --- |
| **Objective** | To build an ML-based multi-class classification model that recommends the best investment option from 5 categories based on a user's complete financial profile |
| **Approach** | Preprocessed 32,424 rows x 20 features of synthetic personal finance data using LabelEncoder and StandardScaler for ML readiness |
| | Engineered 3 domain-specific financial features — Disposable Income, Expense Ratio, and Financial Health Score — from raw income and credit data |
| | Generated a 5-class target variable using rule-based financial logic on income, credit score, age, and debt |
| | Trained and compared 3 models and tuned the best using GridSearchCV with 3-fold cross-validation |
| **Results** | Achieved 99.83% test accuracy with a tuned Random Forest classifier across all 5 investment classes on 6,485 test samples |
| | Obtained per-class F1-scores of 0.92–1.00, with macro precision 0.99 and recall 0.97 |
| | GridSearchCV identified optimal params {n_estimators: 100, max_depth: None, min_samples_split: 5} with 99.73% CV accuracy |
| | Built an interactive prediction interface accepting 13 financial inputs with confidence score and estimated annual return range |

---

## Tech Stack

- **Language:** Python
- **Notebook:** Jupyter Notebook
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Joblib

---

## Dataset

- **File:** `synthetic_personal_finance_dataset.csv`
- **Rows:** 32,424
- **Features:** 20 (age, income, expenses, savings, credit score, loan details, employment, education, region, etc.)
- **Target:** `Investment_Recommendation` — created using rule-based financial logic (5 classes)

---

## ML Pipeline

| Part | Description |
| --- | --- |
| Part 1 | Import Libraries |
| Part 2 | Load Dataset |
| Part 3 | Initial Exploration (shape, info, describe, missing values, duplicates) |
| Part 4 | Exploratory Data Analysis (histograms, heatmap, countplots, boxplots, distributions) |
| Part 5 | Data Cleaning (handle missing values) |
| Part 6 | Feature Engineering (Disposable Income, Expense Ratio, Financial Health Score) |
| Part 7 | Target Variable Creation using financial rules |
| Part 8 | Preprocessing (LabelEncoding, Train-Test Split, StandardScaler) |
| Part 9 | Train and Compare Models (Logistic Regression, Decision Tree, Random Forest) |
| Part 10 | Hyperparameter Tuning with GridSearchCV |
| Part 11 | Model Evaluation (Accuracy, Classification Report, Confusion Matrix, Feature Importance) |
| Part 12 | Save Model (.pkl files) |
| Part 13 | Personalized Investment Recommendation (user input + prediction) |

---

## Feature Engineering

Three new features were created from existing columns:

| Feature | Formula |
| --- | --- |
| Disposable Income | Monthly Income - Monthly Expenses |
| Expense Ratio | Monthly Expenses / Monthly Income |
| Financial Health Score | (Credit Score / 850) x 100 |

---

## Target Variable — Investment Rules

The target column was created using the following financial logic:

| Condition | Recommendation |
| --- | --- |
| Credit >= 750, Income >= $6000, Savings Ratio >= 6, DTI <= 0.3 | Stocks |
| Age <= 30, Income >= $4500, Credit >= 620, DTI <= 0.5 | Cryptocurrency |
| Credit >= 650, Income >= $4000, DTI <= 0.6 | Mutual Fund |
| Credit >= 500, Income >= $2000 | Fixed Deposit |
| Everything else | Government Bond |

## Saved Files

| File | Description |
| --- | --- |
| `investment_model.pkl` | Trained and tuned Random Forest model |
| `scaler.pkl` | Fitted StandardScaler |
| `target_encoder.pkl` | LabelEncoder for the target variable |
| `label_encoders.pkl` | LabelEncoders for categorical input features |

---

## How to Run

1. Clone or download the project folder
2. Make sure `synthetic_personal_finance_dataset.csv` is in the same folder as the notebook
3. Install dependencies:

```
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

4. Open `investment_buddy.ipynb` in Jupyter Notebook
5. Run all cells in order (Kernel > Restart & Run All)
6. When you reach Part 13, enter your financial details to get your recommendation

---

