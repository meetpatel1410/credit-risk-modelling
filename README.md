# Credit Risk Modelling and Loan Approval Optimisation

## Project Overview

This project builds an end-to-end credit risk modelling pipeline using the Home Credit Default Risk dataset from Kaggle. The objective of the project is to predict the probability of customer default and support loan approval decision-making using customer application information, bureau credit history, previous loan applications, and repayment behaviour.

The project simulates a real-world retail banking credit risk workflow and includes:

- Probability of Default (PD) modelling
- Credit bureau feature engineering
- Repayment behaviour analysis
- Risk segmentation
- Loan approval optimisation
- Expected loss calculation
- Business interpretation of model outputs

The project was developed using Python, Pandas, NumPy, Scikit-learn, and Jupyter Notebook.

---

# Business Problem

Financial institutions face significant financial losses when customers fail to repay loans. Before approving a loan, banks therefore analyse customer financial history, repayment behaviour, and existing liabilities to estimate the likelihood of default.

The objective of this project is to:

- Identify high-risk borrowers
- Reduce expected financial losses
- Improve loan approval decisions
- Segment customers into different risk categories
- Support risk-based lending strategies

The project aims to replicate how retail banking credit risk models are developed and interpreted in real-world lending environments.

---

# Dataset

Dataset used:

Home Credit Default Risk Dataset

Source:
https://www.kaggle.com/competitions/home-credit-default-risk

The project uses multiple relational datasets to simulate real banking data environments.

| Dataset | Description |
|---|---|
| application_train.csv | Current customer loan application data |
| bureau.csv | Historical credit bureau information from external lenders |
| previous_application.csv | Customer’s previous Home Credit loan applications |
| installments_payments.csv | Historical installment repayment behaviour |

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- PySpark (planned extension)

---

# Project Workflow

## 1. Data Loading and Integration

Multiple banking datasets were loaded and merged using customer identifiers (`SK_ID_CURR`) to create a unified customer-level modelling dataset.

This simulates how banks combine:
- application data
- bureau records
- repayment history
- previous lending behaviour

into a single analytical view of the customer.

---

# 2. Missing Value Analysis

The datasets contained significant missing values, which is common in real banking systems where customer information may be incomplete or unavailable.

Missing value percentages were analysed across all datasets, and missing-value indicator features were created for important variables.

Example missing-value indicators:
- `EXT_SOURCE_2_MISSING_FLAG`
- `TOTAL_BUREAU_DEBT_MISSING_FLAG`

In banking analytics, missingness itself can contain useful risk information. For example, customers without bureau information may behave differently from customers with extensive credit history.

Median imputation was used for numerical variables because banking data often contains extreme outliers.

---

# 3. Feature Engineering

One of the most important stages of credit risk modelling is feature engineering. Banks rarely use raw variables directly for decision-making. Instead, behavioural and financial risk indicators are engineered to better capture customer risk profiles.

Several banking-style risk features were created.

---

## Credit-to-Income Ratio

```python
AMT_CREDIT / AMT_INCOME_TOTAL
```

This feature measures the customer’s loan burden relative to income.

Customers with higher credit exposure compared to income are generally more likely to experience repayment difficulties.

Banks commonly use affordability ratios during underwriting and credit assessment.

---

## Annuity-to-Income Ratio

```python
AMT_ANNUITY / AMT_INCOME_TOTAL
```

This represents repayment burden relative to income.

Higher repayment obligations may increase customer financial stress and default risk.

---

## Bureau Debt Features

Features such as:
- `TOTAL_BUREAU_DEBT`
- `TOTAL_BUREAU_OVERDUE`
- `BUREAU_DEBT_TO_INCOME_RATIO`

were created using bureau credit history data.

These variables capture customer indebtedness and external credit exposure from other lenders.

Banks heavily rely on bureau data to evaluate whether customers are already financially overleveraged before approving additional credit.

---

## Repayment Behaviour Features

Features such as:
- `AVG_PAYMENT_DELAY`
- `MAX_PAYMENT_DELAY`
- `TOTAL_LATE_PAYMENTS`
- `PAYMENT_COMPLETION_RATIO`

were engineered using installment repayment history.

Repayment behaviour is often one of the strongest predictors of future default risk in banking analytics.

Customers with repeated late payments or poor payment completion ratios are generally considered higher risk borrowers.

---

# 4. Probability of Default (PD) Modelling

A Logistic Regression model was trained to estimate customer probability of default.

Logistic Regression is widely used in banking risk modelling because:
- it predicts probabilities directly
- model coefficients are interpretable
- regulators prefer explainable models
- scorecards can be developed from model outputs

The modelling pipeline included:
- median imputation
- feature scaling
- class imbalance handling using balanced class weights

---

# 5. Risk Segmentation

Customers were segmented into multiple risk bands based on predicted probability of default.

| Risk Band | Probability of Default |
|---|---|
| Low Risk | 0–5% |
| Medium Risk | 5–10% |
| High Risk | 10–20% |
| Very High Risk | 20%+ |

Risk segmentation helps banks:
- identify risky borrowers
- optimise approval strategies
- determine pricing and interest rates
- monitor portfolio quality
- estimate portfolio-level expected losses

Customers classified into higher risk bands demonstrated progressively higher actual default rates and expected losses, indicating that the model successfully separated low-risk and high-risk borrowers.

---

# 6. Loan Approval Optimisation

A simple approval/rejection strategy was implemented using probability thresholds.

Example strategy:

- PD < 10% → Approve
- PD ≥ 10% → Reject

This simulates how banks use probability of default thresholds during underwriting and automated loan decisioning.

The model demonstrated that approved customers had significantly lower actual default rates compared to rejected customers.

---

# 7. Expected Loss Calculation

Expected loss was estimated using the banking risk formula:

:contentReference[oaicite:0]{index=0}

Where:
- PD = Probability of Default
- LGD = Loss Given Default
- EAD = Exposure at Default

For demonstration purposes:
- LGD was assumed to be 45%
- EAD was approximated using customer loan amount (`AMT_CREDIT`)

Expected loss calculations help banks estimate potential portfolio losses and optimise lending decisions.

---

# Model Performance

## ROC-AUC Score

The Logistic Regression model achieved approximately:

```text
ROC-AUC ≈ 0.72
```

This indicates good discriminatory power between risky and non-risky borrowers for a first-stage retail banking probability of default model.

---

# Key Insights

- Customers with higher repayment burden and overdue debt demonstrated significantly higher predicted default risk.
- External source credit scores (`EXT_SOURCE_2`, `EXT_SOURCE_3`) were among the strongest predictors of lower default probability.
- Customers classified into higher risk bands exhibited progressively higher actual default rates and expected losses.
- Repayment behaviour features substantially improved model performance and risk separation capability.
- Behavioural and bureau-based features provided stronger predictive power than standalone application features.

---

# Visualisations

The project includes several visualisations for model interpretation and business analysis:

- ROC Curve
- Confusion Matrix
- Risk Band Distribution
- Actual Default Rate by Risk Band
- Expected Loss by Risk Band
- Feature Importance Graphs
- Approval vs Rejection Analysis

---

# Example Business Outcomes

The model demonstrated the ability to:

- identify higher-risk borrowers before loan approval
- segment customers into meaningful risk categories
- reduce portfolio risk exposure through approval thresholds
- estimate expected financial losses
- support data-driven lending decisions

This type of modelling is commonly used in:
- retail banking
- personal lending
- SME lending
- credit card underwriting
- portfolio risk management

---

# Repository Structure

```text
credit-risk-modelling/
│
├── data/
├── notebooks/
│   └── credit_risk_model.ipynb
│
├── outputs/
│   ├── approval_strategy_summary.csv
│   ├── risk_band_summary.csv
│   └── credit_risk_predictions.csv
│
├── images/
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── risk_band_summary.png
│   └── feature_importance.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Future Improvements

Potential future enhancements include:

- XGBoost and LightGBM models
- SHAP explainability
- Hyperparameter tuning
- Cross-validation
- Probability calibration
- PySpark pipeline
- Streamlit dashboard
- AWS deployment
- Real-time scoring API

---

# Author

Meet Patel

Master of Information Technology — RMIT University
