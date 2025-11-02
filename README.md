# Telco Customer Churn Analysis

## Project Overview

This project focuses on analyzing customer churn in a telecommunications company using the Telco Customer Churn dataset from Kaggle. The primary objective is to understand the factors contributing to customer churn and develop machine learning models to predict which customers are likely to churn. The insights gained from this analysis can inform targeted customer retention strategies to minimize churn and maximize customer lifetime value.

## Dataset

The dataset used in this project is the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) available on Kaggle. It contains 7,043 observations and 21 features, including:

*   Customer demographic information (gender, senior citizen, partner, dependents)
*   Services subscribed (phone service, multiple lines, internet service, online security, online backup, device protection, tech support, streaming TV, streaming movies)
*   Contract and billing information (contract type, paperless billing, payment method, monthly charges, total charges)
*   Churn status (the target variable)

## Methodology

The project follows a standard data science methodology:

1.  **Data Extraction and Loading:** The dataset was downloaded and loaded into a pandas DataFrame.
2.  **Data Cleaning and Preprocessing:** Handled missing values, converted data types, and harmonized inconsistent labels in service-related columns.
3.  **Exploratory Data Analysis (EDA):** Conducted a thorough EDA to understand the data distribution, identify patterns, and visualize relationships between features and churn. This included analyzing demographic profiles, service usage, contract and billing behavior, customer tenure, and feature correlations. Statistical tests (Chi-square) were performed to assess the significance of categorical features in relation to churn.
4.  **Feature Engineering and Selection:** Prepared the data for modeling by creating different feature sets based on EDA insights and statistical test results. This involved using a custom transformer for label harmonization and applying encoding techniques (Ordinal Encoding for binary features, One-Hot Encoding for multi-category features) and scaling (StandardScaler for numerical features).
5.  **Model Development and Evaluation:** Trained and evaluated several classification models to predict customer churn:
    *   Logistic Regression
    *   Random Forest
    *   XGBoost
    *   Support Vector Machine (SVC)
    *   k-Nearest Neighbors (k-NN)

    Models were trained within a scikit-learn pipeline, and hyperparameters were tuned using GridSearchCV with cross-validation, optimizing for the ROC-AUC metric. Model performance was evaluated using classification reports, confusion matrices, and ROC curves. Three different feature scenarios were explored to assess the impact of feature selection on model performance.

## Key Findings

The EDA revealed that factors such as customer tenure, contract type (month-to-month having higher churn), internet service type (Fiber Optic having higher churn), payment method (Electronic Check having higher churn), senior citizen status, absence of partners and dependents, and lack of value-added services (online security, tech support, etc.) are significantly associated with higher churn rates.

Comparing the model performances, the **XGBoost Classifier** generally performed the best across the tested scenarios, achieving the highest ROC-AUC score in Scenario 1 (including all features).

