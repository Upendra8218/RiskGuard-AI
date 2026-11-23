# AI-Powered Loan Approval, Credit Risk and Fraud Detection System

## Project Overview

This project develops an AI-based decision-support system for analyzing credit risk and detecting potentially fraudulent transactions.

The system combines:

- Credit-risk prediction
- Fraud detection
- Risk scoring
- Risk classification
- Explainable AI
- Interactive decision support

## Objectives

1. Predict the probability of loan default.
2. Calculate an illustrative credit-risk score.
3. Classify applicants into LOW, MEDIUM, and HIGH risk.
4. Detect potentially fraudulent transactions.
5. Handle severe class imbalance.
6. Tune fraud detection thresholds.
7. Provide model explanations using SHAP.
8. Combine credit and fraud predictions into a decision-support system.

## Datasets

### Credit Risk Dataset

Home Credit Default Risk dataset.

Target:

`TARGET`

where:

- `0` = No Default
- `1` = Default

### Fraud Dataset

Credit Card Fraud Detection dataset.

Target:

`Class`

where:

- `0` = Legitimate transaction
- `1` = Fraudulent transaction

## Machine Learning Models

### Credit Risk

- Logistic Regression
- Decision Tree
- Random Forest

### Fraud Detection

- Logistic Regression
- SMOTE for class imbalance

## Evaluation Metrics

The project evaluates:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Log Loss
- Brier Score
- Confusion Matrix

## Explainability

SHAP is used to identify important factors influencing model predictions.

## Risk Score

The illustrative credit-risk score is:

Risk Score = 100 × (1 − Default Probability)

The score is a project-defined score and is not an official bank credit score.

## Important Disclaimer

This project is an academic decision-support prototype.

It should not be used to automatically approve or reject real loans without appropriate validation, governance, human oversight, regulatory compliance, and institutional policies.