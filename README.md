# Anomaly Detection in Credit Card Fraud

Detecting fraudulent credit card transactions in a severely imbalanced dataset (0.17% fraud), comparing unsupervised anomaly detection against tuned supervised classifiers.

**Suggested repo name:** `credit-card-fraud-anomaly-detection`

## Overview

Credit card fraud detection is the canonical extreme-imbalance ML problem: fraud is rare, expensive to miss, and expensive to over-flag (false positives block legitimate purchases). This project builds both an unsupervised anomaly detector (Isolation Forest, trained with zero fraud labels) and tuned supervised classifiers, and shows exactly how much supervised learning adds when labels are available.

## Dataset

- **Source:** ULB Machine Learning Group Credit Card Fraud dataset (mirrored via [nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection](https://github.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection)), originally on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size:** 284,807 European cardholder transactions over 2 days in September 2013
- **Target:** `Class` (1 = fraud, 0 = legitimate) — **0.173% fraud rate** (492 frauds)
- **Features:** `V1`-`V28` (PCA-transformed, original features confidential), `Time`, `Amount`

## Repository Structure

```
credit-card-fraud-anomaly-detection/
├── README.md
├── requirements.txt
└── notebooks/
    └── f01_Credit_Card_Fraud_Anomaly_Detection.ipynb
```

## Getting Started

```bash
git clone https://github.com/<your-username>/credit-card-fraud-anomaly-detection.git
cd credit-card-fraud-anomaly-detection
pip install -r requirements.txt
jupyter notebook notebooks/f01_Credit_Card_Fraud_Anomaly_Detection.ipynb
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

## Methodology

1. **EDA** — extreme class imbalance visualization, transaction amount/time patterns, correlation of PCA components with fraud.
2. **Unsupervised anomaly detection** — Isolation Forest trained with **no fraud labels at all**, evaluated against the (held-out) true labels purely to benchmark it.
3. **Supervised modeling** — trained on a class-balanced subsample (all frauds + a 10:1 sample of legitimate transactions, standard practice for this exact dataset) but **evaluated on the untouched, naturally-imbalanced test set**, so metrics reflect real deployment.
4. **Model comparison** — Logistic Regression, Random Forest, XGBoost via 3-fold stratified CV (F1).
5. **Tuning** — `GridSearchCV` over XGBoost (n_estimators, max_depth, learning_rate).
6. **Evaluation** — precision/recall, ROC-AUC, and (most importantly given the imbalance) **PR-AUC**.

## Results

| Approach | Metric | Value |
|---|---|---|
| Isolation Forest (unsupervised) | Fraud F1 | 0.25 (precision 0.25, recall 0.26) |
| **XGBoost (tuned, supervised)** | Fraud F1 | **0.57** (precision 0.42, recall 0.89) |
| | **Test ROC-AUC** | **0.982** |
| | **Test PR-AUC** | **0.780** |

**Top features:** V17, V14, V12, V10, V16 — a handful of PCA components carry most of the fraud signal, consistent with published analyses of this dataset.

## Key Insights

- Supervised learning **substantially outperforms** unsupervised anomaly detection here — fraud in this dataset has a specific, learnable pattern in PCA space, not just generic statistical unusualness.
- PR-AUC (0.78), not accuracy, is the right headline metric — a model that predicts "always legitimate" would score 99.83% accuracy while catching zero fraud.
- The tuned model catches 89% of fraud (recall) at 42% precision — a reasonable operating point for flagging transactions for step-up authentication rather than outright blocking.

## Future Work

- Cost-sensitive threshold tuning based on the actual $ cost of false positives vs. false negatives.
- Try autoencoder-based anomaly detection as an additional unsupervised baseline.
- Explore SMOTE and other oversampling techniques as an alternative to undersampling.

## Data Source

Dataset collected and analyzed during a research collaboration of Worldline and the Machine Learning Group (ULB) on big data mining and fraud detection. Distributed via Kaggle for research.

