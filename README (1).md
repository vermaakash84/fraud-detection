# 🕵️ Credit Card Fraud Detection using Autoencoder & XGBoost

This project implements and compares **unsupervised** (Autoencoder) and **supervised** (XGBoost) learning methods to detect fraudulent credit card transactions from a highly imbalanced dataset.

---

## 📦 Project Overview

| Model       | Type         | AUC-ROC | Precision | Recall | F1 Score | Threshold/Quantile |
|-------------|--------------|---------|-----------|--------|----------|---------------------|
| Autoencoder | Unsupervised | 0.9494  | 0.8314    | 0.5711 | 0.6771   | Quantile = 0.999     |
| XGBoost     | Supervised   | 0.9765  | 0.8632    | 0.8367 | 0.8497   | Threshold = 0.3      |

> ✅ XGBoost performed best overall.  
> ✅ Autoencoder detected fraud well under high precision settings, useful for anomaly detection pipelines.

---

## 📊 Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Total records:** 284,807
- **Fraudulent cases:** 492 (~0.172%)
- **Features:** 30 anonymized features + `Time`, `Amount`, `Class`

---

## 🧠 Models Implemented

### 🔁 Autoencoder (Unsupervised Deep Learning)
- Built using **PyTorch**
- Trained on **normal (non-fraudulent)** data only
- Uses **Reconstruction Error** to detect anomalies
- Tuned using **quantile thresholds**

### ✅ XGBoost (Supervised Machine Learning)
- Used full labeled data
- Handled class imbalance with `scale_pos_weight`
- Tuned using **GridSearchCV**
- Evaluated across various decision thresholds

---

## 📈 Evaluation Metrics

### Autoencoder at different Quantiles:

| Quantile | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| 0.990    | 0.4128    | 0.8130 | 0.5476   |
| **0.999**| **0.8314**| 0.5711 | **0.6771** |

### XGBoost at different Thresholds:

| Threshold | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| 0.5       | 0.9286    | 0.7879 | 0.8525   |
| **0.3**   | 0.8632    | 0.8367 | **0.8497** |

---

## 📂 Project Files

| File                          | Description                                |
|-------------------------------|--------------------------------------------|
| `autoencoder_pipeline.ipynb` | Autoencoder training and evaluation        |
| `xgboost_model.ipynb`        | XGBoost training and threshold tuning      |
| `requirements.txt`           | Project dependencies                       |
| `README.md`                  | This documentation                        |
| `plots/`                     | Visualization outputs (ROC, F1, PR curves) |
| `models/`                    | Saved `.pth` and `.pkl` models (optional)  |

---

## 🚀 How to Run

### Step 1: Clone the repo
```bash
git clone https://github.com/your-username/fraud-detection-autoencoder-xgboost.git
cd fraud-detection-autoencoder-xgboost
```
