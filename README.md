# 📊 Customer Churn Prediction Pipeline

An end-to-end **Machine Learning pipeline** for predicting customer churn in a Telco-style business — from synthetic data generation through EDA, feature engineering, XGBoost classification, evaluation, and SHAP interpretability.

---

## ✨ Features

- 🏗️ **Synthetic Data Generation** — Realistic Telco-style dataset (10,000 customers) with logistic churn labels based on contract type, tenure, charges, and more
- 🔍 **Exploratory Data Analysis (EDA)** — Distribution plots by churn status, correlation heatmap
- ⚙️ **Feature Engineering** — 8 domain-driven features (tenure bins, ticket rate, contract risk score, auto-payment flag, etc.)
- 🤖 **XGBoost Model** — Tuned via `RandomizedSearchCV` (60 iterations × 5-fold stratified CV)
- 📈 **Comprehensive Evaluation** — Accuracy, F1, ROC-AUC, Average Precision, confusion matrix
- 🔬 **SHAP Interpretability** — Global feature importance via TreeExplainer
- 📁 **Auto-saved Outputs** — All plots and the model report saved to `outputs/`

---

## 🗂️ Project Structure

```
churn_prediction/
├── churn_pipeline.py       # Full end-to-end ML pipeline (single script)
├── requirements.txt        # Python dependencies
└── outputs/                # Auto-generated on run
    ├── eda_distributions.png
    ├── correlation_heatmap.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── feature_importance.png
    ├── shap_summary.png
    └── model_report.txt
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Preprocessing, train/test split, CV, metrics |
| `xgboost` | Gradient boosted classifier |
| `shap` | Model interpretability |
| `matplotlib`, `seaborn` | Visualizations |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/ojasvcode/churn_prediction.git
cd churn_prediction

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
python churn_pipeline.py
```

The pipeline will print progress for each step and save all outputs to the `outputs/` directory.

---

## 🔬 Pipeline Steps

| Step | Description |
|---|---|
| **1. Data Generation** | 10,000 synthetic Telco-style customers |
| **2. EDA** | Distribution plots + correlation heatmap |
| **3. Feature Engineering** | 8 new domain features |
| **4. Preprocessing** | Label encoding, one-hot encoding, 80/20 split |
| **5. Model Training** | XGBoost + RandomizedSearchCV (60 iter × 5 fold) |
| **6. Evaluation** | Confusion matrix, ROC, PR curve, feature importance |
| **7. SHAP Analysis** | Global feature impact summary plot |

---

## 📊 Key Features Engineered

| Feature | Description |
|---|---|
| `charge_per_month_of_tenure` | Spending intensity over time |
| `tenure_bin` | Tenure bucketed into 4 groups |
| `is_new_customer` | Flag for tenure ≤ 6 months |
| `has_premium_support` | Online security OR tech support enabled |
| `high_monthly_spend` | Monthly charges > $70 |
| `ticket_rate` | Support tickets normalized by tenure |
| `contract_risk_score` | Month-to-month=2, One year=1, Two year=0 |
| `auto_payment` | Bank transfer or Credit card payment |

---

## 📈 Sample Results

| Metric | Score |
|---|---|
| ROC-AUC | ~0.92+ |
| F1 (Churn class) | ~0.78+ |
| Accuracy | ~0.85+ |

*(Results vary slightly due to random seed in hyperparameter search)*

---

## 📄 License

This project is for educational purposes only.

---

<div align="center">Made with ❤️ by <a href="https://github.com/ojasvcode">ojasvcode</a></div>
