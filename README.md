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
├── export_model.py         # Trains and serializes XGBoost bundle to joblib
├── predictor.py            # ChurnPredictor inference & retention advice engine
├── schemas.py              # Pydantic validation & API schemas
├── app.py                  # Production FastAPI REST application
├── test_api.py             # Automated unit & integration tests
├── requirements.txt        # Python dependencies
└── outputs/                # Auto-generated on run
    ├── churn_model.joblib  # Serialized model & preprocessing artifact
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

## 🌐 FastAPI REST Service

The project includes a production REST API for real-time customer churn scoring, batch prediction, and retention recommendations.

### Start the API Server

```bash
# Activate virtual environment
source venv/bin/activate

# Launch Uvicorn dev server
uvicorn app:app --reload --port 8000
```

Once running, explore the interactive Swagger documentation at:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status and overview |
| `GET` | `/health` | Service health & model readiness check |
| `GET` | `/model/info` | Model metadata, training metrics, top features |
| `POST` | `/predict` | Real-time churn scoring for a single customer |
| `POST` | `/predict/batch` | Bulk scoring for multiple customers |
| `POST` | `/predict/csv` | Upload a CSV file and score all rows |

### Sample Single Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST-1049",
    "gender": "Female",
    "senior_citizen": 0,
    "has_partner": 0,
    "has_dependents": 0,
    "tenure": 2,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "tech_support": "No",
    "monthly_charges": 89.50,
    "num_support_tickets": 4
  }'
```

### Sample Response

```json
{
  "customer_id": "CUST-1049",
  "churn_probability": 0.8412,
  "churn_prediction": true,
  "risk_tier": "High Risk",
  "key_risk_factors": [
    "Month-to-month contract (high churn vulnerability)",
    "Early customer lifecycle (2 months tenure)",
    "Frequent support tickets (4 tickets filed)",
    "Payment method: Electronic check (higher friction/delinquency rate)",
    "High monthly spend ($89.50/mo)",
    "High charges without bundled protection/support"
  ],
  "recommended_retention_actions": [
    "Offer 15% discount incentive to upgrade to a 1-year or 2-year contract.",
    "Trigger VIP onboarding outreach and customer success check-in.",
    "Priority ticket escalation: Assign dedicated Tier-2 specialist to resolve open issues.",
    "Promote $5/mo bill credit for enabling automated card or bank payments.",
    "Offer complimentary 6-month trial of Tech Support and Online Security."
  ]
}
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
