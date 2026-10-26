"""
Model Training & Artifact Serialization Script
==============================================
Trains an XGBoost model with optimized hyperparameters and saves the
full pipeline bundle (model, encoders, feature columns, metrics)
to `outputs/churn_model.joblib` for REST API inference.
"""

import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_ARTIFACT_PATH = os.path.join(OUTPUT_DIR, "churn_model.joblib")


def generate_training_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate Telco-style churn training records."""
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], n)
    senior_citizen = rng.choice([0, 1], n, p=[0.84, 0.16])
    has_partner = rng.choice([0, 1], n, p=[0.52, 0.48])
    has_dependents = rng.choice([0, 1], n, p=[0.70, 0.30])
    tenure = np.clip(rng.normal(32, 20, n).astype(int), 1, 72)
    contract_type = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n,
        p=[0.34, 0.23, 0.22, 0.21],
    )
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    online_security = np.where(
        internet_service == "No", "No internet", rng.choice(["Yes", "No"], n, p=[0.35, 0.65])
    )
    tech_support = np.where(
        internet_service == "No", "No internet", rng.choice(["Yes", "No"], n, p=[0.35, 0.65])
    )

    base_monthly = rng.uniform(18, 30, n)
    fiber_add = np.where(internet_service == "Fiber optic", rng.uniform(25, 50, n), 0)
    dsl_add = np.where(internet_service == "DSL", rng.uniform(10, 25, n), 0)
    monthly_charges = np.round(base_monthly + fiber_add + dsl_add, 2)
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 50, n), 2)
    total_charges = np.clip(total_charges, monthly_charges, None)

    num_support_tickets = rng.poisson(1.5, n)

    # Churn probability calculation
    churn_logit = np.full(n, -1.2)
    churn_logit = np.where(contract_type == "Month-to-month", churn_logit + 1.5, churn_logit)
    churn_logit = np.where(contract_type == "Two year", churn_logit - 1.8, churn_logit)
    churn_logit += -0.04 * tenure
    churn_logit = np.where(tenure < 6, churn_logit + 1.0, churn_logit)
    churn_logit += 0.015 * monthly_charges
    churn_logit = np.where(internet_service == "Fiber optic", churn_logit + 0.6, churn_logit)
    churn_logit = np.where(online_security == "Yes", churn_logit - 0.7, churn_logit)
    churn_logit = np.where(tech_support == "Yes", churn_logit - 0.7, churn_logit)
    churn_logit = np.where(payment_method == "Electronic check", churn_logit + 0.5, churn_logit)
    churn_logit += 0.25 * num_support_tickets
    churn_logit = np.where(senior_citizen == 1, churn_logit + 0.3, churn_logit)
    churn_logit = np.where(has_dependents == 1, churn_logit - 0.3, churn_logit)

    # Interactions
    churn_logit = np.where((tenure < 12) & (contract_type == "Month-to-month"), churn_logit + 0.8, churn_logit)
    churn_logit = np.where(
        (monthly_charges > 65) & (tech_support != "Yes") & (online_security != "Yes"),
        churn_logit + 0.5,
        churn_logit,
    )
    churn_logit += rng.normal(0, 0.3, n)

    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churned = rng.binomial(1, churn_prob)

    return pd.DataFrame({
        "gender": gender,
        "senior_citizen": senior_citizen,
        "has_partner": has_partner,
        "has_dependents": has_dependents,
        "tenure": tenure,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "online_security": online_security,
        "tech_support": tech_support,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_support_tickets": num_support_tickets,
        "churned": churned,
    })


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain features matching training pipeline."""
    df = df.copy()
    df["charge_per_month_of_tenure"] = np.round(df["total_charges"] / (df["tenure"] + 1), 2)
    df["tenure_bin"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 120],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
    )
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["has_premium_support"] = (
        (df["online_security"] == "Yes") | (df["tech_support"] == "Yes")
    ).astype(int)
    df["high_monthly_spend"] = (df["monthly_charges"] > 70).astype(int)
    df["ticket_rate"] = np.round(df["num_support_tickets"] / (df["tenure"] + 1), 4)
    contract_risk_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["contract_risk_score"] = df["contract_type"].map(contract_risk_map).fillna(1).astype(int)
    auto_methods = {"Bank transfer", "Credit card"}
    df["auto_payment"] = df["payment_method"].isin(auto_methods).astype(int)
    return df


def train_and_export():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating synthetic dataset...")
    df_raw = generate_training_data()
    df_feats = compute_features(df_raw)

    target = "churned"
    label_enc_cols = ["tenure_bin"]
    onehot_cols = [
        "gender", "contract_type", "payment_method",
        "internet_service", "online_security", "tech_support"
    ]

    le = LabelEncoder()
    # Fit on all categories
    le.fit(["0-12m", "13-24m", "25-48m", "49-72m"])
    df_feats["tenure_bin"] = le.transform(df_feats["tenure_bin"].astype(str))

    df_encoded = pd.get_dummies(df_feats, columns=onehot_cols, drop_first=True)
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training XGBoost with tuned hyperparameters...")
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.6,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.01,
        reg_lambda=1.5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
        "f1": float(round(f1_score(y_test, y_pred), 4)),
        "roc_auc": float(round(roc_auc_score(y_test, y_proba), 4)),
        "average_precision": float(round(average_precision_score(y_test, y_proba), 4)),
    }
    print(f"  → Metrics: ROC-AUC={metrics['roc_auc']}, F1={metrics['f1']}, Accuracy={metrics['accuracy']}")

    # Extract top features by gain
    gain_scores = model.get_booster().get_score(importance_type="gain")
    sorted_features = sorted(gain_scores.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:10]]

    # Bundle for serving
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "label_encoder_classes": le.classes_.tolist(),
        "onehot_cols": onehot_cols,
        "decision_threshold": 0.5,
        "metrics": metrics,
        "top_features": top_features,
        "algorithm": "XGBoost Classifier (Gradient Boosted Trees)",
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(bundle, MODEL_ARTIFACT_PATH)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Model successfully exported to {MODEL_ARTIFACT_PATH}")
    return bundle


if __name__ == "__main__":
    train_and_export()
