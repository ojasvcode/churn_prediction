"""
Customer Churn Prediction Pipeline
====================================
End-to-end ML pipeline: synthetic data → EDA → feature engineering →
XGBoost classification → evaluation → SHAP interpretability.

Author : Antigravity AI
Date   : 2026-08-17
"""

import os
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score,
)


import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── plot styling ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("viridis", 10)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
def generate_data(n: int = 10_000) -> pd.DataFrame:
    """Create a realistic Telco-style churn dataset."""
    print("=" * 70)
    print("1. GENERATING SYNTHETIC DATA")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # --- customer demographics ---
    gender = rng.choice(["Male", "Female"], n)
    senior_citizen = rng.choice([0, 1], n, p=[0.84, 0.16])
    has_partner = rng.choice([0, 1], n, p=[0.52, 0.48])
    has_dependents = rng.choice([0, 1], n, p=[0.70, 0.30])

    # --- account info ---
    tenure = np.clip(rng.normal(32, 20, n).astype(int), 1, 72)
    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # --- services ---
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )
    online_security = np.where(
        internet_service == "No",
        "No internet",
        rng.choice(["Yes", "No"], n, p=[0.35, 0.65]),
    )
    tech_support = np.where(
        internet_service == "No",
        "No internet",
        rng.choice(["Yes", "No"], n, p=[0.35, 0.65]),
    )

    # --- charges ---
    base_monthly = rng.uniform(18, 30, n)
    fiber_add = np.where(internet_service == "Fiber optic", rng.uniform(25, 50, n), 0)
    dsl_add = np.where(internet_service == "DSL", rng.uniform(10, 25, n), 0)
    monthly_charges = np.round(base_monthly + fiber_add + dsl_add, 2)
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 50, n), 2)
    total_charges = np.clip(total_charges, monthly_charges, None)

    # --- support tickets ---
    num_support_tickets = rng.poisson(1.5, n)

    # --- churn label (stronger conditional probabilities + interactions) ---
    churn_logit = np.full(n, -1.2)  # base ≈ 23% in logit space

    # Contract type (dominant signal)
    churn_logit = np.where(contract_type == "Month-to-month", churn_logit + 1.5, churn_logit)
    churn_logit = np.where(contract_type == "Two year", churn_logit - 1.8, churn_logit)

    # Tenure (strong inverse relationship)
    churn_logit += -0.04 * tenure  # longer tenure → less churn
    churn_logit = np.where(tenure < 6, churn_logit + 1.0, churn_logit)

    # Monthly charges (positive correlation)
    churn_logit += 0.015 * monthly_charges

    # Internet & support services
    churn_logit = np.where(internet_service == "Fiber optic", churn_logit + 0.6, churn_logit)
    churn_logit = np.where(online_security == "Yes", churn_logit - 0.7, churn_logit)
    churn_logit = np.where(tech_support == "Yes", churn_logit - 0.7, churn_logit)

    # Payment method
    churn_logit = np.where(payment_method == "Electronic check", churn_logit + 0.5, churn_logit)

    # Support tickets (strong positive)
    churn_logit += 0.25 * num_support_tickets

    # Demographics
    churn_logit = np.where(senior_citizen == 1, churn_logit + 0.3, churn_logit)
    churn_logit = np.where(has_dependents == 1, churn_logit - 0.3, churn_logit)

    # ── Interaction effects ──
    # New customer + month-to-month = very high risk
    churn_logit = np.where(
        (tenure < 12) & (contract_type == "Month-to-month"),
        churn_logit + 0.8, churn_logit
    )
    # High charges + no support = high risk
    churn_logit = np.where(
        (monthly_charges > 65) & (tech_support != "Yes") & (online_security != "Yes"),
        churn_logit + 0.5, churn_logit
    )

    # Add small noise for realism
    churn_logit += rng.normal(0, 0.3, n)

    # Convert logit to probability via sigmoid
    churn_prob = 1 / (1 + np.exp(-churn_logit))

    churned = rng.binomial(1, churn_prob)

    df = pd.DataFrame({
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

    print(f"  → Generated {len(df):,} records")
    print(f"  → Churn rate: {df['churned'].mean():.1%}")
    print(f"  → Features: {df.shape[1] - 1}")
    print()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def run_eda(df: pd.DataFrame) -> None:
    """Generate and save EDA visualizations."""
    print("=" * 70)
    print("2. EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # -- 2a. Distribution plots ------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Feature Distributions by Churn Status", fontsize=16, fontweight="bold", y=1.02)

    num_features = ["tenure", "monthly_charges", "total_charges", "num_support_tickets"]
    cat_features = ["contract_type", "internet_service"]

    for ax, feat in zip(axes.flat[:4], num_features):
        for label, color in [(0, PALETTE[2]), (1, PALETTE[7])]:
            subset = df[df["churned"] == label][feat]
            ax.hist(subset, bins=30, alpha=0.6, label=f"{'Churn' if label else 'Stay'}", color=color, edgecolor="white")
        ax.set_title(feat.replace("_", " ").title(), fontweight="bold")
        ax.legend()

    for ax, feat in zip(axes.flat[4:], cat_features):
        ct = pd.crosstab(df[feat], df["churned"], normalize="index") * 100
        ct.plot(kind="bar", stacked=True, ax=ax, color=[PALETTE[2], PALETTE[7]], edgecolor="white")
        ax.set_title(f"Churn % by {feat.replace('_', ' ').title()}", fontweight="bold")
        ax.set_ylabel("Percentage")
        ax.legend(["Stay", "Churn"], loc="upper right")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_distributions.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")

    # -- 2b. Correlation heatmap -----------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, ax=ax, linewidths=0.5, square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-driven features."""
    print("=" * 70)
    print("3. FEATURE ENGINEERING")
    print("=" * 70)

    df = df.copy()

    # Spending intensity
    df["charge_per_month_of_tenure"] = np.round(df["total_charges"] / (df["tenure"] + 1), 2)
    print("  + charge_per_month_of_tenure")

    # Tenure bins
    df["tenure_bin"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
    )
    print("  + tenure_bin")

    # New customer flag
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    print("  + is_new_customer")

    # Premium support
    df["has_premium_support"] = (
        (df["online_security"] == "Yes") | (df["tech_support"] == "Yes")
    ).astype(int)
    print("  + has_premium_support")

    # High monthly spend
    df["high_monthly_spend"] = (df["monthly_charges"] > 70).astype(int)
    print("  + high_monthly_spend")

    # Ticket rate (normalized complaint frequency)
    df["ticket_rate"] = np.round(df["num_support_tickets"] / (df["tenure"] + 1), 4)
    print("  + ticket_rate")

    # Contract risk score
    contract_risk_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["contract_risk_score"] = df["contract_type"].map(contract_risk_map)
    print("  + contract_risk_score")

    # Auto payment flag
    auto_methods = {"Bank transfer", "Credit card"}
    df["auto_payment"] = df["payment_method"].isin(auto_methods).astype(int)
    print("  + auto_payment")

    print(f"\n  → Total features after engineering: {df.shape[1] - 1}")
    print()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame):
    """Encode, split the data. No scaling needed for tree-based models."""
    print("=" * 70)
    print("4. PREPROCESSING")
    print("=" * 70)

    df = df.copy()
    target = "churned"

    # --- Encode categoricals ---
    label_enc_cols = ["tenure_bin"]
    onehot_cols = ["gender", "contract_type", "payment_method",
                   "internet_service", "online_security", "tech_support"]

    le = LabelEncoder()
    for col in label_enc_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
    print(f"  → Encoded features, shape: {df.shape}")

    # --- Split ---
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  → Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    print(f"  → Class balance: {dict(y_train.value_counts())}")
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    print(f"  → scale_pos_weight will be set to: {neg / pos:.2f}")
    print()

    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODEL TRAINING — XGBoost
# ═══════════════════════════════════════════════════════════════════════════════
def train_model(X_train, y_train):
    """Tune and train an XGBoost classifier via RandomizedSearchCV."""
    print("=" * 70)
    print("5. MODEL TRAINING — XGBoost with RandomizedSearchCV")
    print("=" * 70)

    # Compute scale_pos_weight from training data
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    spw = neg_count / pos_count

    param_distributions = {
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
        "n_estimators": [100, 200, 300, 500],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.3],
        "scale_pos_weight": [1, spw * 0.75, spw, spw * 1.25],
        "reg_alpha": [0, 0.01, 0.1],
        "reg_lambda": [1, 1.5, 2],
    }

    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",
        random_state=42,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=60,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    print("  → Running hyperparameter search (60 iterations × 5 folds) …")
    search.fit(X_train, y_train)

    best = search.best_estimator_
    print(f"\n  → Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"  → Best params:")
    for k, v in sorted(search.best_params_.items()):
        print(f"      {k}: {v}")
    print()

    return best


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, feature_names: list):
    """Comprehensive evaluation with plots and metrics."""
    print("=" * 70)
    print("6. EVALUATION")
    print("=" * 70)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["Stay", "Churn"])

    print(f"\n  Accuracy      : {acc:.4f}")
    print(f"  F1 (Churn)    : {f1:.4f}")
    print(f"  ROC-AUC       : {roc_auc:.4f}")
    print(f"  Avg Precision : {ap:.4f}")
    print(f"\n{report}")

    # --- Save text report ---
    report_path = os.path.join(OUTPUT_DIR, "model_report.txt")
    with open(report_path, "w") as f:
        f.write("Customer Churn Prediction — Model Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy           : {acc:.4f}\n")
        f.write(f"F1 Score (Churn)   : {f1:.4f}\n")
        f.write(f"ROC-AUC            : {roc_auc:.4f}\n")
        f.write(f"Average Precision  : {ap:.4f}\n\n")
        f.write("Classification Report\n")
        f.write("-" * 50 + "\n")
        f.write(report + "\n\n")
        f.write("Best Hyperparameters\n")
        f.write("-" * 50 + "\n")
        for k, v in sorted(model.get_params().items()):
            if k not in ("callbacks", "kwargs"):
                f.write(f"  {k}: {v}\n")
    print(f"  ✓ Saved: {report_path}")

    # -- 6a. Confusion matrix --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"],
        linewidths=1, linecolor="white",
        annot_kws={"size": 16},
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold", pad=12)
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")

    # -- 6b. ROC curve ----------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=PALETTE[7], lw=2.5, label=f"XGBoost (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE[7])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")

    # -- 6c. Precision-Recall curve --------------------------------------------
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color=PALETTE[4], lw=2.5, label=f"XGBoost (AP = {ap:.3f})")
    ax.fill_between(recall, precision, alpha=0.15, color=PALETTE[4])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision–Recall Curve", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    path = os.path.join(OUTPUT_DIR, "precision_recall_curve.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")

    # -- 6d. Feature importance (gain) -----------------------------------------
    importances = model.get_booster().get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame.from_dict(importances, orient="index", columns=["gain"])
        .sort_values("gain", ascending=True)
        .tail(15)
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(imp_df.index, imp_df["gain"], color=PALETTE[5], edgecolor="white", height=0.6)
    ax.set_xlabel("Gain", fontsize=12)
    ax.set_title("Top-15 Feature Importance (XGBoost Gain)", fontsize=15, fontweight="bold")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + max(imp_df["gain"]) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{width:.0f}", va="center", fontsize=9)
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    print()

    return {"accuracy": acc, "f1": f1, "roc_auc": roc_auc, "avg_precision": ap}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SHAP INTERPRETABILITY
# ═══════════════════════════════════════════════════════════════════════════════
def shap_analysis(model, X_test, feature_names: list):
    """Generate SHAP summary plot for global interpretability."""
    print("=" * 70)
    print("7. SHAP INTERPRETABILITY")
    print("=" * 70)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, max_display=15)
    plt.title("SHAP Summary — Feature Impact on Churn Prediction", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shap_summary.png")
    plt.savefig(path)
    plt.close("all")
    print(f"  ✓ Saved: {path}")

    # Top-10 features by mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top10 = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)[:10]
    print("\n  Top-10 most impactful features (mean |SHAP|):")
    for i, (feat, val) in enumerate(top10, 1):
        print(f"    {i:2d}. {feat:<35s} {val:.4f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "█" * 70)
    print("  CUSTOMER CHURN PREDICTION PIPELINE")
    print("█" * 70 + "\n")

    # Step 1 — Data
    df = generate_data()

    # Step 2 — EDA
    run_eda(df)

    # Step 3 — Feature engineering
    df = engineer_features(df)

    # Step 4 — Preprocessing
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)

    # Step 5 — Training
    model = train_model(X_train, y_train)

    # Step 6 — Evaluation
    metrics = evaluate_model(model, X_test, y_test, feature_names)

    # Step 7 — SHAP
    shap_analysis(model, X_test, feature_names)

    # Summary
    print("=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  ROC-AUC       : {metrics['roc_auc']:.4f}")
    print(f"  F1 (Churn)    : {metrics['f1']:.4f}")
    print(f"  Avg Precision : {metrics['avg_precision']:.4f}")
    print(f"\n  All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
