"""
Customer Churn Prediction — Inference Engine
============================================
Handles artifact loading, feature preprocessing alignment, vectorized
probability scoring, risk factor diagnosis, and retention advice generation.
"""

import os
from typing import List, Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd

from schemas import (
    CustomerInput,
    PredictionResponse,
    BatchCustomerInput,
    BatchPredictionResponse,
    BatchSummary,
    ModelInfoResponse,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_ARTIFACT_PATH = os.path.join(OUTPUT_DIR, "churn_model.joblib")


class ChurnPredictor:
    """Production inference engine for Customer Churn Prediction."""

    def __init__(self, artifact_path: str = MODEL_ARTIFACT_PATH):
        self.artifact_path = artifact_path
        self._load_or_train()

    def _load_or_train(self):
        """Load serialized bundle, or trigger training if artifact not present."""
        if not os.path.exists(self.artifact_path):
            print(f"[Predictor] Artifact not found at {self.artifact_path}. Exporting now...")
            import export_model
            self.bundle = export_model.train_and_export()
        else:
            self.bundle = joblib.load(self.artifact_path)

        self.model = self.bundle["model"]
        self.feature_names = self.bundle["feature_names"]
        self.decision_threshold = self.bundle.get("decision_threshold", 0.5)
        self.metrics = self.bundle.get("metrics", {})
        self.top_features = self.bundle.get("top_features", [])
        self.algorithm = self.bundle.get("algorithm", "XGBoost Classifier")
        self.version = self.bundle.get("version", "1.0.0")
        self.trained_at = self.bundle.get("trained_at", "")
        self.label_classes = self.bundle.get("label_encoder_classes", ["0-12m", "13-24m", "25-48m", "49-72m"])
        self.onehot_cols = self.bundle.get(
            "onehot_cols",
            ["gender", "contract_type", "payment_method", "internet_service", "online_security", "tech_support"]
        )

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply domain feature transformations matching model training."""
        df = df.copy()
        
        # Spending intensity
        df["charge_per_month_of_tenure"] = np.round(
            df["total_charges"] / (df["tenure"] + 1), 2
        )

        # Tenure bins
        df["tenure_bin"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 120],
            labels=["0-12m", "13-24m", "25-48m", "49-72m"],
        )

        # Flags & risk metrics
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

        # Label encode tenure_bin
        bin_to_code = {cls: idx for idx, cls in enumerate(self.label_classes)}
        df["tenure_bin"] = df["tenure_bin"].astype(str).map(bin_to_code).fillna(0).astype(int)

        return df

    def transform_to_features(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert a list of raw customer dictionaries to the exact XGBoost feature matrix."""
        df_raw = pd.DataFrame(records)
        df_feats = self._engineer_features(df_raw)

        # One-hot encode categoricals
        df_encoded = pd.get_dummies(df_feats, columns=self.onehot_cols, drop_first=True)

        # Align with training features — guarantees zero-fill for unseen / single-row schemas
        df_aligned = df_encoded.reindex(columns=self.feature_names, fill_value=0)
        return df_aligned

    @staticmethod
    def diagnose_risk(cust: Dict[str, Any], proba: float) -> Tuple[List[str], List[str]]:
        """Identify key drivers for customer risk and generate retention interventions."""
        risk_factors = []
        actions = []

        # Contract risk
        if cust.get("contract_type") == "Month-to-month":
            risk_factors.append("Month-to-month contract (high churn vulnerability)")
            actions.append("Offer 15% discount incentive to upgrade to a 1-year or 2-year contract.")

        # Tenure risk
        tenure = cust.get("tenure", 0)
        if tenure <= 6:
            risk_factors.append(f"Early customer lifecycle ({tenure} months tenure)")
            actions.append("Trigger VIP onboarding outreach and customer success check-in.")

        # Support tickets
        tickets = cust.get("num_support_tickets", 0)
        if tickets >= 3:
            risk_factors.append(f"Frequent support tickets ({tickets} tickets filed)")
            actions.append("Priority ticket escalation: Assign dedicated Tier-2 specialist to resolve open issues.")
        elif tickets == 0 and proba > 0.4:
            risk_factors.append("Zero support interactions (potential silent disengagement)")
            actions.append("Send proactive feature usage tips or satisfaction survey.")

        # Payment friction
        if cust.get("payment_method") == "Electronic check":
            risk_factors.append("Payment method: Electronic check (higher friction/delinquency rate)")
            actions.append("Promote $5/mo bill credit for enabling automated card or bank payments.")

        # Spend & Service bundle
        monthly = cust.get("monthly_charges", 0.0)
        if monthly > 75.0:
            risk_factors.append(f"High monthly spend (${monthly:.2f}/mo)")
            if cust.get("tech_support") != "Yes" or cust.get("online_security") != "Yes":
                risk_factors.append("High charges without bundled protection/support")
                actions.append("Offer complimentary 6-month trial of Tech Support and Online Security.")

        # Fallback if low risk
        if not risk_factors:
            risk_factors.append("Standard tenure and stable engagement profile")
            actions.append("Maintain standard marketing communication; monitor for usage drops.")

        return risk_factors, actions

    def predict_single(self, customer: CustomerInput) -> PredictionResponse:
        """Perform real-time scoring for an individual customer."""
        cust_dict = customer.model_dump()
        df_matrix = self.transform_to_features([cust_dict])

        proba = float(self.model.predict_proba(df_matrix)[:, 1][0])
        is_churn = bool(proba >= self.decision_threshold)

        if proba >= 0.70:
            risk_tier = "High Risk"
        elif proba >= 0.35:
            risk_tier = "Medium Risk"
        else:
            risk_tier = "Low Risk"

        factors, actions = self.diagnose_risk(cust_dict, proba)

        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=round(proba, 4),
            churn_prediction=is_churn,
            risk_tier=risk_tier,
            key_risk_factors=factors,
            recommended_retention_actions=actions,
        )

    def predict_batch(self, batch: BatchCustomerInput) -> BatchPredictionResponse:
        """Vectorized bulk scoring for multiple customers."""
        cust_dicts = [c.model_dump() for c in batch.customers]
        df_matrix = self.transform_to_features(cust_dicts)

        probas = self.model.predict_proba(df_matrix)[:, 1]

        predictions: List[PredictionResponse] = []
        high_risk_count = 0
        med_risk_count = 0
        low_risk_count = 0
        churn_count = 0

        for cust_dict, proba in zip(cust_dicts, probas):
            p = float(proba)
            is_churn = bool(p >= self.decision_threshold)
            if is_churn:
                churn_count += 1

            if p >= 0.70:
                tier = "High Risk"
                high_risk_count += 1
            elif p >= 0.35:
                tier = "Medium Risk"
                med_risk_count += 1
            else:
                tier = "Low Risk"
                low_risk_count += 1

            factors, actions = self.diagnose_risk(cust_dict, p)

            predictions.append(
                PredictionResponse(
                    customer_id=cust_dict.get("customer_id"),
                    churn_probability=round(p, 4),
                    churn_prediction=is_churn,
                    risk_tier=tier,
                    key_risk_factors=factors,
                    recommended_retention_actions=actions,
                )
            )

        summary = BatchSummary(
            total_customers=len(predictions),
            churn_predicted_count=churn_count,
            high_risk_count=high_risk_count,
            medium_risk_count=med_risk_count,
            low_risk_count=low_risk_count,
            average_churn_probability=round(float(np.mean(probas)), 4),
        )

        return BatchPredictionResponse(summary=summary, predictions=predictions)

    def get_metadata(self) -> ModelInfoResponse:
        """Return model metadata, performance metrics, and top gain features."""
        return ModelInfoResponse(
            model_name="Customer Churn Predictor",
            algorithm=self.algorithm,
            version=self.version,
            trained_at=self.trained_at,
            decision_threshold=self.decision_threshold,
            metrics=self.metrics,
            feature_count=len(self.feature_names),
            top_features=self.top_features,
        )
