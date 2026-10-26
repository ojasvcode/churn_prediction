"""
Customer Churn Prediction — Pydantic Validation & API Schemas
==============================================================
Defines the request and response models for single and batch predictions,
model metadata, and retention recommendations.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator


class CustomerInput(BaseModel):
    """Customer profile features required for churn inference."""
    customer_id: Optional[str] = Field(None, description="Unique customer ID (e.g., CUST-1049)")
    
    # Demographics
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    senior_citizen: int = Field(0, ge=0, le=1, description="1 if customer is a senior citizen, 0 otherwise")
    has_partner: int = Field(0, ge=0, le=1, description="1 if customer has a partner, 0 otherwise")
    has_dependents: int = Field(0, ge=0, le=1, description="1 if customer has dependents, 0 otherwise")
    
    # Account & Contract
    tenure: int = Field(..., ge=0, le=120, description="Customer tenure in months")
    contract_type: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Contract duration type"
    )
    payment_method: Literal[
        "Electronic check", "Mailed check", "Bank transfer", "Credit card"
    ] = Field(..., description="Billing payment method")
    
    # Services
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Internet service technology"
    )
    online_security: Literal["Yes", "No", "No internet"] = Field(
        ..., description="Online security add-on status"
    )
    tech_support: Literal["Yes", "No", "No internet"] = Field(
        ..., description="Tech support add-on status"
    )
    
    # Billing & Usage
    monthly_charges: float = Field(..., gt=0.0, description="Monthly recurring charge in USD")
    total_charges: Optional[float] = Field(
        None, ge=0.0, description="Total lifetime charges in USD (defaults to monthly_charges * tenure)"
    )
    num_support_tickets: int = Field(
        0, ge=0, description="Total technical/billing support tickets opened"
    )

    @model_validator(mode="after")
    def populate_total_charges(self) -> "CustomerInput":
        """If total_charges is omitted or 0 for existing customers, estimate it."""
        if self.total_charges is None:
            self.total_charges = round(self.monthly_charges * max(self.tenure, 1), 2)
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "CUST-9821",
                "gender": "Female",
                "senior_citizen": 0,
                "has_partner": 0,
                "has_dependents": 0,
                "tenure": 3,
                "contract_type": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "tech_support": "No",
                "monthly_charges": 85.50,
                "total_charges": 256.50,
                "num_support_tickets": 4,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Prediction result for a single customer."""
    customer_id: Optional[str] = Field(None, description="Customer identifier if provided")
    churn_probability: float = Field(..., description="Estimated churn probability between 0.0 and 1.0")
    churn_prediction: bool = Field(..., description="True if customer is predicted to churn")
    risk_tier: Literal["Low Risk", "Medium Risk", "High Risk"] = Field(
        ..., description="Categorical risk rating based on probability"
    )
    key_risk_factors: List[str] = Field(
        default_factory=list, description="Top behavioral or contract factors contributing to risk"
    )
    recommended_retention_actions: List[str] = Field(
        default_factory=list, description="Targeted proactive retention steps"
    )


class BatchCustomerInput(BaseModel):
    """Batch input wrapper for scoring multiple customers simultaneously."""
    customers: List[CustomerInput] = Field(..., min_length=1, description="List of customer profiles")

    model_config = {
        "json_schema_extra": {
            "example": {
                "customers": [
                    {
                        "customer_id": "CUST-001",
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
                        "monthly_charges": 92.0,
                        "num_support_tickets": 3,
                    },
                    {
                        "customer_id": "CUST-002",
                        "gender": "Male",
                        "senior_citizen": 0,
                        "has_partner": 1,
                        "has_dependents": 1,
                        "tenure": 48,
                        "contract_type": "Two year",
                        "payment_method": "Credit card",
                        "internet_service": "DSL",
                        "online_security": "Yes",
                        "tech_support": "Yes",
                        "monthly_charges": 45.0,
                        "num_support_tickets": 0,
                    },
                ]
            }
        }
    }


class BatchSummary(BaseModel):
    """Aggregate statistics for a batch scoring run."""
    total_customers: int
    churn_predicted_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    average_churn_probability: float


class BatchPredictionResponse(BaseModel):
    """Response wrapper for batch scoring results."""
    summary: BatchSummary
    predictions: List[PredictionResponse]


class ModelInfoResponse(BaseModel):
    """Metadata regarding the loaded machine learning model."""
    model_name: str
    algorithm: str
    version: str
    trained_at: str
    decision_threshold: float
    metrics: dict
    feature_count: int
    top_features: List[str]
