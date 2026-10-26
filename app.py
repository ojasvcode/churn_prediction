"""
Customer Churn Prediction — FastAPI REST Application
====================================================
Exposes REST endpoints for real-time customer churn scoring, batch processing,
CSV bulk prediction, model performance metadata, and proactive retention actions.
"""

from contextlib import asynccontextmanager
import io
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import (
    CustomerInput,
    PredictionResponse,
    BatchCustomerInput,
    BatchPredictionResponse,
    ModelInfoResponse,
)
from predictor import ChurnPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact into memory upon service startup."""
    print("🚀 Initializing Customer Churn Predictor service...")
    predictor = ChurnPredictor()
    app.state.predictor = predictor
    print(f"✓ Model loaded: {predictor.algorithm} (ROC-AUC: {predictor.metrics.get('roc_auc', 'N/A')})")
    yield
    print("🛑 Shutting down Customer Churn Predictor service.")


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production REST service for real-time customer churn scoring, bulk predictions, "
        "risk factor diagnostics, and automated retention recommendations powered by XGBoost."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """API entry point with status and documentation links."""
    return {
        "service": "Customer Churn Prediction API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "health_check": "/health",
        "endpoints": {
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "predict_csv": "POST /predict/csv",
            "model_metadata": "GET /model/info",
        },
    }


@app.get("/health", tags=["General"])
async def health():
    """Liveness and readiness probe for containerized environments."""
    predictor: ChurnPredictor = getattr(app.state, "predictor", None)
    is_ready = predictor is not None and predictor.model is not None
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": is_ready,
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model Info"])
async def model_info():
    """Return model architecture, training performance metrics, and top gain features."""
    predictor: ChurnPredictor = getattr(app.state, "predictor", None)
    if not predictor:
        raise HTTPException(status_code=503, detail="Model is not yet loaded.")
    return predictor.get_metadata()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
)
async def predict_single(customer: CustomerInput):
    """
    Score an individual customer for churn risk.
    
    Returns:
    - **churn_probability**: Probability estimate (0.0 to 1.0)
    - **churn_prediction**: Boolean flag (True = likely to churn)
    - **risk_tier**: 'Low Risk', 'Medium Risk', or 'High Risk'
    - **key_risk_factors**: Top contributing drivers
    - **recommended_retention_actions**: Targeted proactive retention strategies
    """
    predictor: ChurnPredictor = getattr(app.state, "predictor", None)
    if not predictor:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    
    try:
        return predictor.predict_single(customer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
)
async def predict_batch(batch: BatchCustomerInput):
    """
    Vectorized bulk prediction for a list of customers.
    
    Provides per-customer predictions and an aggregate summary of risk distributions.
    """
    predictor: ChurnPredictor = getattr(app.state, "predictor", None)
    if not predictor:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    try:
        return predictor.predict_batch(batch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")


@app.post("/predict/csv", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file containing customer rows, run inference, and return scored results.
    """
    predictor: ChurnPredictor = getattr(app.state, "predictor", None)
    if not predictor:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

    # Convert DataFrame records to CustomerInput objects
    customers = []
    for idx, row in df.iterrows():
        row_dict = row.dropna().to_dict()
        try:
            customers.append(CustomerInput(**row_dict))
        except Exception as err:
            raise HTTPException(
                status_code=422,
                detail=f"Row {idx + 1} validation failed: {str(err)}",
            )

    batch_input = BatchCustomerInput(customers=customers)
    return predictor.predict_batch(batch_input)
