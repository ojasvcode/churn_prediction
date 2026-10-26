"""
Customer Churn Prediction API — Automated Test Suite
=====================================================
Validates all REST endpoints, data validation schemas, edge cases,
batch processing, and CSV upload functionality.
"""

import io
import unittest
from fastapi.testclient import TestClient
from app import app


class TestChurnAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize TestClient within the FastAPI lifespan context."""
        cls.client = TestClient(app)
        # Enter lifespan context
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        """Exit TestClient lifespan context."""
        cls.client.__exit__(None, None, None)

    def test_root_endpoint(self):
        """Test GET / returns online status and endpoint map."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("endpoints", data)

    def test_health_endpoint(self):
        """Test GET /health returns healthy status."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_model_info_endpoint(self):
        """Test GET /model/info returns model metrics and top features."""
        res = self.client.get("/model/info")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["model_name"], "Customer Churn Predictor")
        self.assertIn("roc_auc", data["metrics"])
        self.assertGreater(data["metrics"]["roc_auc"], 0.75)
        self.assertIsInstance(data["top_features"], list)
        self.assertGreater(len(data["top_features"]), 0)

    def test_predict_single_high_risk_customer(self):
        """Test single prediction on an inherently high-churn customer profile."""
        payload = {
            "customer_id": "TEST-HIGH-01",
            "gender": "Female",
            "senior_citizen": 1,
            "has_partner": 0,
            "has_dependents": 0,
            "tenure": 2,
            "contract_type": "Month-to-month",
            "payment_method": "Electronic check",
            "internet_service": "Fiber optic",
            "online_security": "No",
            "tech_support": "No",
            "monthly_charges": 95.0,
            "total_charges": 190.0,
            "num_support_tickets": 4,
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["customer_id"], "TEST-HIGH-01")
        self.assertTrue(data["churn_prediction"])
        self.assertGreaterEqual(data["churn_probability"], 0.50)
        self.assertIn(data["risk_tier"], ["High Risk", "Medium Risk"])
        self.assertGreater(len(data["key_risk_factors"]), 0)
        self.assertGreater(len(data["recommended_retention_actions"]), 0)

    def test_predict_single_low_risk_customer(self):
        """Test single prediction on an inherently loyal customer profile."""
        payload = {
            "customer_id": "TEST-LOW-01",
            "gender": "Male",
            "senior_citizen": 0,
            "has_partner": 1,
            "has_dependents": 1,
            "tenure": 60,
            "contract_type": "Two year",
            "payment_method": "Credit card",
            "internet_service": "DSL",
            "online_security": "Yes",
            "tech_support": "Yes",
            "monthly_charges": 42.0,
            "total_charges": 2520.0,
            "num_support_tickets": 0,
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["customer_id"], "TEST-LOW-01")
        self.assertFalse(data["churn_prediction"])
        self.assertLess(data["churn_probability"], 0.40)
        self.assertEqual(data["risk_tier"], "Low Risk")

    def test_predict_batch(self):
        """Test batch prediction scoring multiple customers simultaneously."""
        payload = {
            "customers": [
                {
                    "customer_id": "BATCH-01",
                    "gender": "Female",
                    "senior_citizen": 0,
                    "has_partner": 0,
                    "has_dependents": 0,
                    "tenure": 1,
                    "contract_type": "Month-to-month",
                    "payment_method": "Electronic check",
                    "internet_service": "Fiber optic",
                    "online_security": "No",
                    "tech_support": "No",
                    "monthly_charges": 90.0,
                    "num_support_tickets": 3,
                },
                {
                    "customer_id": "BATCH-02",
                    "gender": "Male",
                    "senior_citizen": 0,
                    "has_partner": 1,
                    "has_dependents": 1,
                    "tenure": 50,
                    "contract_type": "Two year",
                    "payment_method": "Bank transfer",
                    "internet_service": "DSL",
                    "online_security": "Yes",
                    "tech_support": "Yes",
                    "monthly_charges": 35.0,
                    "num_support_tickets": 0,
                },
            ]
        }
        res = self.client.post("/predict/batch", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        summary = data["summary"]
        self.assertEqual(summary["total_customers"], 2)
        self.assertEqual(len(data["predictions"]), 2)
        self.assertGreaterEqual(summary["average_churn_probability"], 0.0)
        self.assertLessEqual(summary["average_churn_probability"], 1.0)

    def test_validation_error_invalid_contract(self):
        """Test that invalid categorical values trigger HTTP 422 Unprocessable Entity."""
        payload = {
            "gender": "Female",
            "senior_citizen": 0,
            "has_partner": 0,
            "has_dependents": 0,
            "tenure": 10,
            "contract_type": "Lifetime",  # Invalid!
            "payment_method": "Credit card",
            "internet_service": "DSL",
            "online_security": "Yes",
            "tech_support": "Yes",
            "monthly_charges": 50.0,
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 422)

    def test_predict_csv_upload(self):
        """Test uploading a CSV file to POST /predict/csv."""
        csv_content = (
            "customer_id,gender,senior_citizen,has_partner,has_dependents,tenure,"
            "contract_type,payment_method,internet_service,online_security,tech_support,"
            "monthly_charges,total_charges,num_support_tickets\n"
            "CSV-01,Female,0,0,0,3,Month-to-month,Electronic check,Fiber optic,No,No,85.0,255.0,3\n"
            "CSV-02,Male,0,1,1,55,Two year,Credit card,DSL,Yes,Yes,45.0,2475.0,0\n"
        )
        csv_file = io.BytesIO(csv_content.encode("utf-8"))
        res = self.client.post(
            "/predict/csv",
            files={"file": ("test_customers.csv", csv_file, "text/csv")},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["summary"]["total_customers"], 2)
        self.assertEqual(len(data["predictions"]), 2)
        self.assertEqual(data["predictions"][0]["customer_id"], "CSV-01")
        self.assertEqual(data["predictions"][1]["customer_id"], "CSV-02")


if __name__ == "__main__":
    unittest.main()
