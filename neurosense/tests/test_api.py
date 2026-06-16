"""NeuroSense Unit Tests — API Schemas and Endpoints.

Tests for Pydantic schemas and FastAPI endpoint validation:
- ClinicalInput field validation and range constraints
- PredictionResponse serialisation
- API endpoint integration tests
"""

from __future__ import annotations

import pytest


# ═════════════════════════════════════════════════════════════════
#  Schema Tests
# ═════════════════════════════════════════════════════════════════


class TestClinicalInput:
    """Test suite for ClinicalInput Pydantic schema."""

    def test_valid_input(self):
        """Valid clinical data creates ClinicalInput successfully."""
        from neurosense.api.schemas import ClinicalInput

        data = ClinicalInput(
            cag_repeat=44.0,
            uhdrs_motor=18.0,
            uhdrs_cognitive=142.0,
            tfc_score=9.0,
            age=42.0,
        )

        assert data.cag_repeat == 44.0
        assert data.uhdrs_motor == 18.0
        assert data.age == 42.0

    def test_cag_below_threshold_raises(self):
        """CAG repeat below 36 raises validation error."""
        from pydantic import ValidationError

        from neurosense.api.schemas import ClinicalInput

        with pytest.raises(ValidationError):
            ClinicalInput(
                cag_repeat=30.0,  # Below HD threshold
                uhdrs_motor=18.0,
                uhdrs_cognitive=142.0,
                age=42.0,
            )

    def test_uhdrs_motor_above_max_raises(self):
        """UHDRS motor score above 124 raises validation error."""
        from pydantic import ValidationError

        from neurosense.api.schemas import ClinicalInput

        with pytest.raises(ValidationError):
            ClinicalInput(
                cag_repeat=44.0,
                uhdrs_motor=130.0,  # Above max
                uhdrs_cognitive=142.0,
                age=42.0,
            )

    def test_age_range_validation(self):
        """Age outside 18-90 range raises validation error."""
        from pydantic import ValidationError

        from neurosense.api.schemas import ClinicalInput

        with pytest.raises(ValidationError):
            ClinicalInput(
                cag_repeat=44.0,
                uhdrs_motor=18.0,
                uhdrs_cognitive=142.0,
                age=10.0,  # Too young
            )

    def test_tfc_defaults_to_13(self):
        """TFC score defaults to 13.0 when not provided."""
        from neurosense.api.schemas import ClinicalInput

        data = ClinicalInput(
            cag_repeat=44.0,
            uhdrs_motor=18.0,
            uhdrs_cognitive=142.0,
            age=42.0,
        )

        assert data.tfc_score == 13.0

    def test_to_tensor_list(self):
        """to_tensor_list() returns correct ordered values."""
        from neurosense.api.schemas import ClinicalInput

        data = ClinicalInput(
            cag_repeat=44.0,
            uhdrs_motor=18.0,
            uhdrs_cognitive=142.0,
            tfc_score=9.0,
            age=42.0,
        )

        result = data.to_tensor_list()

        assert result == [44.0, 18.0, 142.0, 9.0, 42.0]
        assert len(result) == 5


class TestPredictionResponse:
    """Test suite for PredictionResponse schema."""

    def test_serialisation(self):
        """PredictionResponse serialises to JSON correctly."""
        from neurosense.api.schemas import (
            PredictionResponse,
            StageProbabilities,
        )

        response = PredictionResponse(
            stage="early",
            confidence=0.87,
            stage_probabilities=StageProbabilities(
                pre_manifest=0.08,
                early=0.87,
                advanced=0.05,
            ),
            progression_12mo=4.2,
            progression_24mo=9.1,
            risk_category="medium",
            processing_time_s=12.4,
        )

        data = response.model_dump()

        assert data["stage"] == "early"
        assert data["confidence"] == 0.87
        assert data["risk_category"] == "medium"
        assert data["stage_probabilities"]["early"] == 0.87

    def test_shap_features(self):
        """PredictionResponse includes SHAP features."""
        from neurosense.api.schemas import (
            PredictionResponse,
            SHAPFeature,
            StageProbabilities,
        )

        features = [
            SHAPFeature(name="cag_repeat", value=44.0, impact=0.32),
            SHAPFeature(name="uhdrs_motor", value=18.0, impact=0.28),
        ]

        response = PredictionResponse(
            stage="early",
            confidence=0.87,
            stage_probabilities=StageProbabilities(
                pre_manifest=0.08,
                early=0.87,
                advanced=0.05,
            ),
            progression_12mo=4.2,
            progression_24mo=9.1,
            risk_category="medium",
            shap_features=features,
            processing_time_s=12.4,
        )

        assert len(response.shap_features) == 2
        assert response.shap_features[0].name == "cag_repeat"


class TestHealthResponse:
    """Test suite for HealthResponse schema."""

    def test_healthy_response(self):
        """HealthResponse creates valid health status."""
        from neurosense.api.schemas import HealthResponse

        health = HealthResponse(
            status="healthy",
            gpu_available=False,
            model_loaded=True,
            uptime_seconds=120.5,
            version="1.0.0",
        )

        assert health.status == "healthy"
        assert health.model_loaded is True


# ═════════════════════════════════════════════════════════════════
#  API Endpoint Integration Tests
# ═════════════════════════════════════════════════════════════════


class TestAPIEndpoints:
    """Integration tests for FastAPI endpoints using TestClient."""

    @pytest.fixture
    def client(self):
        """Create a FastAPI test client."""
        from fastapi.testclient import TestClient

        from neurosense.api.main import app

        return TestClient(app)

    def test_root_endpoint(self, client):
        """Root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "NeuroSense API"

    def test_health_endpoint(self, client):
        """Health endpoint returns valid status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "gpu_available" in data
        assert "model_loaded" in data

    def test_version_endpoint(self, client):
        """Version endpoint returns version info."""
        response = client.get("/version")

        assert response.status_code == 200
        data = response.json()
        assert "api_version" in data
        assert "python_version" in data
        assert "torch_version" in data
