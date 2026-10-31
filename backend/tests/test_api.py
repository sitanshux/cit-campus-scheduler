"""
API integration tests.

Tests that the FastAPI endpoints work correctly by directly importing
and calling the endpoint functions (avoiding TestClient compatibility issues).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app
from app.scheduling.routes import generate_schedule


class TestExistingEndpoints:
    """Verify existing endpoint functions still work."""

    def test_root_endpoint(self):
        from app.main import root
        result = root()
        assert result["status"] == "healthy"

    def test_health_endpoint(self):
        from app.main import health_check
        result = health_check()
        assert result["status"] == "healthy"

    def test_dataset_summary_endpoint(self):
        from app.main import dataset_summary
        result = dataset_summary()
        assert result["dataset_id"] == "cit-public-v1"
        assert result["counts"]["sessions"] == 189

    def test_dataset_info_endpoint(self):
        from app.main import dataset_info
        result = dataset_info()
        assert "calendar" in result
        assert result["dataset_id"] == "cit-public-v1"


class TestScheduleEndpoint:
    """Test the schedule generation endpoint function."""

    def test_generate_returns_result(self):
        result = generate_schedule(time_limit_seconds=120)
        assert "success" in result
        assert "status" in result
        assert "scheduled" in result
        assert "unscheduled" in result
        assert "diagnostics" in result
        assert "solve_time_seconds" in result

    def test_generate_schedules_all(self):
        result = generate_schedule(time_limit_seconds=120)
        assert result["status"] in ("OPTIMAL", "FEASIBLE")
        assert result["diagnostics"]["scheduled_count"] == 189
        assert result["success"] is True

    def test_schedule_route_registered(self):
        """Verify the route is registered on the app via OpenAPI schema."""
        openapi_schema = app.openapi()
        assert "/api/schedule/generate" in openapi_schema["paths"]
