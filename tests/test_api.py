"""Tests for FastAPI endpoints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


class TestHealthEndpoints:

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_status(self, client):
        r = client.get("/status")
        assert r.status_code == 200
        assert "data_loaded" in r.json()


class TestPipeline:

    def test_load_sample(self, client):
        r = client.post("/api/v1/ingest/load-sample")
        assert r.status_code == 200
        assert r.json()["status"] == "loaded"
        assert "course" in r.json()["datasets"]

    def test_clean(self, client):
        # Ensure data is loaded first
        client.post("/api/v1/ingest/load-sample")
        r = client.post("/api/v1/pipeline/clean")
        assert r.status_code == 200
        assert r.json()["status"] == "cleaned"

    def test_transform(self, client):
        client.post("/api/v1/ingest/load-sample")
        client.post("/api/v1/pipeline/clean")
        r = client.post("/api/v1/pipeline/transform")
        assert r.status_code == 200
        assert r.json()["status"] == "transformed"
        assert "course_activity" in r.json()["tables"]


class TestAnalytics:

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, client):
        """Run full pipeline before analytics tests."""
        client.post("/api/v1/pipeline/run-all")

    def test_learner_analytics(self, client):
        r = client.get("/api/v1/analytics/learner")
        assert r.status_code == 200
        assert "completion" in r.json()

    def test_course_analytics(self, client):
        r = client.get("/api/v1/analytics/course")
        assert r.status_code == 200
        assert "completion_rates" in r.json()

    def test_engagement_compute(self, client):
        r = client.post("/api/v1/analytics/engagement/compute")
        assert r.status_code == 200
        assert "distribution" in r.json()

    def test_engagement_get(self, client):
        client.post("/api/v1/analytics/engagement/compute")
        r = client.get("/api/v1/analytics/engagement")
        assert r.status_code == 200
        assert "scores" in r.json()

    def test_skill_gaps(self, client):
        r = client.get("/api/v1/analytics/skills/gaps")
        assert r.status_code == 200
        assert "gaps" in r.json()

    def test_insights(self, client):
        r = client.post("/api/v1/insights/generate")
        assert r.status_code == 200
        assert r.json()["total_insights"] > 0

    def test_dashboard_kpis(self, client):
        r = client.get("/api/v1/dashboard/kpis")
        assert r.status_code == 200
        data = r.json()
        assert "total_learners" in data
        assert "overall_completion_rate" in data


class TestEdgeCases:

    def test_analytics_without_pipeline(self):
        """Fresh client — analytics should fail gracefully."""
        from api.main import store, DataStore
        # Reset store
        original = store.__dict__.copy()
        store.__init__()

        fresh_client = TestClient(app)
        r = fresh_client.get("/api/v1/analytics/learner")
        assert r.status_code == 400

        # Restore
        store.__dict__.update(original)

    def test_nonexistent_learner(self, client):
        client.post("/api/v1/pipeline/run-all")
        r = client.get("/api/v1/analytics/learner/nonexistent@test.com")
        assert r.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
