"""Tests for analytics engines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np

from pipeline.ingestion import DataIngestor
from pipeline.cleaning import DataCleaner
from pipeline.transformation import DataTransformer
from analytics.learner import LearnerAnalytics
from analytics.course import CourseAnalytics
from analytics.program import ProgramAnalytics
from analytics.video import VideoAnalytics
from analytics.engagement import EngagementScoringEngine
from analytics.skills import SkillIntelligenceEngine
from analytics.predictive import PredictiveAnalytics
from analytics.insights import InsightEngine
from core.config import settings


@pytest.fixture(scope="module")
def transformed_data():
    """Load and transform sample data once for all tests."""
    ingestor = DataIngestor()
    raw = ingestor.ingest_all(settings.data_dir)
    cleaner = DataCleaner()
    cleaned = {k: cleaner.clean(v) for k, v in raw.items()}
    transformer = DataTransformer()
    return transformer.transform_all(cleaned)


@pytest.fixture
def course_activity(transformed_data):
    return transformed_data["course_activity"]


@pytest.fixture
def video_activity(transformed_data):
    return transformed_data.get("video_activity", pd.DataFrame())


@pytest.fixture
def program_activity(transformed_data):
    return transformed_data.get("program_activity", pd.DataFrame())


# ── Learner Analytics ───────────────────────────────────────────────

class TestLearnerAnalytics:

    def test_active_vs_inactive(self, course_activity, transformed_data):
        engine = LearnerAnalytics(transformed_data["learners"], course_activity)
        result = engine.active_vs_inactive()
        assert "active" in result
        assert "inactive" in result
        assert result["active"] + result["inactive"] == result["total"]

    def test_completion_summary(self, course_activity, transformed_data):
        engine = LearnerAnalytics(transformed_data["learners"], course_activity)
        result = engine.completion_summary()
        assert 0 <= result["completion_rate"] <= 100
        assert result["total_enrollments"] > 0

    def test_top_learners(self, course_activity, transformed_data):
        engine = LearnerAnalytics(transformed_data["learners"], course_activity)
        top = engine.top_learners(5)
        assert len(top) <= 5
        assert "score" in top.columns

    def test_learning_velocity(self, course_activity, transformed_data):
        engine = LearnerAnalytics(transformed_data["learners"], course_activity)
        velocity = engine.learning_velocity()
        assert not velocity.empty
        assert "hours_per_week" in velocity.columns


# ── Course Analytics ────────────────────────────────────────────────

class TestCourseAnalytics:

    def test_completion_rates(self, course_activity):
        engine = CourseAnalytics(course_activity)
        rates = engine.completion_rates()
        assert not rates.empty
        assert "completion_rate" in rates.columns
        assert all(0 <= r <= 100 for r in rates["completion_rate"])

    def test_drop_off_analysis(self, course_activity):
        engine = CourseAnalytics(course_activity)
        result = engine.drop_off_analysis()
        assert not result.empty
        assert "progress_bucket" in result.columns

    def test_avg_time_to_completion(self, course_activity):
        engine = CourseAnalytics(course_activity)
        result = engine.avg_time_to_completion()
        # May be empty if no completions, but should not error
        if not result.empty:
            assert "avg_days" in result.columns
            assert all(result["avg_days"] >= 0)

    def test_engagement_vs_completion(self, course_activity):
        engine = CourseAnalytics(course_activity)
        result = engine.engagement_vs_completion()
        assert "avg_hours" in result.columns
        assert "completion_rate" in result.columns


# ── Program Analytics ───────────────────────────────────────────────

class TestProgramAnalytics:

    def test_success_rates(self, program_activity):
        if program_activity.empty:
            pytest.skip("No program data")
        engine = ProgramAnalytics(program_activity)
        rates = engine.success_rates()
        assert not rates.empty
        assert "success_rate" in rates.columns


# ── Video Analytics ─────────────────────────────────────────────────

class TestVideoAnalytics:

    def test_watch_frequency(self, video_activity):
        if video_activity.empty:
            pytest.skip("No video data")
        engine = VideoAnalytics(video_activity)
        freq = engine.watch_frequency()
        assert not freq.empty
        assert "total_watches" in freq.columns

    def test_rewatch_patterns(self, video_activity):
        if video_activity.empty:
            pytest.skip("No video data")
        engine = VideoAnalytics(video_activity)
        result = engine.rewatch_patterns()
        # May be empty if no rewatches
        if not result.empty:
            assert "total_rewatches" in result.columns


# ── Engagement Scoring ──────────────────────────────────────────────

class TestEngagementScoring:

    def test_compute_scores(self, course_activity, video_activity):
        engine = EngagementScoringEngine()
        scores = engine.compute(course_activity, video_activity)
        assert not scores.empty
        assert "score" in scores.columns
        assert "category" in scores.columns
        assert all(0 <= s <= 100 for s in scores["score"])

    def test_categories_valid(self, course_activity, video_activity):
        engine = EngagementScoringEngine()
        scores = engine.compute(course_activity, video_activity)
        valid_cats = {"High", "Medium", "Low", "At-Risk"}
        assert set(scores["category"].unique()).issubset(valid_cats)

    def test_category_distribution(self, course_activity, video_activity):
        engine = EngagementScoringEngine()
        scores = engine.compute(course_activity, video_activity)
        dist = engine.category_distribution(scores)
        assert "distribution" in dist
        assert dist["total_learners"] == len(scores)


# ── Skill Intelligence ──────────────────────────────────────────────

class TestSkillIntelligence:

    def test_compute_skill_scores(self, course_activity):
        engine = SkillIntelligenceEngine(course_activity)
        scores = engine.compute_skill_scores()
        assert not scores.empty
        assert "skill_name" in scores.columns
        assert "skill_score" in scores.columns
        assert all(0 <= s <= 100 for s in scores["skill_score"])

    def test_skill_gap_analysis(self, course_activity):
        engine = SkillIntelligenceEngine(course_activity)
        gaps = engine.skill_gap_analysis()
        assert not gaps.empty
        assert "gap_index" in gaps.columns

    def test_org_distribution(self, course_activity):
        engine = SkillIntelligenceEngine(course_activity)
        dist = engine.org_skill_distribution()
        assert not dist.empty
        assert "skill_category" in dist.columns


# ── Predictive Analytics ────────────────────────────────────────────

class TestPredictiveAnalytics:

    def test_prepare_features(self, course_activity):
        engine = PredictiveAnalytics()
        features = engine.prepare_features(course_activity)
        assert not features.empty
        assert "avg_progress" in features.columns
        assert "total_hours" in features.columns

    def test_predict_dropout(self, course_activity):
        engine = PredictiveAnalytics()
        features = engine.prepare_features(course_activity)
        result = engine.predict_dropout(features)
        if "error" not in result:
            assert "accuracy" in result
            assert 0 <= result["accuracy"] <= 1
            assert "feature_importances" in result

    def test_predict_completion(self, course_activity):
        engine = PredictiveAnalytics()
        features = engine.prepare_features(course_activity)
        result = engine.predict_completion(features)
        if "error" not in result:
            assert "accuracy" in result


# ── Insight Engine ──────────────────────────────────────────────────

class TestInsightEngine:

    def test_generates_insights(self, course_activity, video_activity, program_activity):
        engine = InsightEngine()
        eng_engine = EngagementScoringEngine()
        engagement = eng_engine.compute(course_activity, video_activity)

        insights = engine.generate_all(course_activity, engagement, video_activity, program_activity)
        assert len(insights) > 0
        assert all("title" in i for i in insights)
        assert all("severity" in i for i in insights)
        assert all(i["severity"] in ("critical", "warning", "info") for i in insights)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
