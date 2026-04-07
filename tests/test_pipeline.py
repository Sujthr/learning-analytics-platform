"""Tests for data pipeline — ingestion, cleaning, transformation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np

from pipeline.ingestion import DataIngestor
from pipeline.cleaning import DataCleaner
from pipeline.transformation import DataTransformer
from core.config import settings


# ── Ingestion Tests ─────────────────────────────────────────────────

class TestDataIngestor:

    def test_ingest_course_csv(self):
        ingestor = DataIngestor()
        df = ingestor.ingest(
            str(Path(settings.data_dir) / "course_activity.csv"),
            source_type="course",
        )
        assert not df.empty
        assert "Email" in df.columns
        assert "Course Id" in df.columns
        assert "Progress (%)" in df.columns
        assert len(df) > 100

    def test_ingest_video_csv(self):
        ingestor = DataIngestor()
        df = ingestor.ingest(
            str(Path(settings.data_dir) / "video_clip_activity.csv"),
            source_type="video",
        )
        assert not df.empty
        assert "Video Id" in df.columns

    def test_ingest_all(self):
        ingestor = DataIngestor()
        datasets = ingestor.ingest_all(settings.data_dir)
        assert "course" in datasets
        assert "video" in datasets
        assert len(datasets) >= 2

    def test_schema_validation_fails_on_bad_data(self):
        ingestor = DataIngestor()
        # Create a temp CSV missing required columns
        import tempfile
        bad_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            bad_df.to_csv(f, index=False)
            tmp_path = f.name

        with pytest.raises(ValueError, match="missing required columns"):
            ingestor.ingest(tmp_path, source_type="course")

        Path(tmp_path).unlink()

    def test_file_not_found(self):
        ingestor = DataIngestor()
        with pytest.raises(FileNotFoundError):
            ingestor.ingest("/nonexistent/file.csv", source_type="course")


# ── Cleaning Tests ──────────────────────────────────────────────────

class TestDataCleaner:

    def _sample_df(self):
        return pd.DataFrame({
            "Email": ["a@test.com", "b@test.com", "a@test.com", "c@test.com"],
            "score": [85.0, np.nan, 85.0, 92.0],
            "category": ["A", "B", "A", "A"],
            "timestamp": ["2024-01-01", "2024-02-15", "2024-01-01", "invalid"],
            "Progress (%)": [50, 110, 50, -5],
        })

    def test_removes_duplicates(self):
        cleaner = DataCleaner()
        df = self._sample_df()
        cleaned = cleaner.clean(df)
        # Original has a duplicate row (indices 0 and 2 are identical)
        assert len(cleaned) < len(df)

    def test_fills_missing_numeric(self):
        cleaner = DataCleaner(numeric_strategy="median")
        df = self._sample_df()
        cleaned = cleaner.clean(df)
        assert cleaned["score"].isnull().sum() == 0

    def test_fills_missing_categorical(self):
        cleaner = DataCleaner(categorical_strategy="mode")
        df = self._sample_df()
        cleaned = cleaner.clean(df)
        assert cleaned["category"].isnull().sum() == 0

    def test_clips_percentages(self):
        cleaner = DataCleaner()
        df = self._sample_df()
        cleaned = cleaner.clean(df)
        assert cleaned["Progress (%)"].max() <= 100
        assert cleaned["Progress (%)"].min() >= 0

    def test_cleaning_report(self):
        cleaner = DataCleaner()
        df = self._sample_df()
        cleaner.clean(df)
        report = cleaner.cleaning_report
        assert "original_rows" in report
        assert "cleaned_rows" in report
        assert report["original_rows"] == 4


# ── Transformation Tests ────────────────────────────────────────────

class TestDataTransformer:

    def _load_sample(self):
        ingestor = DataIngestor()
        return ingestor.ingest_all(settings.data_dir)

    def test_transform_all(self):
        datasets = self._load_sample()
        transformer = DataTransformer()
        result = transformer.transform_all(datasets)
        assert "learners" in result
        assert "courses" in result
        assert "course_activity" in result
        assert not result["learners"].empty
        assert "email" in result["learners"].columns

    def test_unified_learner_view(self):
        datasets = self._load_sample()
        transformer = DataTransformer()
        transformer.transform_all(datasets)
        unified = transformer.get_unified_learner_view()
        assert not unified.empty
        assert "courses_enrolled" in unified.columns
        assert "avg_progress" in unified.columns

    def test_course_activity_boolean(self):
        datasets = self._load_sample()
        transformer = DataTransformer()
        result = transformer.transform_all(datasets)
        ca = result["course_activity"]
        # is_completed should be boolean-like (True/False values)
        unique_vals = set(ca["is_completed"].dropna().unique())
        assert unique_vals.issubset({True, False})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
