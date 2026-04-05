"""Feature Engineering Engine for the Learning Analytics Platform.

Computes learner engagement and performance features from integrated datasets.
"""

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generates analytical features from integrated learner data."""

    def __init__(self, config: dict):
        self.config = config
        self.engagement_weights = config.get("engagement_score", {}).get("weights", {
            "session_frequency": 0.3,
            "video_completion": 0.25,
            "assessment_score": 0.25,
            "forum_activity": 0.2,
        })
        self.session_window = config.get("session_frequency", {}).get("window_days", 30)
        self.video_threshold = config.get("video_completion", {}).get("threshold", 0.8)
        self.improvement_method = config.get("assessment_improvement", {}).get("method", "slope")
        self._custom_features: dict[str, Callable] = {}

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature computations and return augmented DataFrame."""
        logger.info("Generating features for %d rows", len(df))
        df = df.copy()

        df = self.compute_session_frequency(df, self.session_window)
        df = self.compute_video_completion(df, self.video_threshold)
        df = self.compute_assessment_improvement(df, self.improvement_method)
        df = self.compute_engagement_score(df)

        # Add dropout flag based on enrollment status
        if "enrollment_status" in df.columns:
            df["dropout"] = (df["enrollment_status"] == "withdrawn").astype(int)

        # Add final_score as target for regression
        if "gpa" in df.columns:
            df["final_score"] = df["gpa"]
        elif "score" in df.columns:
            df["final_score"] = df["score"]

        # Run custom features
        for name, func in self._custom_features.items():
            try:
                df[name] = func(df)
                logger.info("Computed custom feature: %s", name)
            except Exception as e:
                logger.error("Failed to compute custom feature '%s': %s", name, e)

        logger.info("Feature generation complete: %d columns total", len(df.columns))
        return df

    def compute_engagement_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute weighted engagement score from component features.

        Components (normalized to 0-1): session_frequency, video_completion_pct,
        avg_assessment_score, forum_activity.
        """
        df = df.copy()
        components = {}

        # Normalize each component to 0-1 range
        if "session_frequency" in df.columns:
            vals = df["session_frequency"]
            components["session_frequency"] = self._normalize(vals)

        if "video_completion_pct" in df.columns:
            components["video_completion"] = df["video_completion_pct"].clip(0, 1)

        if "score" in df.columns:
            components["assessment_score"] = self._normalize(df["score"])
        elif "avg_assessment_score" in df.columns:
            components["assessment_score"] = self._normalize(df["avg_assessment_score"])

        if "forum_posts" in df.columns:
            components["forum_activity"] = self._normalize(df["forum_posts"])

        if not components:
            df["engagement_score"] = np.nan
            logger.warning("No components available for engagement score")
            return df

        score = pd.Series(0.0, index=df.index)
        total_weight = 0.0
        for key, series in components.items():
            weight = self.engagement_weights.get(key, 0.0)
            score += series.fillna(0) * weight
            total_weight += weight

        if total_weight > 0:
            score /= total_weight  # Re-normalize by actual weights used

        df["engagement_score"] = score.round(4)
        logger.info("Computed engagement_score (mean=%.3f)", df["engagement_score"].mean())
        return df

    def compute_session_frequency(
        self, df: pd.DataFrame, window_days: int = 30
    ) -> pd.DataFrame:
        """Compute session count per student (or per student-course)."""
        df = df.copy()

        # Use record counts if available
        count_cols = [c for c in df.columns if c.endswith("_record_count")]
        if count_cols:
            df["session_frequency"] = df[count_cols].sum(axis=1)
        elif "student_id" in df.columns:
            # Count rows per student as proxy
            freq = df.groupby("student_id").cumcount() + 1
            df["session_frequency"] = freq
        else:
            df["session_frequency"] = 1

        logger.info("Computed session_frequency (mean=%.1f)", df["session_frequency"].mean())
        return df

    def compute_video_completion(
        self, df: pd.DataFrame, threshold: float = 0.8
    ) -> pd.DataFrame:
        """Compute video completion percentage per student."""
        df = df.copy()

        if "completion_pct" in df.columns:
            df["video_completion_pct"] = df["completion_pct"]
            df["video_completed"] = (df["completion_pct"] >= threshold).astype(int)
        else:
            df["video_completion_pct"] = np.nan
            df["video_completed"] = np.nan

        logger.info("Computed video_completion_pct")
        return df

    def compute_assessment_improvement(
        self, df: pd.DataFrame, method: str = "slope"
    ) -> pd.DataFrame:
        """Compute rate of improvement in assessment scores.

        Methods:
            - slope: Linear regression slope of scores over time.
            - pct_change: Average percentage change between consecutive scores.
        """
        df = df.copy()

        if "score" not in df.columns and "gpa" not in df.columns:
            df["assessment_improvement"] = np.nan
            return df

        score_col = "score" if "score" in df.columns else "gpa"

        if "student_id" in df.columns:
            if method == "pct_change":
                df["assessment_improvement"] = (
                    df.groupby("student_id")[score_col]
                    .pct_change()
                    .fillna(0)
                )
            else:
                # Simple difference as improvement proxy (slope requires time ordering)
                df["assessment_improvement"] = (
                    df.groupby("student_id")[score_col]
                    .diff()
                    .fillna(0)
                )
        else:
            df["assessment_improvement"] = 0.0

        logger.info("Computed assessment_improvement (method=%s)", method)
        return df

    def add_custom_feature(self, name: str, func: Callable[[pd.DataFrame], pd.Series]):
        """Register a custom feature function.

        Args:
            name: Feature column name.
            func: Callable that takes a DataFrame and returns a Series.
        """
        self._custom_features[name] = func
        logger.info("Registered custom feature: %s", name)

    def get_feature_summary(self, df: pd.DataFrame) -> dict:
        """Return descriptive statistics for all engineered features."""
        feature_cols = [
            "engagement_score", "session_frequency", "video_completion_pct",
            "video_completed", "assessment_improvement", "dropout", "final_score",
        ] + list(self._custom_features.keys())

        available = [c for c in feature_cols if c in df.columns]
        summary = {}
        for col in available:
            series = df[col].dropna()
            if pd.api.types.is_numeric_dtype(series):
                summary[col] = {
                    "count": int(series.count()),
                    "mean": round(float(series.mean()), 4),
                    "std": round(float(series.std()), 4),
                    "min": round(float(series.min()), 4),
                    "max": round(float(series.max()), 4),
                }

        return summary

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        """Min-max normalize a Series to [0, 1]."""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)
