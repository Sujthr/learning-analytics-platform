"""Engagement Scoring Engine — composite score from progress, hours, video, recency."""

import pandas as pd
import numpy as np
from datetime import datetime


class EngagementScoringEngine:
    """
    Compute engagement scores (0-100) per learner using:
      - Progress %
      - Learning hours
      - Video interactions
      - Recency of activity

    Categories: High (75-100), Medium (50-74), Low (25-49), At-Risk (0-24)
    """

    CATEGORIES = [
        (75, "High"),
        (50, "Medium"),
        (25, "Low"),
        (0, "At-Risk"),
    ]

    def __init__(
        self,
        weights: dict | None = None,
        reference_date: datetime | None = None,
    ):
        self.weights = weights or {
            "progress_pct": 0.25,
            "learning_hours": 0.25,
            "video_interactions": 0.25,
            "recency_score": 0.25,
        }
        self.reference_date = reference_date or datetime.now()

    def compute(
        self,
        course_activity: pd.DataFrame,
        video_activity: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute engagement score for each learner.

        Returns DataFrame with: email, score, category, and component scores.
        """
        # 1. Progress component
        progress = (
            course_activity.groupby("email")["progress_pct"]
            .mean()
            .reset_index()
            .rename(columns={"progress_pct": "progress_component"})
        )

        # 2. Learning hours component (normalized to 0-100)
        hours = (
            course_activity.groupby("email")["learning_hours"]
            .sum()
            .reset_index()
            .rename(columns={"learning_hours": "total_hours"})
        )
        max_hours = hours["total_hours"].quantile(0.95) or 1
        hours["hours_component"] = (hours["total_hours"] / max_hours * 100).clip(0, 100)

        # 3. Video interaction component
        if video_activity is not None and not video_activity.empty:
            video = video_activity.groupby("email").agg(
                avg_vid_completion=("completion_pct", "mean"),
                total_watches=("watch_count", "sum"),
            ).reset_index()
            max_watches = video["total_watches"].quantile(0.95) or 1
            video["video_component"] = (
                video["avg_vid_completion"] * 0.6
                + (video["total_watches"] / max_watches * 100).clip(0, 100) * 0.4
            ).round(2)
        else:
            video = progress[["email"]].copy()
            video["video_component"] = 50.0  # neutral if no video data

        # 4. Recency component
        if "last_activity_ts" in course_activity.columns:
            recency = (
                course_activity.groupby("email")["last_activity_ts"]
                .max()
                .reset_index()
            )
            recency["last_activity_ts"] = pd.to_datetime(recency["last_activity_ts"], errors="coerce")
            ref = pd.Timestamp(self.reference_date)
            recency["days_since"] = (ref - recency["last_activity_ts"]).dt.days.clip(lower=0)
            max_days = 365
            recency["recency_component"] = (
                (1 - recency["days_since"] / max_days) * 100
            ).clip(0, 100).round(2)
        else:
            recency = progress[["email"]].copy()
            recency["recency_component"] = 50.0

        # Merge all components
        result = progress.merge(hours[["email", "hours_component"]], on="email", how="left")
        result = result.merge(video[["email", "video_component"]], on="email", how="left")
        result = result.merge(recency[["email", "recency_component"]], on="email", how="left")
        result = result.fillna(0)

        # Weighted score
        w = self.weights
        result["score"] = (
            result["progress_component"] * w["progress_pct"]
            + result["hours_component"] * w["learning_hours"]
            + result["video_component"] * w["video_interactions"]
            + result["recency_component"] * w["recency_score"]
        ).round(1)

        result["category"] = result["score"].apply(self._categorize)

        return result[["email", "score", "category",
                        "progress_component", "hours_component",
                        "video_component", "recency_component"]]

    def _categorize(self, score: float) -> str:
        for threshold, label in self.CATEGORIES:
            if score >= threshold:
                return label
        return "At-Risk"

    def category_distribution(self, scores_df: pd.DataFrame) -> dict:
        """Count learners per engagement category."""
        dist = scores_df["category"].value_counts().to_dict()
        total = len(scores_df)
        return {
            "distribution": dist,
            "total_learners": total,
            "avg_score": round(scores_df["score"].mean(), 1),
            "median_score": round(scores_df["score"].median(), 1),
        }
