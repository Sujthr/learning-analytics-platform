"""Learner Analytics — active/inactive, velocity, completion, engagement."""

import pandas as pd
import numpy as np
from datetime import timedelta


class LearnerAnalytics:
    """Compute learner-level analytics from unified learner view."""

    def __init__(self, learner_df: pd.DataFrame, course_activity_df: pd.DataFrame):
        self.learners = learner_df.copy()
        self.activity = course_activity_df.copy()

    def active_vs_inactive(self, days: int = 90) -> dict:
        """Classify learners as active/inactive based on recency."""
        if "last_activity_ts" not in self.activity.columns:
            return {"active": 0, "inactive": 0, "pct_active": 0}

        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        last_act = self.activity.groupby("email")["last_activity_ts"].max()
        active = (last_act >= cutoff).sum()
        inactive = (last_act < cutoff).sum()
        total = active + inactive
        return {
            "active": int(active),
            "inactive": int(inactive),
            "total": int(total),
            "pct_active": round(active / total * 100, 1) if total else 0,
            "cutoff_days": days,
        }

    def learning_velocity(self) -> pd.DataFrame:
        """Hours per week per learner."""
        df = self.activity.copy()
        if "enrollment_ts" not in df.columns or "last_activity_ts" not in df.columns:
            return pd.DataFrame()

        df["enrollment_ts"] = pd.to_datetime(df["enrollment_ts"], errors="coerce")
        df["last_activity_ts"] = pd.to_datetime(df["last_activity_ts"], errors="coerce")
        df["duration_weeks"] = (
            (df["last_activity_ts"] - df["enrollment_ts"]).dt.days / 7
        ).clip(lower=1)

        per_learner = df.groupby("email").agg(
            total_hours=("learning_hours", "sum"),
            total_weeks=("duration_weeks", "max"),
        ).reset_index()
        per_learner["hours_per_week"] = (
            per_learner["total_hours"] / per_learner["total_weeks"]
        ).round(2)
        return per_learner

    def completion_summary(self) -> dict:
        """Overall completion statistics."""
        total_enrollments = len(self.activity)
        completed = self.activity["is_completed"].sum() if "is_completed" in self.activity.columns else 0
        rate = round(completed / total_enrollments * 100, 1) if total_enrollments else 0

        return {
            "total_enrollments": int(total_enrollments),
            "total_completed": int(completed),
            "completion_rate": rate,
            "avg_progress": round(self.activity["progress_pct"].mean(), 1),
            "median_progress": round(self.activity["progress_pct"].median(), 1),
        }

    def completion_by_dimension(self, dimension: str) -> pd.DataFrame:
        """Completion rate by Business Unit, Role, Location, etc."""
        df = self.activity.copy()
        col_map = {
            "business_unit": "Business Unit",
            "role": "Role",
            "location": "Location",
        }
        col = col_map.get(dimension, dimension)
        if col not in df.columns:
            return pd.DataFrame()

        grouped = df.groupby(col).agg(
            enrollments=("email", "count"),
            completed=("is_completed", "sum"),
            avg_progress=("progress_pct", "mean"),
            avg_hours=("learning_hours", "mean"),
        ).reset_index()
        grouped["completion_rate"] = (grouped["completed"] / grouped["enrollments"] * 100).round(1)
        return grouped.sort_values("completion_rate", ascending=False)

    def top_learners(self, n: int = 20) -> pd.DataFrame:
        """Top N learners by completion and hours."""
        per_learner = self.activity.groupby("email").agg(
            courses_completed=("is_completed", "sum"),
            total_hours=("learning_hours", "sum"),
            avg_progress=("progress_pct", "mean"),
            courses_enrolled=("course_id", "count"),
        ).reset_index()
        per_learner["score"] = (
            per_learner["courses_completed"] * 10
            + per_learner["avg_progress"] * 0.5
            + per_learner["total_hours"] * 0.3
        )
        return per_learner.sort_values("score", ascending=False).head(n)

    def get_all_metrics(self) -> dict:
        """Return all learner analytics."""
        return {
            "active_inactive": self.active_vs_inactive(),
            "completion": self.completion_summary(),
            "velocity": self.learning_velocity().describe().to_dict(),
            "top_learners": self.top_learners(10).to_dict(orient="records"),
        }
