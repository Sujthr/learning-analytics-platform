"""Course Analytics — completion rate, drop-off, avg time, engagement vs completion."""

import pandas as pd
import numpy as np


class CourseAnalytics:
    """Compute course-level analytics."""

    def __init__(self, course_activity_df: pd.DataFrame):
        self.df = course_activity_df.copy()

    def completion_rates(self) -> pd.DataFrame:
        """Per-course completion rate and stats."""
        agg_dict = {
            "enrollments": ("email", "count"),
            "completed": ("is_completed", "sum"),
            "avg_progress": ("progress_pct", "mean"),
        }
        if "learning_hours" in self.df.columns:
            agg_dict["avg_hours"] = ("learning_hours", "mean")
        if "grade" in self.df.columns:
            agg_dict["avg_grade"] = ("grade", "mean")

        grouped = self.df.groupby(["course_id", "course_name"]).agg(**agg_dict).reset_index()
        grouped["completion_rate"] = (grouped["completed"] / grouped["enrollments"] * 100).round(1)
        grouped["drop_off_rate"] = (100 - grouped["completion_rate"]).round(1)
        return grouped.sort_values("completion_rate", ascending=False)

    def drop_off_analysis(self) -> pd.DataFrame:
        """Analyze where learners drop off (progress distribution)."""
        bins = [0, 10, 25, 50, 75, 90, 100]
        labels = ["0-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-100%"]
        self.df["progress_bucket"] = pd.cut(
            self.df["progress_pct"], bins=bins, labels=labels, include_lowest=True
        )

        result = (
            self.df.groupby(["course_id", "course_name", "progress_bucket"], observed=True)
            .size()
            .reset_index(name="learner_count")
        )
        return result

    def avg_time_to_completion(self) -> pd.DataFrame:
        """Average days from enrollment to completion per course."""
        completed = self.df[self.df["is_completed"] == True].copy()
        if completed.empty:
            return pd.DataFrame()

        if "completion_ts" not in completed.columns or "enrollment_ts" not in completed.columns:
            # Fall back to last_activity_ts - enrollment_ts
            if "last_activity_ts" in completed.columns and "enrollment_ts" in completed.columns:
                completed["enrollment_ts"] = pd.to_datetime(completed["enrollment_ts"], errors="coerce")
                completed["last_activity_ts"] = pd.to_datetime(completed["last_activity_ts"], errors="coerce")
                completed["days_to_complete"] = (
                    completed["last_activity_ts"] - completed["enrollment_ts"]
                ).dt.days
            else:
                return pd.DataFrame()
        else:
            completed["enrollment_ts"] = pd.to_datetime(completed["enrollment_ts"], errors="coerce")
            completed["completion_ts"] = pd.to_datetime(completed["completion_ts"], errors="coerce")
            completed["days_to_complete"] = (
                completed["completion_ts"] - completed["enrollment_ts"]
            ).dt.days

        result = completed.groupby(["course_id", "course_name"]).agg(
            avg_days=("days_to_complete", "mean"),
            median_days=("days_to_complete", "median"),
            min_days=("days_to_complete", "min"),
            max_days=("days_to_complete", "max"),
            completions=("email", "count"),
        ).reset_index()
        result["avg_days"] = result["avg_days"].round(1)
        result["median_days"] = result["median_days"].round(1)
        return result

    def engagement_vs_completion(self) -> pd.DataFrame:
        """Correlation between learning hours and completion."""
        per_course = self.df.groupby("course_id").agg(
            avg_hours=("learning_hours", "mean"),
            completion_rate=("is_completed", "mean"),
            avg_progress=("progress_pct", "mean"),
        ).reset_index()
        per_course["completion_rate"] = (per_course["completion_rate"] * 100).round(1)
        return per_course

    def get_all_metrics(self) -> dict:
        rates = self.completion_rates()
        return {
            "completion_rates": rates.to_dict(orient="records"),
            "avg_completion_rate": round(rates["completion_rate"].mean(), 1),
            "avg_drop_off_rate": round(rates["drop_off_rate"].mean(), 1),
            "time_to_completion": self.avg_time_to_completion().to_dict(orient="records"),
            "total_courses": len(rates),
        }
