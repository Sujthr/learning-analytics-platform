"""Video Analytics — watch frequency, completion, rewatch patterns."""

import pandas as pd
import numpy as np


class VideoAnalytics:
    """Compute video-level analytics."""

    def __init__(self, video_activity_df: pd.DataFrame):
        self.df = video_activity_df.copy()

    def watch_frequency(self) -> pd.DataFrame:
        """Watch count per video across all learners."""
        if self.df.empty:
            return pd.DataFrame()

        grouped = self.df.groupby(["video_id", "video_name", "course_id"]).agg(
            total_watches=("watch_count", "sum"),
            unique_viewers=("email", "nunique"),
            avg_watch_count=("watch_count", "mean"),
        ).reset_index()
        return grouped.sort_values("total_watches", ascending=False)

    def completion_rates(self) -> pd.DataFrame:
        """Average completion % per video."""
        if self.df.empty:
            return pd.DataFrame()

        grouped = self.df.groupby(["video_id", "video_name", "course_id"]).agg(
            avg_completion=("completion_pct", "mean"),
            median_completion=("completion_pct", "median"),
            viewers=("email", "nunique"),
        ).reset_index()
        grouped["avg_completion"] = grouped["avg_completion"].round(1)
        return grouped.sort_values("avg_completion", ascending=True)

    def rewatch_patterns(self) -> pd.DataFrame:
        """Identify most-rewatched videos (watch_count > 1)."""
        if self.df.empty:
            return pd.DataFrame()

        rewatched = self.df[self.df["watch_count"] > 1].copy()
        if rewatched.empty:
            return pd.DataFrame()

        grouped = rewatched.groupby(["video_id", "video_name", "course_id"]).agg(
            total_rewatches=("watch_count", "sum"),
            avg_rewatch_count=("watch_count", "mean"),
            rewatchers=("email", "nunique"),
        ).reset_index()
        grouped["avg_rewatch_count"] = grouped["avg_rewatch_count"].round(1)
        return grouped.sort_values("total_rewatches", ascending=False)

    def course_video_summary(self) -> pd.DataFrame:
        """Summary of video engagement per course."""
        if self.df.empty:
            return pd.DataFrame()

        grouped = self.df.groupby("course_id").agg(
            total_videos=("video_id", "nunique"),
            total_watch_seconds=("watch_seconds", "sum"),
            avg_completion=("completion_pct", "mean"),
            avg_watch_count=("watch_count", "mean"),
            unique_viewers=("email", "nunique"),
        ).reset_index()
        grouped["avg_completion"] = grouped["avg_completion"].round(1)
        grouped["total_watch_hours"] = (grouped["total_watch_seconds"] / 3600).round(1)
        return grouped

    def get_all_metrics(self) -> dict:
        freq = self.watch_frequency()
        comp = self.completion_rates()
        return {
            "total_videos": int(freq["video_id"].nunique()) if not freq.empty else 0,
            "avg_completion": round(comp["avg_completion"].mean(), 1) if not comp.empty else 0,
            "most_watched": freq.head(10).to_dict(orient="records") if not freq.empty else [],
            "most_rewatched": self.rewatch_patterns().head(10).to_dict(orient="records"),
            "lowest_completion": comp.head(10).to_dict(orient="records") if not comp.empty else [],
            "course_summary": self.course_video_summary().to_dict(orient="records"),
        }
