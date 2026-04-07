"""Program Analytics — success rate, course completion distribution."""

import pandas as pd
import numpy as np


class ProgramAnalytics:
    """Compute program-level analytics."""

    def __init__(self, program_activity_df: pd.DataFrame):
        self.df = program_activity_df.copy()

    def success_rates(self) -> pd.DataFrame:
        """Per-program completion/success rate."""
        if self.df.empty:
            return pd.DataFrame()

        grouped = self.df.groupby(["program_id", "program_name"]).agg(
            enrollments=("email", "count"),
            completed=("is_completed", "sum"),
            avg_progress=("progress_pct", "mean"),
            avg_hours=("learning_hours", "mean"),
        ).reset_index()
        grouped["success_rate"] = (grouped["completed"] / grouped["enrollments"] * 100).round(1)
        return grouped.sort_values("success_rate", ascending=False)

    def course_completion_distribution(self) -> pd.DataFrame:
        """Distribution of how many courses learners complete within programs."""
        if self.df.empty or "courses_completed" not in self.df.columns:
            return pd.DataFrame()

        dist = (
            self.df.groupby(["program_id", "program_name", "courses_completed"])
            .size()
            .reset_index(name="learner_count")
        )
        return dist

    def progress_by_program(self) -> pd.DataFrame:
        """Average progress per program."""
        if self.df.empty:
            return pd.DataFrame()
        return (
            self.df.groupby(["program_id", "program_name"])
            .agg(
                avg_progress=("progress_pct", "mean"),
                median_progress=("progress_pct", "median"),
                learners=("email", "nunique"),
            )
            .reset_index()
            .round(1)
        )

    def get_all_metrics(self) -> dict:
        rates = self.success_rates()
        return {
            "success_rates": rates.to_dict(orient="records") if not rates.empty else [],
            "avg_success_rate": round(rates["success_rate"].mean(), 1) if not rates.empty else 0,
            "total_programs": len(rates),
            "progress_by_program": self.progress_by_program().to_dict(orient="records"),
        }
