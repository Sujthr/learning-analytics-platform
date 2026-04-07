"""Insight Engine — automatically generate actionable insights from analytics."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional


class InsightEngine:
    """Generate automated, human-readable insights from analytics results."""

    def __init__(self):
        self.insights: list[dict] = []

    def generate_all(
        self,
        course_activity: pd.DataFrame,
        engagement_scores: pd.DataFrame | None = None,
        video_activity: pd.DataFrame | None = None,
        program_activity: pd.DataFrame | None = None,
    ) -> list[dict]:
        """Run all insight generators and return sorted list."""
        self.insights = []

        self._completion_insights(course_activity)
        self._engagement_insights(course_activity, engagement_scores)
        self._drop_off_insights(course_activity)
        self._business_unit_insights(course_activity)
        self._video_insights(video_activity)
        self._program_insights(program_activity)
        self._time_insights(course_activity)

        # Sort by severity: critical > warning > info
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        self.insights.sort(key=lambda x: severity_order.get(x["severity"], 3))

        return self.insights

    def _add(self, category: str, severity: str, title: str, description: str,
             metric_value: float = None, metadata: dict = None):
        self.insights.append({
            "category": category,
            "severity": severity,
            "title": title,
            "description": description,
            "metric_value": metric_value,
            "metadata": metadata or {},
            "generated_at": datetime.now().isoformat(),
        })

    def _completion_insights(self, df: pd.DataFrame):
        overall_rate = df["is_completed"].mean() * 100 if "is_completed" in df.columns else 0

        if overall_rate < 30:
            self._add("completion", "critical",
                       f"Low overall completion rate: {overall_rate:.1f}%",
                       f"Only {overall_rate:.1f}% of enrollments result in completion. "
                       "Consider reviewing course difficulty, engagement strategies, or time allocation.",
                       overall_rate)
        elif overall_rate < 50:
            self._add("completion", "warning",
                       f"Below-average completion rate: {overall_rate:.1f}%",
                       f"Completion rate of {overall_rate:.1f}% is below the 50% benchmark.",
                       overall_rate)

        # Per-course completion
        if "course_name" in df.columns:
            course_rates = df.groupby("course_name")["is_completed"].mean() * 100
            worst = course_rates.idxmin()
            best = course_rates.idxmax()

            if course_rates[worst] < 15:
                self._add("course", "warning",
                           f"Course '{worst}' has only {course_rates[worst]:.1f}% completion",
                           "This course may need content review, better prerequisites, or pacing adjustments.",
                           course_rates[worst],
                           {"course": worst})

            if course_rates[best] > 60:
                self._add("course", "info",
                           f"Top performing course: '{best}' at {course_rates[best]:.1f}% completion",
                           "Study this course's structure as a best-practice template.",
                           course_rates[best],
                           {"course": best})

    def _engagement_insights(self, course_df: pd.DataFrame, engagement_df: pd.DataFrame | None):
        if engagement_df is None or engagement_df.empty:
            return

        # High engagement correlates with completion
        merged = course_df.groupby("email")["is_completed"].any().reset_index()
        merged.columns = ["email", "has_completion"]
        merged = merged.merge(engagement_df[["email", "score", "category"]], on="email", how="inner")

        high_eng = merged[merged["category"] == "High"]
        low_eng = merged[merged["category"].isin(["Low", "At-Risk"])]

        if len(high_eng) > 5 and len(low_eng) > 5:
            high_comp = high_eng["has_completion"].mean() * 100
            low_comp = low_eng["has_completion"].mean() * 100
            ratio = high_comp / low_comp if low_comp > 0 else float("inf")

            if ratio > 1.5:
                self._add("engagement", "info",
                           f"High engagement learners are {ratio:.1f}x more likely to complete",
                           f"High engagement completion: {high_comp:.0f}% vs "
                           f"Low/At-Risk: {low_comp:.0f}%. "
                           "Investing in engagement interventions could improve outcomes.",
                           ratio)

        # At-risk count
        at_risk = len(engagement_df[engagement_df["category"] == "At-Risk"])
        total = len(engagement_df)
        if at_risk > 0:
            pct = at_risk / total * 100
            severity = "critical" if pct > 25 else "warning"
            self._add("engagement", severity,
                       f"{at_risk} learners ({pct:.1f}%) are At-Risk",
                       "These learners show low progress, minimal hours, and infrequent activity. "
                       "Consider targeted outreach or nudge campaigns.",
                       pct,
                       {"at_risk_count": at_risk})

    def _drop_off_insights(self, df: pd.DataFrame):
        if "progress_pct" not in df.columns:
            return

        # Find the most common drop-off point
        incomplete = df[df.get("is_completed", pd.Series(dtype=bool)) == False]
        if incomplete.empty:
            return

        bins = [0, 10, 25, 50, 75, 100]
        labels = ["0-10%", "10-25%", "25-50%", "50-75%", "75-100%"]
        buckets = pd.cut(incomplete["progress_pct"], bins=bins, labels=labels, include_lowest=True)
        top_bucket = buckets.value_counts().idxmax()
        top_count = buckets.value_counts().max()
        pct_of_drops = top_count / len(incomplete) * 100

        self._add("drop_off", "warning",
                   f"Highest drop-off zone: {top_bucket} progress ({pct_of_drops:.0f}% of non-completers)",
                   f"{top_count} learners stalled in the {top_bucket} range. "
                   "Examine course content at this stage for friction points.",
                   pct_of_drops)

    def _business_unit_insights(self, df: pd.DataFrame):
        bu_col = None
        for candidate in ["Business Unit", "business_unit"]:
            if candidate in df.columns:
                bu_col = candidate
                break
        if not bu_col:
            return

        bu_rates = df.groupby(bu_col).agg(
            completion_rate=("is_completed", "mean"),
            avg_hours=("learning_hours", "mean"),
            count=("email", "count"),
        ).reset_index()
        bu_rates["completion_rate"] *= 100

        overall = bu_rates["completion_rate"].mean()
        for _, row in bu_rates.iterrows():
            diff = row["completion_rate"] - overall
            if diff < -15 and row["count"] >= 10:
                self._add("business_unit", "warning",
                           f"'{row[bu_col]}' has {abs(diff):.0f}% lower completion than average",
                           f"Completion rate: {row['completion_rate']:.1f}% vs org avg {overall:.1f}%. "
                           f"Avg learning hours: {row['avg_hours']:.1f}h. "
                           "May indicate time/resource constraints.",
                           row["completion_rate"],
                           {"business_unit": row[bu_col]})

    def _video_insights(self, video_df: pd.DataFrame | None):
        if video_df is None or video_df.empty:
            return

        avg_completion = video_df["completion_pct"].mean() if "completion_pct" in video_df.columns else 0
        if avg_completion < 50:
            self._add("video", "warning",
                       f"Average video completion is only {avg_completion:.1f}%",
                       "Learners are watching less than half of video content on average. "
                       "Consider shorter videos or more interactive content.",
                       avg_completion)

        # Most rewatched
        if "watch_count" in video_df.columns:
            rewatched = video_df[video_df["watch_count"] > 2]
            if len(rewatched) > 10:
                top = rewatched.groupby("video_name")["watch_count"].sum().idxmax()
                self._add("video", "info",
                           f"Most rewatched video: '{top}'",
                           "High rewatch rates may indicate complex material needing better explanation, "
                           "or highly valued content worth referencing.",
                           None, {"video": top})

    def _program_insights(self, program_df: pd.DataFrame | None):
        if program_df is None or program_df.empty:
            return

        if "is_completed" in program_df.columns:
            rate = program_df["is_completed"].mean() * 100
            if rate < 20:
                self._add("program", "warning",
                           f"Program completion rate is very low: {rate:.1f}%",
                           "Most learners are not completing full programs. "
                           "Consider program length, pacing, or flexibility.",
                           rate)

    def _time_insights(self, df: pd.DataFrame):
        if "learning_hours" not in df.columns:
            return

        avg_hours = df.groupby("email")["learning_hours"].sum().mean()
        self._add("engagement", "info",
                   f"Average total learning hours per learner: {avg_hours:.1f}h",
                   "Benchmark against industry averages and organizational targets.",
                   avg_hours)
