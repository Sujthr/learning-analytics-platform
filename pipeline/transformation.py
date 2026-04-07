"""Data Transformation — build unified data models from raw Coursera CSVs."""

import logging
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataTransformer:
    """Transform cleaned Coursera datasets into unified analytical models."""

    def __init__(self):
        self.learners_df: pd.DataFrame = pd.DataFrame()
        self.courses_df: pd.DataFrame = pd.DataFrame()
        self.course_activity_df: pd.DataFrame = pd.DataFrame()
        self.program_activity_df: pd.DataFrame = pd.DataFrame()
        self.spec_activity_df: pd.DataFrame = pd.DataFrame()
        self.video_activity_df: pd.DataFrame = pd.DataFrame()

    def transform_all(self, datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Transform all datasets into unified models.

        Args:
            datasets: dict with keys 'course', 'program', 'specialization', 'video'

        Returns:
            dict of transformed DataFrames
        """
        result = {}

        if "course" in datasets:
            self.learners_df = self._build_learner_table(datasets["course"])
            self.courses_df = self._build_course_table(datasets["course"])
            self.course_activity_df = self._build_course_activity(datasets["course"])
            result["learners"] = self.learners_df
            result["courses"] = self.courses_df
            result["course_activity"] = self.course_activity_df

        if "program" in datasets:
            self.program_activity_df = self._build_program_activity(datasets["program"])
            result["program_activity"] = self.program_activity_df

        if "specialization" in datasets:
            self.spec_activity_df = self._build_specialization_activity(datasets["specialization"])
            result["specialization_activity"] = self.spec_activity_df

        if "video" in datasets:
            self.video_activity_df = self._build_video_activity(datasets["video"])
            result["video_activity"] = self.video_activity_df

        logger.info(f"Transformation complete: {list(result.keys())}")
        return result

    def _build_learner_table(self, course_df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique learner dimension from course data."""
        learner_cols = ["Email", "External ID"]
        optional = ["Name", "Business Unit", "Role", "Location"]
        cols = learner_cols + [c for c in optional if c in course_df.columns]

        learners = course_df[cols].drop_duplicates(subset=["Email"]).reset_index(drop=True)
        learners.columns = [c.lower().replace(" ", "_").replace("(%)", "pct") for c in learners.columns]

        # Compute enrollment date = earliest enrollment
        if "Enrollment Timestamp" in course_df.columns:
            ts_col = pd.to_datetime(course_df["Enrollment Timestamp"], errors="coerce")
            enroll_min = ts_col.groupby(course_df["Email"]).min().reset_index()
            enroll_min.columns = ["email", "enrollment_date"]
            learners = learners.merge(enroll_min, on="email", how="left")

        # Activity status: active if any activity in last 90 days
        if "Last Activity Timestamp" in course_df.columns:
            ts_col = pd.to_datetime(course_df["Last Activity Timestamp"], errors="coerce")
            last_act = ts_col.groupby(course_df["Email"]).max().reset_index()
            last_act.columns = ["email", "last_activity"]
            learners = learners.merge(last_act, on="email", how="left")
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
            learners["is_active"] = learners["last_activity"] >= cutoff

        learners.insert(0, "learner_id", range(1, len(learners) + 1))
        return learners

    def _build_course_table(self, course_df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique course dimension."""
        cols = ["Course Id", "Course Name"]
        optional = ["Course Slug", "Institution"]
        cols += [c for c in optional if c in course_df.columns]

        courses = course_df[cols].drop_duplicates(subset=["Course Id"]).reset_index(drop=True)
        courses.columns = [c.lower().replace(" ", "_").replace("(%)", "pct") for c in courses.columns]

        # Estimated hours per course
        if "Learning Hours" in course_df.columns:
            hours = (
                course_df[course_df["Completed"] == "Yes"]
                .groupby("Course Id")["Learning Hours"]
                .median()
                .reset_index()
            )
            hours.columns = ["course_id", "estimated_hours"]
            courses = courses.merge(hours, on="course_id", how="left")

        return courses

    def _build_course_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize course activity fact table."""
        rename_map = {
            "Email": "email",
            "External ID": "external_id",
            "Course Id": "course_id",
            "Course Name": "course_name",
            "Enrollment Timestamp": "enrollment_ts",
            "Completion Timestamp": "completion_ts",
            "Progress (%)": "progress_pct",
            "Grade (%)": "grade",
            "Learning Hours": "learning_hours",
            "Completed": "is_completed",
            "Last Activity Timestamp": "last_activity_ts",
        }
        out = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Normalize boolean
        if "is_completed" in out.columns:
            out["is_completed"] = out["is_completed"].map(
                {"Yes": True, "No": False, True: True, False: False}
            ).fillna(False)

        # Convert grade to float
        if "grade" in out.columns:
            out["grade"] = pd.to_numeric(out["grade"], errors="coerce")

        return out

    def _build_program_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "Email": "email",
            "External ID": "external_id",
            "Program Id": "program_id",
            "Program Name": "program_name",
            "Program Slug": "program_slug",
            "Total Courses in Program": "total_courses",
            "Courses Completed": "courses_completed",
            "Progress (%)": "progress_pct",
            "Learning Hours": "learning_hours",
            "Completed": "is_completed",
            "Enrollment Timestamp": "enrollment_ts",
            "Last Activity Timestamp": "last_activity_ts",
        }
        out = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "is_completed" in out.columns:
            out["is_completed"] = out["is_completed"].map(
                {"Yes": True, "No": False}
            ).fillna(False)
        return out

    def _build_specialization_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "Email": "email",
            "External ID": "external_id",
            "Specialization Id": "spec_id",
            "Specialization Name": "spec_name",
            "Specialization Slug": "spec_slug",
            "Total Courses": "total_courses",
            "Courses Completed": "courses_completed",
            "Progress (%)": "progress_pct",
            "Learning Hours": "learning_hours",
            "Completed": "is_completed",
            "Enrollment Timestamp": "enrollment_ts",
            "Last Activity Timestamp": "last_activity_ts",
        }
        out = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "is_completed" in out.columns:
            out["is_completed"] = out["is_completed"].map(
                {"Yes": True, "No": False}
            ).fillna(False)
        return out

    def _build_video_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "Email": "email",
            "External ID": "external_id",
            "Course Id": "course_id",
            "Course Name": "course_name",
            "Video Name": "video_name",
            "Video Id": "video_id",
            "Watch Duration (seconds)": "watch_seconds",
            "Total Duration (seconds)": "total_seconds",
            "Completion (%)": "completion_pct",
            "Watch Count": "watch_count",
            "Last Watch Timestamp": "last_watch_ts",
        }
        out = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        return out

    def get_unified_learner_view(self) -> pd.DataFrame:
        """Build a single learner-level summary joining all activity sources."""
        if self.learners_df.empty or self.course_activity_df.empty:
            raise ValueError("Run transform_all() first")

        base = self.learners_df.copy()

        # Course-level aggregates
        agg_dict = {
            "courses_enrolled": ("course_id", "count"),
            "courses_completed": ("is_completed", "sum"),
            "avg_progress": ("progress_pct", "mean"),
        }
        if "learning_hours" in self.course_activity_df.columns:
            agg_dict["total_learning_hours"] = ("learning_hours", "sum")
        if "grade" in self.course_activity_df.columns:
            agg_dict["avg_grade"] = ("grade", "mean")

        ca = self.course_activity_df.groupby("email").agg(**agg_dict).reset_index()
        ca["completion_rate"] = (ca["courses_completed"] / ca["courses_enrolled"] * 100).round(1)
        base = base.merge(ca, on="email", how="left")

        # Video-level aggregates
        if not self.video_activity_df.empty:
            va = self.video_activity_df.groupby("email").agg(
                videos_watched=("video_id", "count"),
                avg_video_completion=("completion_pct", "mean"),
                total_watch_seconds=("watch_seconds", "sum"),
                total_rewatches=("watch_count", "sum"),
            ).reset_index()
            base = base.merge(va, on="email", how="left")

        # Program-level aggregates
        if not self.program_activity_df.empty:
            pa = self.program_activity_df.groupby("email").agg(
                programs_enrolled=("program_id", "count"),
                programs_completed=("is_completed", "sum"),
                avg_program_progress=("progress_pct", "mean"),
            ).reset_index()
            base = base.merge(pa, on="email", how="left")

        return base
