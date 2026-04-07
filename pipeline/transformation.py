"""Data Transformation — build unified data models from raw Coursera CSVs.

Handles both real Coursera enterprise exports and generated sample data.
Gracefully skips columns that don't exist in the source.
"""

import logging
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _safe_rename(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Rename only columns that exist."""
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def _safe_bool(series: pd.Series) -> pd.Series:
    """Convert Yes/No/True/False strings to boolean."""
    return series.map({
        "Yes": True, "No": False,
        "yes": True, "no": False,
        True: True, False: False,
        "True": True, "False": False,
    }).fillna(False).astype(bool)


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

    # ── Learner dimension ───────────────────────────────────────────

    def _build_learner_table(self, course_df: pd.DataFrame) -> pd.DataFrame:
        id_col = "Email"
        optional = ["Name", "External ID", "Business Unit", "Role", "Location",
                     "Location Region", "Location Country", "Manager Name"]
        cols = [id_col] + [c for c in optional if c in course_df.columns]

        learners = course_df[cols].drop_duplicates(subset=["Email"]).reset_index(drop=True)
        learners.columns = [c.lower().replace(" ", "_").replace("(%)", "pct") for c in learners.columns]

        # Enrollment date
        if "Enrollment Timestamp" in course_df.columns:
            ts_col = pd.to_datetime(course_df["Enrollment Timestamp"], errors="coerce")
            enroll_min = ts_col.groupby(course_df["Email"]).min().reset_index()
            enroll_min.columns = ["email", "enrollment_date"]
            learners = learners.merge(enroll_min, on="email", how="left")

        # Active/inactive
        if "Last Activity Timestamp" in course_df.columns:
            ts_col = pd.to_datetime(course_df["Last Activity Timestamp"], errors="coerce")
            last_act = ts_col.groupby(course_df["Email"]).max().reset_index()
            last_act.columns = ["email", "last_activity"]
            learners = learners.merge(last_act, on="email", how="left")
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
            learners["is_active"] = learners["last_activity"] >= cutoff

        learners.insert(0, "learner_id", range(1, len(learners) + 1))
        return learners

    # ── Course dimension ────────────────────────────────────────────

    def _build_course_table(self, course_df: pd.DataFrame) -> pd.DataFrame:
        # Use Course Id if available, fall back to Course Name
        id_col = "Course Id" if "Course Id" in course_df.columns else "Course Name"
        cols = [id_col]
        if "Course Name" in course_df.columns and id_col != "Course Name":
            cols.append("Course Name")
        for opt in ["Course Slug", "Institution", "Course Type"]:
            if opt in course_df.columns:
                cols.append(opt)

        courses = course_df[cols].drop_duplicates(subset=[id_col]).reset_index(drop=True)
        courses.columns = [c.lower().replace(" ", "_").replace("(%)", "pct") for c in courses.columns]

        # Rename id column to course_id consistently
        first_col = courses.columns[0]
        if first_col != "course_id":
            courses = courses.rename(columns={first_col: "course_id"})

        # Estimated hours
        if "Learning Hours" in course_df.columns and "Completed" in course_df.columns:
            completed = course_df[course_df["Completed"] == "Yes"]
            if not completed.empty:
                hours = completed.groupby(id_col)["Learning Hours"].median().reset_index()
                hours.columns = ["course_id", "estimated_hours"]
                courses = courses.merge(hours, on="course_id", how="left")

        return courses

    # ── Course activity fact ────────────────────────────────────────

    def _build_course_activity(self, df: pd.DataFrame) -> pd.DataFrame:
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
            "Estimated Learning Hours": "estimated_learning_hours",
            "Completed": "is_completed",
            "Last Activity Timestamp": "last_activity_ts",
            "Course Type": "course_type",
            "Business Unit": "business_unit",
            "Role": "role",
            "Location": "location",
            "Location Region": "location_region",
            "Location Country": "location_country",
            "Program Name": "program_name",
            "Removed From Program": "removed_from_program",
        }
        out = _safe_rename(df, rename_map)

        # Ensure we have a course identifier
        if "course_id" not in out.columns and "course_name" in out.columns:
            out["course_id"] = out["course_name"]

        # Boolean conversion
        if "is_completed" in out.columns:
            out["is_completed"] = _safe_bool(out["is_completed"])

        # Numeric conversions
        if "grade" in out.columns:
            out["grade"] = pd.to_numeric(out["grade"], errors="coerce")
        if "progress_pct" in out.columns:
            out["progress_pct"] = pd.to_numeric(out["progress_pct"], errors="coerce").fillna(0)
        if "learning_hours" in out.columns:
            out["learning_hours"] = pd.to_numeric(out["learning_hours"], errors="coerce").fillna(0)

        return out

    # ── Program activity ────────────────────────────────────────────

    def _build_program_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "Email": "email",
            "External ID": "external_id",
            "Program Id": "program_id",
            "Program Name": "program_name",
            "Program Slug": "program_slug",
            "Total Courses in Program": "total_courses",
            "Total Courses": "total_courses",
            "Courses Completed": "courses_completed",
            "Progress (%)": "progress_pct",
            "Learning Hours": "learning_hours",
            "Completed": "is_completed",
            "Enrollment Timestamp": "enrollment_ts",
            "Last Activity Timestamp": "last_activity_ts",
        }
        out = _safe_rename(df, rename_map)

        if "program_id" not in out.columns and "program_name" in out.columns:
            out["program_id"] = out["program_name"]

        if "is_completed" in out.columns:
            out["is_completed"] = _safe_bool(out["is_completed"])

        for col in ["courses_completed", "total_courses"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

        # Compute progress from courses if not present
        if "progress_pct" not in out.columns and "courses_completed" in out.columns and "total_courses" in out.columns:
            out["progress_pct"] = (out["courses_completed"] / out["total_courses"].clip(lower=1) * 100).round(1)

        if "learning_hours" in out.columns:
            out["learning_hours"] = pd.to_numeric(out["learning_hours"], errors="coerce").fillna(0)

        return out

    # ── Specialization activity ─────────────────────────────────────

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
            "Completion Timestamp": "completion_ts",
            "Institution": "institution",
            "Program Name": "program_name",
        }
        out = _safe_rename(df, rename_map)

        if "spec_id" not in out.columns and "spec_name" in out.columns:
            out["spec_id"] = out["spec_name"]

        if "is_completed" in out.columns:
            out["is_completed"] = _safe_bool(out["is_completed"])

        for col in ["courses_completed", "total_courses"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

        if "progress_pct" not in out.columns and "courses_completed" in out.columns and "total_courses" in out.columns:
            out["progress_pct"] = (out["courses_completed"] / out["total_courses"].clip(lower=1) * 100).round(1)

        if "learning_hours" in out.columns:
            out["learning_hours"] = pd.to_numeric(out["learning_hours"], errors="coerce").fillna(0)

        return out

    # ── Video activity ──────────────────────────────────────────────

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
            "First Watch Timestamp": "first_watch_ts",
            "Learning Hours": "learning_hours",
            "Estimated Learning Hours": "estimated_learning_hours",
            "Video Completed": "video_completed",
            "Led to Enrollment": "led_to_enrollment",
        }
        out = _safe_rename(df, rename_map)

        if "course_id" not in out.columns and "course_name" in out.columns:
            out["course_id"] = out["course_name"]
        if "video_id" not in out.columns and "video_name" in out.columns:
            out["video_id"] = out["video_name"]

        # Compute watch_seconds from learning_hours if not present
        if "watch_seconds" not in out.columns and "learning_hours" in out.columns:
            out["learning_hours"] = pd.to_numeric(out["learning_hours"], errors="coerce").fillna(0)
            out["watch_seconds"] = (out["learning_hours"] * 3600).round(1)

        # Compute total_seconds from estimated_learning_hours if not present
        if "total_seconds" not in out.columns and "estimated_learning_hours" in out.columns:
            out["estimated_learning_hours"] = pd.to_numeric(out["estimated_learning_hours"], errors="coerce").fillna(0)
            out["total_seconds"] = (out["estimated_learning_hours"] * 3600).round(1)

        for col in ["completion_pct", "watch_count", "watch_seconds", "total_seconds"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

        return out

    # ── Unified learner view ────────────────────────────────────────

    def get_unified_learner_view(self) -> pd.DataFrame:
        if self.learners_df.empty or self.course_activity_df.empty:
            raise ValueError("Run transform_all() first")

        base = self.learners_df.copy()

        # Course-level aggregates
        ca_cols = self.course_activity_df.columns
        agg_dict = {
            "courses_enrolled": ("course_id", "count"),
            "courses_completed": ("is_completed", "sum"),
            "avg_progress": ("progress_pct", "mean"),
        }
        if "learning_hours" in ca_cols:
            agg_dict["total_learning_hours"] = ("learning_hours", "sum")
        if "grade" in ca_cols:
            agg_dict["avg_grade"] = ("grade", "mean")

        ca = self.course_activity_df.groupby("email").agg(**agg_dict).reset_index()
        ca["completion_rate"] = (ca["courses_completed"] / ca["courses_enrolled"] * 100).round(1)
        base = base.merge(ca, on="email", how="left")

        # Video-level aggregates
        if not self.video_activity_df.empty:
            vid_cols = self.video_activity_df.columns
            vid_agg = {"videos_watched": ("video_id", "count")}
            if "completion_pct" in vid_cols:
                vid_agg["avg_video_completion"] = ("completion_pct", "mean")
            if "watch_seconds" in vid_cols:
                vid_agg["total_watch_seconds"] = ("watch_seconds", "sum")
            if "watch_count" in vid_cols:
                vid_agg["total_rewatches"] = ("watch_count", "sum")

            va = self.video_activity_df.groupby("email").agg(**vid_agg).reset_index()
            base = base.merge(va, on="email", how="left")

        # Specialization-level aggregates
        if not self.spec_activity_df.empty and "spec_id" in self.spec_activity_df.columns:
            spec_agg = {"specs_enrolled": ("spec_id", "count")}
            if "is_completed" in self.spec_activity_df.columns:
                spec_agg["specs_completed"] = ("is_completed", "sum")
            if "progress_pct" in self.spec_activity_df.columns:
                spec_agg["avg_spec_progress"] = ("progress_pct", "mean")

            sa = self.spec_activity_df.groupby("email").agg(**spec_agg).reset_index()
            base = base.merge(sa, on="email", how="left")

        # Program-level aggregates
        if not self.program_activity_df.empty and "program_id" in self.program_activity_df.columns:
            prog_agg = {"programs_enrolled": ("program_id", "count")}
            if "is_completed" in self.program_activity_df.columns:
                prog_agg["programs_completed"] = ("is_completed", "sum")
            if "progress_pct" in self.program_activity_df.columns:
                prog_agg["avg_program_progress"] = ("progress_pct", "mean")

            pa = self.program_activity_df.groupby("email").agg(**prog_agg).reset_index()
            base = base.merge(pa, on="email", how="left")

        return base
