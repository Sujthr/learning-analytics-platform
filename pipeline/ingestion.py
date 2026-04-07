"""Data Ingestion — validate schema, load CSV/Excel, normalize column names.

Supports both:
  - Real Coursera enterprise exports (actual column names from Coursera platform)
  - Generated sample data (column names from our data generator)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Column name mapping: Coursera enterprise export -> normalized internal names
# Left = real Coursera column, Right = our internal name
COLUMN_MAP = {
    "course": {
        # Identity
        "Name": "Name",
        "Email": "Email",
        "External Id": "External ID",
        "External ID": "External ID",
        # Course info
        "Course": "Course Name",
        "Course Name": "Course Name",
        "Course Id": "Course Id",
        "course_id": "Course Id",
        "Course Slug": "Course Slug",
        "University": "Institution",
        "Institution": "Institution",
        # Timestamps
        "Enrollment Time": "Enrollment Timestamp",
        "Enrollment Timestamp": "Enrollment Timestamp",
        "Completion Time": "Completion Timestamp",
        "Completion Timestamp": "Completion Timestamp",
        "Last Course Activity Time": "Last Activity Timestamp",
        "Last Activity Timestamp": "Last Activity Timestamp",
        # Metrics
        "Overall Progress": "Progress (%)",
        "Progress (%)": "Progress (%)",
        "Course Grade": "Grade (%)",
        "Grade (%)": "Grade (%)",
        "Learning Hours": "Learning Hours",
        "Estimated Learning Hours": "Estimated Learning Hours",
        "Completed": "Completed",
        "Course Type": "Course Type",
        # Org data
        "Business Unit": "Business Unit",
        "Business Unit 2": "Business Unit 2",
        "Job Title": "Role",
        "Role": "Role",
        "Location City": "Location",
        "Location": "Location",
        "Location Region": "Location Region",
        "Location Country": "Location Country",
        "Program Name": "Program Name",
        "Program Slug": "Program Slug",
        "Manager Name": "Manager Name",
        "Manager Email": "Manager Email",
        "Removed From Program": "Removed From Program",
    },
    "specialization": {
        "Name": "Name",
        "Email": "Email",
        "External Id": "External ID",
        "External ID": "External ID",
        "Specialization": "Specialization Name",
        "Specialization Name": "Specialization Name",
        "Specialization Id": "Specialization Id",
        "Specialization Slug": "Specialization Slug",
        "University": "Institution",
        "Enrollment Time": "Enrollment Timestamp",
        "Enrollment Timestamp": "Enrollment Timestamp",
        "Last Specialization Activity Time": "Last Activity Timestamp",
        "Last Activity Timestamp": "Last Activity Timestamp",
        "Specialization Completion Time": "Completion Timestamp",
        "# Completed Courses": "Courses Completed",
        "Courses Completed": "Courses Completed",
        "# Courses in Specialization": "Total Courses",
        "Total Courses": "Total Courses",
        "Completed": "Completed",
        "Learning Hours": "Learning Hours",
        "Program Name": "Program Name",
        "Program Slug": "Program Slug",
        "Business Unit": "Business Unit",
        "Job Title": "Role",
        "Location City": "Location",
        "Location Region": "Location Region",
        "Location Country": "Location Country",
        "Removed From Program": "Removed From Program",
    },
    "video": {
        "Name": "Name",
        "Email": "Email",
        "External Id": "External ID",
        "External ID": "External ID",
        "Course": "Course Name",
        "Course Name": "Course Name",
        "Course Id": "Course Id",
        "Course Slug": "Course Slug",
        "Clip Name": "Video Id",
        "Video Id": "Video Id",
        "Clip Title": "Video Name",
        "Video Name": "Video Name",
        "Clip Url": "Video URL",
        "First Video Clip Activity Time": "First Watch Timestamp",
        "Last Video Clip Activity Time": "Last Watch Timestamp",
        "Last Watch Timestamp": "Last Watch Timestamp",
        "Progress Percentage": "Completion (%)",
        "Completion (%)": "Completion (%)",
        "# of Times Watched": "Watch Count",
        "Watch Count": "Watch Count",
        "Completed at least 90% of clip": "Video Completed",
        "Learning Hours": "Learning Hours",
        "Estimated Learning Hours": "Estimated Learning Hours",
        "Led to Course Enrollment": "Led to Enrollment",
        "Program Name": "Program Name",
        "Business Unit": "Business Unit",
        "Job Title": "Role",
        "Location City": "Location",
        "Removed From Program": "Removed From Program",
    },
    "program": {
        "Name": "Name",
        "Email": "Email",
        "External Id": "External ID",
        "External ID": "External ID",
        "Program Id": "Program Id",
        "Program Name": "Program Name",
        "Program Slug": "Program Slug",
        "Total Courses in Program": "Total Courses",
        "Courses Completed": "Courses Completed",
        "Progress (%)": "Progress (%)",
        "Learning Hours": "Learning Hours",
        "Completed": "Completed",
        "Enrollment Timestamp": "Enrollment Timestamp",
        "Last Activity Timestamp": "Last Activity Timestamp",
        "Business Unit": "Business Unit",
        "Job Title": "Role",
        "Location City": "Location",
    },
}

# Minimum required columns after normalization
REQUIRED_COLS = {
    "course": ["Email", "Course Name"],
    "specialization": ["Email", "Specialization Name"],
    "video": ["Email", "Course Name"],
    "program": ["Email", "Program Name"],
}


class DataIngestor:
    """Load and normalize Coursera CSV/Excel datasets."""

    def __init__(self, chunk_size: int = 50_000):
        self.chunk_size = chunk_size
        self._load_log: list[dict] = []

    def ingest(
        self,
        file_path: str,
        source_type: str,
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        """Read a file, normalize columns, and validate.

        Args:
            file_path: Path to CSV or Excel file.
            source_type: One of 'course', 'program', 'specialization', 'video'.
            encoding: File encoding.

        Returns:
            DataFrame with normalized column names.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext == ".csv":
            df = self._read_csv(path, encoding)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # Normalize column names
        df = self._normalize_columns(df, source_type)

        # Validate
        self._validate_schema(df, source_type, path.name)

        # Attach metadata
        df.attrs["source_type"] = source_type
        df.attrs["source_file"] = path.name
        df.attrs["row_count"] = len(df)

        self._load_log.append({
            "file": path.name,
            "source_type": source_type,
            "rows": len(df),
            "columns": len(df.columns),
        })
        logger.info(f"Ingested {path.name}: {len(df)} rows, {len(df.columns)} cols")
        return df

    def _read_csv(self, path: Path, encoding: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            logger.warning(f"Encoding {encoding} failed, trying latin-1")
            return pd.read_csv(path, encoding="latin-1")

    def _normalize_columns(self, df: pd.DataFrame, source_type: str) -> pd.DataFrame:
        """Rename columns from Coursera export format to our internal names."""
        col_map = COLUMN_MAP.get(source_type, {})
        if not col_map:
            return df

        rename = {}
        for orig_col in df.columns:
            if orig_col in col_map:
                rename[orig_col] = col_map[orig_col]

        if rename:
            df = df.rename(columns=rename)
            logger.info(f"Normalized {len(rename)} column names for {source_type}")

        return df

    def _validate_schema(self, df: pd.DataFrame, source_type: str, filename: str):
        required = REQUIRED_COLS.get(source_type, [])
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"Schema validation failed for {filename} ({source_type}): "
                f"missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

    def ingest_all(self, data_dir: str) -> dict[str, pd.DataFrame]:
        """Auto-detect and load Coursera datasets from a directory.

        Looks for files by keyword matching:
          - 'course' (not 'specialization') -> course
          - 'program' -> program
          - 'speciali' -> specialization
          - 'video' or 'clip' -> video
        """
        data_dir = Path(data_dir)
        all_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.xlsx"))

        if not all_files:
            raise FileNotFoundError(f"No CSV/Excel files found in {data_dir}")

        datasets = {}
        used_files = set()

        # Detect file types by name keywords
        type_keywords = [
            ("video", ["video", "clip"]),
            ("specialization", ["speciali"]),
            ("program", ["program"]),
            ("course", ["course"]),
        ]

        for source_type, keywords in type_keywords:
            for f in all_files:
                if f in used_files:
                    continue
                fname_lower = f.name.lower()
                if any(kw in fname_lower for kw in keywords):
                    # Prefer the richer file (with Name, Business Unit, etc.)
                    # i.e., skip "Modified" versions if original exists
                    if "modified" in fname_lower:
                        # Check if a non-modified version exists
                        has_original = any(
                            g != f and g not in used_files
                            and any(kw in g.name.lower() for kw in keywords)
                            and "modified" not in g.name.lower()
                            for g in all_files
                        )
                        if has_original:
                            logger.info(f"Skipping modified file: {f.name}")
                            used_files.add(f)
                            continue

                    try:
                        datasets[source_type] = self.ingest(str(f), source_type)
                        used_files.add(f)
                        break  # one file per type
                    except Exception as e:
                        logger.warning(f"Failed to ingest {f.name} as {source_type}: {e}")

        if not datasets:
            raise ValueError(f"No valid datasets found in {data_dir}")

        return datasets

    @property
    def load_summary(self) -> list[dict]:
        return self._load_log
