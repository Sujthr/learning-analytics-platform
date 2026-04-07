"""Data Ingestion — validate schema, load CSV/Excel, handle encoding."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Expected schemas for each Coursera dataset type
SCHEMAS = {
    "course": {
        "required": ["Email", "External ID", "Course Id", "Course Name", "Progress (%)"],
        "optional": ["Name", "Business Unit", "Role", "Location", "Course Slug",
                      "Institution", "Enrollment Timestamp", "Completion Timestamp",
                      "Grade (%)", "Learning Hours", "Completed", "Last Activity Timestamp"],
    },
    "program": {
        "required": ["Email", "External ID", "Program Id", "Program Name", "Progress (%)"],
        "optional": ["Name", "Business Unit", "Program Slug", "Total Courses in Program",
                      "Courses Completed", "Learning Hours", "Completed",
                      "Enrollment Timestamp", "Last Activity Timestamp"],
    },
    "specialization": {
        "required": ["Email", "External ID", "Specialization Id", "Specialization Name", "Progress (%)"],
        "optional": ["Name", "Business Unit", "Specialization Slug", "Total Courses",
                      "Courses Completed", "Learning Hours", "Completed",
                      "Enrollment Timestamp", "Last Activity Timestamp"],
    },
    "video": {
        "required": ["Email", "External ID", "Course Id", "Video Id"],
        "optional": ["Course Name", "Video Name", "Watch Duration (seconds)",
                      "Total Duration (seconds)", "Completion (%)", "Watch Count",
                      "Last Watch Timestamp"],
    },
}


class DataIngestor:
    """Load and validate Coursera CSV/Excel datasets."""

    def __init__(self, chunk_size: int = 50_000):
        self.chunk_size = chunk_size
        self._load_log: list[dict] = []

    def ingest(
        self,
        file_path: str,
        source_type: str,
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        """Read a file and validate its schema.

        Args:
            file_path: Path to CSV or Excel file.
            source_type: One of 'course', 'program', 'specialization', 'video'.
            encoding: File encoding.

        Returns:
            Validated DataFrame.
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

        self._validate_schema(df, source_type, path.name)

        # attach metadata
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

    def _validate_schema(self, df: pd.DataFrame, source_type: str, filename: str):
        schema = SCHEMAS.get(source_type)
        if not schema:
            logger.warning(f"No schema defined for '{source_type}', skipping validation")
            return

        missing = [col for col in schema["required"] if col not in df.columns]
        if missing:
            raise ValueError(
                f"Schema validation failed for {filename} ({source_type}): "
                f"missing required columns: {missing}"
            )

    def ingest_all(self, data_dir: str) -> dict[str, pd.DataFrame]:
        """Auto-detect and load all 4 Coursera datasets from a directory."""
        data_dir = Path(data_dir)
        mapping = {
            "course_activity": "course",
            "program_activity": "program",
            "specialization_activity": "specialization",
            "video_clip_activity": "video",
        }

        datasets = {}
        for filename_prefix, source_type in mapping.items():
            matches = list(data_dir.glob(f"{filename_prefix}.*"))
            if matches:
                datasets[source_type] = self.ingest(str(matches[0]), source_type)
            else:
                logger.warning(f"No file matching '{filename_prefix}.*' in {data_dir}")

        return datasets

    @property
    def load_summary(self) -> list[dict]:
        return self._load_log
