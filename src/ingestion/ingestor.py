"""Data Ingestion Layer for the Learning Analytics Platform.

Handles reading CSV and Excel files with schema validation and chunked
processing for large datasets.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataIngestor:
    """Ingests data from CSV/Excel files with validation and chunked reading."""

    def __init__(self, config: dict):
        self.config = config
        self.chunk_size = config.get("chunk_size", 50000)
        self.encoding = config.get("encoding", "utf-8")
        self.sources = config.get("sources", {})

    def ingest(
        self,
        file_path: str,
        source_type: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """Read a CSV or Excel file and return a DataFrame.

        Args:
            file_path: Path to the data file.
            source_type: One of 'coursera', 'lms', 'academic' for schema validation.
            chunk_size: Override default chunk size for large files.

        Returns:
            DataFrame with metadata in .attrs (source_type, file_path, row_count).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        cs = chunk_size or self.chunk_size

        logger.info("Ingesting %s (format=%s, source=%s)", path.name, ext, source_type)

        if ext == ".csv":
            df = self._read_csv(path, cs)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path, engine="openpyxl")
        else:
            raise ValueError(f"Unsupported format: {ext}")

        if source_type:
            self.validate_schema(df, source_type)

        df.attrs["source_type"] = source_type
        df.attrs["file_path"] = str(path)
        df.attrs["row_count"] = len(df)

        logger.info("Ingested %d rows, %d columns from %s", len(df), len(df.columns), path.name)
        return df

    def _read_csv(self, path: Path, chunk_size: int) -> pd.DataFrame:
        """Read CSV, using chunks for files larger than chunk_size rows."""
        # Peek at the row count to decide on chunking
        try:
            sample = pd.read_csv(path, nrows=0, encoding=self.encoding)
        except UnicodeDecodeError:
            sample = pd.read_csv(path, nrows=0, encoding="latin-1")
            self.encoding = "latin-1"

        chunks = []
        for chunk in pd.read_csv(path, chunksize=chunk_size, encoding=self.encoding):
            chunks.append(chunk)

        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def ingest_directory(
        self,
        dir_path: str,
        pattern: str = "*.csv",
        source_type: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """Ingest all matching files from a directory.

        Returns:
            Dict mapping filename to DataFrame.
        """
        directory = Path(dir_path)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        results = {}
        for file_path in sorted(directory.glob(pattern)):
            try:
                results[file_path.stem] = self.ingest(str(file_path), source_type)
            except Exception as e:
                logger.error("Failed to ingest %s: %s", file_path.name, e)

        logger.info("Ingested %d files from %s", len(results), dir_path)
        return results

    def validate_schema(self, df: pd.DataFrame, source_type: str) -> bool:
        """Validate DataFrame columns against the expected schema.

        Args:
            df: DataFrame to validate.
            source_type: Source key in config (coursera, lms, academic).

        Returns:
            True if valid.

        Raises:
            ValueError: If required columns are missing.
        """
        if source_type not in self.sources:
            logger.warning("No schema defined for source type: %s", source_type)
            return True

        schema = self.sources[source_type].get("schema", {})
        required = set(schema.get("required_columns", []))
        present = set(df.columns)
        missing = required - present

        if missing:
            raise ValueError(
                f"Schema validation failed for '{source_type}'. "
                f"Missing required columns: {missing}"
            )

        logger.info("Schema validation passed for '%s'", source_type)
        return True
