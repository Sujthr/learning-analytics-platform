"""Data Cleaning Engine for the Learning Analytics Platform.

Provides missing-value imputation, duplicate removal, timestamp normalization,
and automatic type inference.
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and normalizes DataFrames according to configurable strategies."""

    def __init__(self, config: dict):
        self.config = config
        mv = config.get("missing_values", {})
        self.numeric_strategy = mv.get("numeric_strategy", "median")
        self.categorical_strategy = mv.get("categorical_strategy", "mode")
        self.missing_threshold = mv.get("threshold", 0.5)

        dup = config.get("duplicates", {})
        self.dup_subset = dup.get("subset")
        self.dup_keep = dup.get("keep", "first")

        ts = config.get("timestamp", {})
        self.ts_format = ts.get("target_format", "%Y-%m-%d %H:%M:%S")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline.

        Order: type inference -> duplicates -> missing values -> timestamps.
        """
        logger.info("Cleaning DataFrame with %d rows, %d columns", len(df), len(df.columns))
        df = df.copy()
        df = self.infer_types(df)
        df = self.remove_duplicates(df, subset=self.dup_subset, keep=self.dup_keep)
        df = self.handle_missing(
            df,
            numeric_strategy=self.numeric_strategy,
            categorical_strategy=self.categorical_strategy,
            threshold=self.missing_threshold,
        )
        df = self.normalize_timestamps(df, self.ts_format)
        logger.info("Cleaning complete: %d rows, %d columns", len(df), len(df.columns))
        return df

    def handle_missing(
        self,
        df: pd.DataFrame,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Handle missing values according to the specified strategies.

        Args:
            df: Input DataFrame.
            numeric_strategy: One of 'mean', 'median', 'drop', 'zero'.
            categorical_strategy: One of 'mode', 'drop', 'unknown'.
            threshold: Drop columns with more than this fraction missing.
        """
        df = df.copy()

        # Drop columns exceeding threshold
        missing_frac = df.isnull().mean()
        drop_cols = missing_frac[missing_frac > threshold].index.tolist()
        if drop_cols:
            logger.info("Dropping %d columns exceeding %.0f%% missing: %s",
                        len(drop_cols), threshold * 100, drop_cols)
            df.drop(columns=drop_cols, inplace=True)

        # Numeric columns
        num_cols = df.select_dtypes(include=["number"]).columns
        if numeric_strategy == "mean":
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif numeric_strategy == "median":
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        elif numeric_strategy == "zero":
            df[num_cols] = df[num_cols].fillna(0)
        elif numeric_strategy == "drop":
            df.dropna(subset=num_cols, inplace=True)

        # Categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if categorical_strategy == "mode":
            for col in cat_cols:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])
        elif categorical_strategy == "unknown":
            df[cat_cols] = df[cat_cols].fillna("unknown")
        elif categorical_strategy == "drop":
            df.dropna(subset=cat_cols, inplace=True)

        return df

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[list] = None,
        keep: str = "first",
    ) -> pd.DataFrame:
        """Remove duplicate rows."""
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep if keep != "false" else False)
        removed = before - len(df)
        if removed:
            logger.info("Removed %d duplicate rows", removed)
        return df

    def normalize_timestamps(
        self,
        df: pd.DataFrame,
        target_format: str = "%Y-%m-%d %H:%M:%S",
        columns: Optional[list] = None,
    ) -> pd.DataFrame:
        """Detect and normalize timestamp columns to a standard format."""
        df = df.copy()
        ts_cols = columns or self._detect_timestamp_columns(df)

        for col in ts_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    logger.info("Normalized timestamp column: %s", col)
                except Exception as e:
                    logger.warning("Failed to normalize %s: %s", col, e)

        return df

    def _detect_timestamp_columns(self, df: pd.DataFrame) -> list[str]:
        """Heuristically detect columns that contain timestamps."""
        ts_patterns = re.compile(
            r"(timestamp|datetime|date|time|created|updated|_at$|_on$|start|end|session)",
            re.IGNORECASE,
        )
        candidates = [c for c in df.columns if ts_patterns.search(c)]

        # Also check object columns with date-like values
        for col in df.select_dtypes(include=["object"]).columns:
            if col not in candidates:
                sample = df[col].dropna().head(20)
                try:
                    pd.to_datetime(sample)
                    candidates.append(col)
                except (ValueError, TypeError):
                    pass

        return candidates

    def infer_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligently infer and convert column data types."""
        df = df.copy()
        for col in df.select_dtypes(include=["object"]).columns:
            # Try numeric
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() / max(df[col].notna().sum(), 1) > 0.8:
                    df[col] = converted
                    logger.debug("Inferred %s as numeric", col)
                    continue
            except (ValueError, TypeError):
                pass

        return df

    def get_cleaning_report(
        self, df_before: pd.DataFrame, df_after: pd.DataFrame
    ) -> dict:
        """Return a summary of what changed during cleaning."""
        return {
            "rows_before": len(df_before),
            "rows_after": len(df_after),
            "rows_removed": len(df_before) - len(df_after),
            "cols_before": len(df_before.columns),
            "cols_after": len(df_after.columns),
            "cols_removed": len(df_before.columns) - len(df_after.columns),
            "missing_before": int(df_before.isnull().sum().sum()),
            "missing_after": int(df_after.isnull().sum().sum()),
            "missing_filled": int(df_before.isnull().sum().sum()) - int(df_after.isnull().sum().sum()),
        }
