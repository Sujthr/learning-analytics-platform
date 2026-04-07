"""Data Cleaning — missing values, duplicates, type inference, timestamp normalization."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and standardize Coursera datasets."""

    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        missing_threshold: float = 0.5,
        duplicate_keep: str = "first",
    ):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.missing_threshold = missing_threshold
        self.duplicate_keep = duplicate_keep
        self._report: dict = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full cleaning pipeline: types -> duplicates -> missing -> timestamps."""
        original_shape = df.shape
        df = df.copy()

        df = self._drop_high_null_columns(df)
        df = self._infer_types(df)
        df = self._remove_duplicates(df)
        df = self._handle_missing(df)
        df = self._normalize_timestamps(df)
        df = self._clean_percentages(df)

        self._report = {
            "original_rows": original_shape[0],
            "original_cols": original_shape[1],
            "cleaned_rows": len(df),
            "cleaned_cols": len(df.columns),
            "rows_removed": original_shape[0] - len(df),
            "cols_removed": original_shape[1] - len(df.columns),
        }
        logger.info(
            f"Cleaning: {original_shape} -> {df.shape} "
            f"({self._report['rows_removed']} rows, {self._report['cols_removed']} cols removed)"
        )
        return df

    def _drop_high_null_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        null_pct = df.isnull().mean()
        drop_cols = null_pct[null_pct > self.missing_threshold].index.tolist()
        if drop_cols:
            logger.info(f"Dropping high-null columns: {drop_cols}")
            df = df.drop(columns=drop_cols)
        return df

    def _infer_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == object:
                # Try numeric
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > 0.5 * df[col].notna().sum():
                    df[col] = converted
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        if self.duplicate_keep == "false":
            df = df.drop_duplicates(keep=False)
        else:
            df = df.drop_duplicates(keep=self.duplicate_keep)
        removed = before - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                if self.numeric_strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif self.numeric_strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif self.numeric_strategy == "zero":
                    df[col] = df[col].fillna(0)
                elif self.numeric_strategy == "drop":
                    df = df.dropna(subset=[col])
            else:
                if self.categorical_strategy == "mode" and not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif self.categorical_strategy == "unknown":
                    df[col] = df[col].fillna("Unknown")
                elif self.categorical_strategy == "drop":
                    df = df.dropna(subset=[col])
        return df

    def _normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        ts_cols = [c for c in df.columns if "timestamp" in c.lower() or "date" in c.lower()]
        for col in ts_cols:
            if df[col].dtype == object:
                df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
        return df

    def _clean_percentages(self, df: pd.DataFrame) -> pd.DataFrame:
        pct_cols = [c for c in df.columns if "(%)" in c or "pct" in c.lower()]
        for col in pct_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].clip(0, 100)
        return df

    @property
    def cleaning_report(self) -> dict:
        return self._report
