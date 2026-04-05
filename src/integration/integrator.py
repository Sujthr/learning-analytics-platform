"""Data Integration Layer for the Learning Analytics Platform.

Joins multi-source datasets using mapping tables with conflict resolution
and referential integrity checks.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataIntegrator:
    """Integrates datasets from multiple sources via mapping tables."""

    def __init__(self, config: dict):
        self.config = config
        self.join_key = config.get("join_key", "student_id")
        self.conflict_strategy = config.get("conflict_resolution", "latest")
        self.source_priority = config.get("source_priority", [])
        self.mapping_table: Optional[pd.DataFrame] = None
        self._report = {
            "total_sources": 0,
            "matched_records": 0,
            "unmatched_records": 0,
            "conflicts_resolved": 0,
        }

    def load_mapping_table(self, path: str) -> pd.DataFrame:
        """Load the student ID mapping table from CSV."""
        self.mapping_table = pd.read_csv(path)
        logger.info("Loaded mapping table: %d entries, columns=%s",
                     len(self.mapping_table), list(self.mapping_table.columns))
        return self.mapping_table

    def integrate(self, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all source datasets into a single unified DataFrame.

        Args:
            datasets: Dict of {source_name: DataFrame}.

        Returns:
            Merged DataFrame with unified student_id.
        """
        self._report["total_sources"] = len(datasets)

        if not datasets:
            return pd.DataFrame()

        # Map source-specific IDs to canonical student_id
        mapped = {}
        for source, df in datasets.items():
            mapped[source] = self._map_ids(df, source)

        # Aggregate each source to student-course level before joining
        aggregated = {}
        for source, df in mapped.items():
            aggregated[source] = self._aggregate_source(df, source)

        # Merge all sources
        result = self.join_datasets(aggregated, self.join_key)
        self._report["matched_records"] = len(result)

        # Resolve column conflicts
        result = self.resolve_conflicts(result, self.conflict_strategy)

        logger.info("Integration complete: %d rows, %d columns",
                     len(result), len(result.columns))
        return result

    def _map_ids(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Map source-specific student IDs to canonical student_id."""
        if self.mapping_table is None:
            return df

        # Determine which column in mapping table corresponds to this source
        source_col_map = {
            "coursera": "coursera_id",
            "lms": "lms_id",
            "academic": "academic_id",
        }
        source_col = source_col_map.get(source)
        if source_col and source_col in self.mapping_table.columns:
            mapping = self.mapping_table[["student_id", source_col]].copy()
            df = df.merge(
                mapping,
                left_on="student_id",
                right_on=source_col,
                how="left",
                suffixes=("_orig", ""),
            )
            # Replace the source-specific id with canonical id
            if "student_id_orig" in df.columns:
                df.drop(columns=["student_id_orig", source_col], inplace=True, errors="ignore")
            unmatched = df["student_id"].isna().sum()
            if unmatched:
                self._report["unmatched_records"] += unmatched
                logger.warning("%d records from '%s' could not be mapped", unmatched, source)
                df = df.dropna(subset=["student_id"])

        return df

    def _aggregate_source(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Aggregate raw activity/session data to student level."""
        if "student_id" not in df.columns:
            return df

        group_key = ["student_id"]
        if "course_id" in df.columns:
            group_key.append("course_id")

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in numeric_cols and c not in group_key]

        agg_dict = {}
        for col in numeric_cols:
            agg_dict[col] = "mean"
        for col in cat_cols:
            agg_dict[col] = "first"

        if not agg_dict:
            return df

        agg_df = df.groupby(group_key, as_index=False).agg(agg_dict)

        # Add record count
        counts = df.groupby(group_key).size().reset_index(name=f"{source}_record_count")
        agg_df = agg_df.merge(counts, on=group_key, how="left")

        logger.info("Aggregated '%s': %d -> %d rows", source, len(df), len(agg_df))
        return agg_df

    def join_datasets(
        self,
        datasets: dict[str, pd.DataFrame],
        join_key: str,
        mapping_table: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Join multiple DataFrames on a common key using outer merge."""
        dfs = list(datasets.values())
        names = list(datasets.keys())

        if not dfs:
            return pd.DataFrame()

        result = dfs[0].copy()
        for i, df in enumerate(dfs[1:], 1):
            # Determine join columns
            common_keys = [k for k in [join_key, "course_id"] if k in result.columns and k in df.columns]
            if not common_keys:
                common_keys = [join_key]

            result = result.merge(
                df,
                on=common_keys,
                how="outer",
                suffixes=(f"_{names[0]}" if i == 1 else "", f"_{names[i]}"),
            )

        return result

    def resolve_conflicts(self, df: pd.DataFrame, strategy: str = "latest") -> pd.DataFrame:
        """Resolve duplicate columns from merging (e.g., score_coursera, score_lms).

        Strategies:
            - latest: keep the first non-null value
            - source_priority: prefer values from higher-priority sources
        """
        df = df.copy()
        # Find groups of conflicting columns (same base name, different suffixes)
        base_cols = {}
        for col in df.columns:
            for src in self.source_priority + ["coursera", "lms", "academic"]:
                if col.endswith(f"_{src}"):
                    base = col[: -len(f"_{src}")]
                    base_cols.setdefault(base, []).append((col, src))
                    break

        for base, col_sources in base_cols.items():
            if len(col_sources) < 2:
                continue

            self._report["conflicts_resolved"] += 1

            if strategy == "source_priority" and self.source_priority:
                ordered = sorted(col_sources, key=lambda x: (
                    self.source_priority.index(x[1]) if x[1] in self.source_priority else 99
                ))
            else:
                ordered = col_sources

            # Coalesce: take first non-null across priority order
            df[base] = np.nan
            for col, _ in ordered:
                df[base] = df[base].fillna(df[col])

            df.drop(columns=[c for c, _ in col_sources], inplace=True)
            logger.debug("Resolved conflict for '%s' from %d sources", base, len(col_sources))

        return df

    def validate_integrity(self, df: pd.DataFrame) -> dict:
        """Check referential integrity and report issues."""
        issues = {}

        if "student_id" in df.columns:
            null_ids = df["student_id"].isna().sum()
            if null_ids:
                issues["null_student_ids"] = int(null_ids)

        if "course_id" in df.columns:
            null_courses = df["course_id"].isna().sum()
            if null_courses:
                issues["null_course_ids"] = int(null_courses)

        dup_count = df.duplicated(subset=["student_id", "course_id"], keep=False).sum() if {"student_id", "course_id"} <= set(df.columns) else 0
        if dup_count:
            issues["duplicate_student_course_pairs"] = int(dup_count)

        if issues:
            logger.warning("Integrity issues found: %s", issues)
        else:
            logger.info("Referential integrity check passed")

        return issues

    def get_integration_report(self) -> dict:
        """Return merge statistics."""
        return self._report.copy()
