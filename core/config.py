"""Application configuration with environment variable overrides."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class DatabaseConfig:
    url: str = f"sqlite:///{BASE_DIR / 'output' / 'analytics.db'}"
    echo: bool = False


@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    enabled: bool = False  # optional — falls back to in-memory cache


@dataclass
class PipelineConfig:
    chunk_size: int = 50_000
    missing_numeric_strategy: str = "median"
    missing_categorical_strategy: str = "mode"
    missing_threshold: float = 0.95
    duplicate_keep: str = "first"
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class AnalyticsConfig:
    significance_level: float = 0.05
    correlation_method: str = "pearson"
    ml_test_size: float = 0.2
    ml_cv_folds: int = 5
    n_clusters: int = 4
    engagement_weights: dict = field(default_factory=lambda: {
        "progress_pct": 0.25,
        "learning_hours": 0.25,
        "video_interactions": 0.25,
        "recency_score": 0.25,
    })


@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8050
    debug: bool = False
    theme: str = "plotly_white"
    color_palette: list = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ])


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class ReportConfig:
    output_dir: str = str(BASE_DIR / "output" / "reports")
    export_dir: str = str(BASE_DIR / "output" / "exports")
    decimal_places: int = 3
    include_plots: bool = True


@dataclass
class Settings:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    api: APIConfig = field(default_factory=APIConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    data_dir: str = str(BASE_DIR / "data" / "coursera")
    base_dir: str = str(BASE_DIR)

    def __post_init__(self):
        # Allow env overrides
        if url := os.getenv("DATABASE_URL"):
            self.db.url = url
        if os.getenv("REDIS_ENABLED", "").lower() == "true":
            self.redis.enabled = True
        if host := os.getenv("REDIS_HOST"):
            self.redis.host = host
        os.makedirs(self.report.output_dir, exist_ok=True)
        os.makedirs(self.report.export_dir, exist_ok=True)


settings = Settings()
