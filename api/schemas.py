"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class UploadResponse(BaseModel):
    source_type: str
    filename: str
    rows: int
    columns: int
    status: str = "success"


class CleaningReport(BaseModel):
    original_rows: int
    original_cols: int
    cleaned_rows: int
    cleaned_cols: int
    rows_removed: int
    cols_removed: int


class CompletionSummary(BaseModel):
    total_enrollments: int
    total_completed: int
    completion_rate: float
    avg_progress: float
    median_progress: float


class EngagementScoreResponse(BaseModel):
    email: str
    score: float
    category: str
    progress_component: float
    hours_component: float
    video_component: float
    recency_component: float


class InsightResponse(BaseModel):
    category: str
    severity: str
    title: str
    description: str
    metric_value: Optional[float] = None
    generated_at: str


class PredictionResult(BaseModel):
    task: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: Optional[float] = None
    cv_mean_accuracy: float


class SkillGapEntry(BaseModel):
    skill_name: str
    skill_category: str
    avg_score: float
    coverage_pct: float
    gap_index: float


class LearnerProfile(BaseModel):
    email: str
    name: Optional[str] = None
    business_unit: Optional[str] = None
    courses_enrolled: int = 0
    courses_completed: int = 0
    completion_rate: float = 0
    total_hours: float = 0
    engagement_score: Optional[float] = None
    engagement_category: Optional[str] = None


class DashboardKPIs(BaseModel):
    total_learners: int
    active_learners: int
    total_courses: int
    overall_completion_rate: float
    avg_engagement_score: float
    at_risk_count: int
    total_learning_hours: float
    avg_hours_per_learner: float
