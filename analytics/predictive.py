"""Predictive Analytics — dropout, completion, high performer identification.

Key design principle: features must be BEHAVIORAL SIGNALS that are available
before the outcome is known. We never use progress, completion count, or
grades as features because those ARE the outcome.

Behavioral signals used:
  - Learning hours (effort invested)
  - Video watch patterns (engagement depth)
  - Recency of activity (momentum)
  - Number of courses enrolled (ambition/load)
  - Engagement score components (hours, video, recency — NOT progress)
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

logger = logging.getLogger(__name__)

# Features that directly encode the outcome — NEVER use these for prediction
OUTCOME_FEATURES = {
    "avg_progress", "max_progress", "courses_completed",
    "completion_rate", "avg_grade", "engagement_score",
    "progress_component",
}


class PredictiveAnalytics:
    """ML-based predictions for dropout, completion, and high performers."""

    MODELS = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }

    def __init__(self, test_size: float = 0.2, cv_folds: int = 5, random_state: int = 42):
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self._trained_models: dict = {}

    def prepare_features(
        self,
        course_activity: pd.DataFrame,
        engagement_scores: pd.DataFrame | None = None,
        video_activity: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build a learner-level feature matrix.

        Separates BEHAVIORAL features (used for prediction) from
        OUTCOME features (used only as labels).
        """
        # ── Outcome columns (labels only, not model inputs) ─────────
        outcome_agg = {
            "avg_progress": ("progress_pct", "mean"),
            "max_progress": ("progress_pct", "max"),
            "courses_completed": ("is_completed", "sum"),
        }
        outcomes = course_activity.groupby("email").agg(**outcome_agg).reset_index()
        outcomes["completion_rate"] = (
            outcomes["courses_completed"] / course_activity.groupby("email")["course_id"].count().values
        ).fillna(0)

        # ── Behavioral features (safe for prediction) ───────────────
        behavioral_agg = {
            "courses_enrolled": ("course_id", "count"),
        }
        if "learning_hours" in course_activity.columns:
            behavioral_agg["total_hours"] = ("learning_hours", "sum")
            behavioral_agg["avg_hours_per_course"] = ("learning_hours", "mean")
            behavioral_agg["std_hours"] = ("learning_hours", "std")

        features = course_activity.groupby("email").agg(**behavioral_agg).reset_index()

        # Time-based features
        if "enrollment_ts" in course_activity.columns and "last_activity_ts" in course_activity.columns:
            time_feats = course_activity.copy()
            time_feats["enrollment_ts"] = pd.to_datetime(time_feats["enrollment_ts"], errors="coerce")
            time_feats["last_activity_ts"] = pd.to_datetime(time_feats["last_activity_ts"], errors="coerce")

            time_agg = time_feats.groupby("email").agg(
                first_enrollment=("enrollment_ts", "min"),
                last_activity=("last_activity_ts", "max"),
                enrollment_span_days=("enrollment_ts", lambda x: (x.max() - x.min()).days),
            ).reset_index()

            ref_date = pd.Timestamp.now()
            time_agg["days_since_last_activity"] = (ref_date - time_agg["last_activity"]).dt.days.clip(lower=0)
            time_agg["account_age_days"] = (ref_date - time_agg["first_enrollment"]).dt.days.clip(lower=0)

            # Learning velocity: hours per active week
            if "total_hours" in features.columns:
                features = features.merge(time_agg[["email", "days_since_last_activity",
                                                     "account_age_days", "enrollment_span_days"]], on="email", how="left")
                features["hours_per_week"] = (
                    features["total_hours"] / (features["account_age_days"] / 7).clip(lower=1)
                ).round(3)
            else:
                features = features.merge(time_agg[["email", "days_since_last_activity",
                                                     "account_age_days", "enrollment_span_days"]], on="email", how="left")

        # Engagement score COMPONENTS (excluding progress_component which leaks)
        if engagement_scores is not None and not engagement_scores.empty:
            safe_eng_cols = ["email"]
            for col in ["hours_component", "video_component", "recency_component"]:
                if col in engagement_scores.columns:
                    safe_eng_cols.append(col)
            if len(safe_eng_cols) > 1:
                features = features.merge(engagement_scores[safe_eng_cols], on="email", how="left")

        # Video behavioral features
        if video_activity is not None and not video_activity.empty:
            vid_feats = video_activity.groupby("email").agg(
                videos_watched=("video_id", "count"),
                unique_videos=("video_id", "nunique"),
                avg_video_completion=("completion_pct", "mean"),
                total_watch_seconds=("watch_seconds", "sum"),
                total_rewatches=("watch_count", "sum"),
                avg_rewatch_rate=("watch_count", "mean"),
            ).reset_index()
            vid_feats["rewatch_ratio"] = (
                vid_feats["total_rewatches"] / vid_feats["videos_watched"]
            ).fillna(0).round(3)
            features = features.merge(vid_feats, on="email", how="left")

        features = features.fillna(0)

        # Merge outcome columns for label creation (kept separate)
        features = features.merge(outcomes, on="email", how="left")

        return features

    def _get_safe_feature_cols(self, df: pd.DataFrame, extra_exclude: set | None = None) -> list[str]:
        """Return feature columns excluding email, outcomes, and any extra columns."""
        exclude = {"email"} | OUTCOME_FEATURES
        if extra_exclude:
            exclude |= extra_exclude
        return [c for c in df.columns if c not in exclude]

    def predict_dropout(
        self,
        features_df: pd.DataFrame,
        model_type: str = "random_forest",
        dropout_threshold: float = 25.0,
    ) -> dict:
        """Predict which learners are likely to drop out.

        Dropout definition: avg_progress < threshold AND zero completions.
        Features used: behavioral signals ONLY (hours, video, recency, enrollment count).
        """
        df = features_df.copy()
        df["is_dropout"] = (
            (df["avg_progress"] < dropout_threshold) & (df["courses_completed"] == 0)
        ).astype(int)

        feature_cols = self._get_safe_feature_cols(df, {"is_dropout"})
        return self._train_classifier(df, feature_cols, "is_dropout", model_type, "dropout")

    def predict_completion(
        self,
        features_df: pd.DataFrame,
        model_type: str = "random_forest",
    ) -> dict:
        """Predict whether a learner will complete at least one course.

        Features used: behavioral signals ONLY.
        """
        df = features_df.copy()
        df["will_complete"] = (df["courses_completed"] > 0).astype(int)

        feature_cols = self._get_safe_feature_cols(df, {"will_complete"})
        return self._train_classifier(df, feature_cols, "will_complete", model_type, "completion")

    def identify_high_performers(
        self,
        features_df: pd.DataFrame,
        model_type: str = "random_forest",
        percentile: float = 75.0,
    ) -> dict:
        """Identify learners likely to be high performers.

        High performer = top percentile by avg_progress.
        Features used: behavioral signals ONLY.
        """
        df = features_df.copy()
        threshold = df["avg_progress"].quantile(percentile / 100)
        df["is_high_performer"] = (df["avg_progress"] >= threshold).astype(int)

        feature_cols = self._get_safe_feature_cols(df, {"is_high_performer"})
        return self._train_classifier(df, feature_cols, "is_high_performer", model_type, "high_performer")

    def _train_classifier(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        model_type: str,
        task_name: str,
    ) -> dict:
        X = df[feature_cols].values
        y = df[target_col].values

        if len(np.unique(y)) < 2:
            return {"error": f"Only one class present for {task_name}. Cannot train."}

        # Log feature list for transparency
        logger.info(f"[{task_name}] Training with {len(feature_cols)} features: {feature_cols}")
        logger.info(f"[{task_name}] Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        model_cls = self.MODELS.get(model_type, RandomForestClassifier)
        model = model_cls(random_state=self.random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Cross-validation
        cv_folds = min(self.cv_folds, max(2, min(np.bincount(y))))
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring="accuracy")

        # Feature importances
        importances = {}
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(feature_cols, model.feature_importances_.round(4).tolist()))
        elif hasattr(model, "coef_"):
            importances = dict(zip(feature_cols, np.abs(model.coef_[0]).round(4).tolist()))

        # Sort importances descending
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

        # Predictions for all learners
        all_proba = model.predict_proba(scaler.transform(df[feature_cols].values))[:, 1]
        df = df.copy()
        df[f"{task_name}_probability"] = all_proba.round(3)

        self._trained_models[task_name] = {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
        }

        result = {
            "task": task_name,
            "model_type": model_type,
            "features_used": feature_cols,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "cv_mean_accuracy": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "feature_importances": importances,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "class_distribution": {"positive": int(y.sum()), "negative": int(len(y) - y.sum())},
        }
        if y_proba is not None:
            try:
                result["auc_roc"] = round(roc_auc_score(y_test, y_proba), 4)
            except ValueError:
                result["auc_roc"] = None

        # Top at-risk learners
        risk_df = df[["email", f"{task_name}_probability"]].sort_values(
            f"{task_name}_probability", ascending=(task_name != "dropout")
        )
        result["top_flagged"] = risk_df.head(20).to_dict(orient="records")

        return result

    def get_all_predictions(
        self,
        course_activity: pd.DataFrame,
        engagement_scores: pd.DataFrame | None = None,
        video_activity: pd.DataFrame | None = None,
    ) -> dict:
        """Run all predictive models and return results."""
        features = self.prepare_features(course_activity, engagement_scores, video_activity)
        return {
            "dropout": self.predict_dropout(features),
            "completion": self.predict_completion(features),
            "high_performer": self.identify_high_performers(features),
        }
