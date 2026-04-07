"""Predictive Analytics — dropout, completion, high performer identification."""

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
    roc_auc_score, confusion_matrix, classification_report,
)

logger = logging.getLogger(__name__)


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
        """Build a learner-level feature matrix for ML models."""
        agg_dict = {
            "courses_enrolled": ("course_id", "count"),
            "avg_progress": ("progress_pct", "mean"),
            "max_progress": ("progress_pct", "max"),
            "courses_completed": ("is_completed", "sum"),
        }
        if "learning_hours" in course_activity.columns:
            agg_dict["total_hours"] = ("learning_hours", "sum")
            agg_dict["avg_hours_per_course"] = ("learning_hours", "mean")
        if "grade" in course_activity.columns:
            agg_dict["avg_grade"] = ("grade", "mean")

        features = course_activity.groupby("email").agg(**agg_dict).reset_index()

        features["completion_rate"] = (
            features["courses_completed"] / features["courses_enrolled"]
        ).fillna(0)

        # Add engagement scores
        if engagement_scores is not None and not engagement_scores.empty:
            features = features.merge(
                engagement_scores[["email", "score", "progress_component",
                                   "hours_component", "video_component",
                                   "recency_component"]],
                on="email", how="left",
            )
            features = features.rename(columns={"score": "engagement_score"})

        # Add video features
        if video_activity is not None and not video_activity.empty:
            vid_feats = video_activity.groupby("email").agg(
                videos_watched=("video_id", "count"),
                avg_video_completion=("completion_pct", "mean"),
                total_rewatches=("watch_count", "sum"),
            ).reset_index()
            features = features.merge(vid_feats, on="email", how="left")

        features = features.fillna(0)
        return features

    def predict_dropout(
        self,
        features_df: pd.DataFrame,
        model_type: str = "random_forest",
        dropout_threshold: float = 25.0,
    ) -> dict:
        """Predict which learners are likely to drop out.

        Dropout = avg_progress < threshold and no completions.
        """
        df = features_df.copy()
        df["is_dropout"] = ((df["avg_progress"] < dropout_threshold) & (df["courses_completed"] == 0)).astype(int)

        feature_cols = [c for c in df.columns if c not in ["email", "is_dropout"]]
        return self._train_classifier(df, feature_cols, "is_dropout", model_type, "dropout")

    def predict_completion(
        self,
        features_df: pd.DataFrame,
        model_type: str = "random_forest",
    ) -> dict:
        """Predict completion likelihood (will the learner complete at least one course?)."""
        df = features_df.copy()
        df["will_complete"] = (df["courses_completed"] > 0).astype(int)

        feature_cols = [c for c in df.columns if c not in ["email", "will_complete", "courses_completed", "completion_rate"]]
        return self._train_classifier(df, feature_cols, "will_complete", model_type, "completion")

    def identify_high_performers(
        self,
        features_df: pd.DataFrame,
        model_type: str = "random_forest",
        percentile: float = 75.0,
    ) -> dict:
        """Identify learners likely to be high performers."""
        df = features_df.copy()
        threshold = df["avg_progress"].quantile(percentile / 100)
        df["is_high_performer"] = (df["avg_progress"] >= threshold).astype(int)

        feature_cols = [c for c in df.columns if c not in ["email", "is_high_performer"]]
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
        cv_scores = cross_val_score(model, X_scaled, y, cv=min(self.cv_folds, len(y) // 5 or 2), scoring="accuracy")

        # Feature importances
        importances = {}
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(feature_cols, model.feature_importances_.round(4).tolist()))
        elif hasattr(model, "coef_"):
            importances = dict(zip(feature_cols, np.abs(model.coef_[0]).round(4).tolist()))

        # Predictions for all learners
        all_proba = model.predict_proba(scaler.transform(df[feature_cols].values))[:, 1]
        df[f"{task_name}_probability"] = all_proba.round(3)

        self._trained_models[task_name] = {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
        }

        result = {
            "task": task_name,
            "model_type": model_type,
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
            result["auc_roc"] = round(roc_auc_score(y_test, y_proba), 4)

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
