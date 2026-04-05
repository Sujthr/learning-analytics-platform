"""Analytics Engine for the Learning Analytics Platform.

Provides EDA, hypothesis testing, and machine learning (regression,
classification, clustering) capabilities.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """Statistical analysis and machine learning engine."""

    def __init__(self, config: dict):
        self.config = config
        self.alpha = config.get("hypothesis_testing", {}).get("significance_level", 0.05)
        self.corr_method = config.get("eda", {}).get("correlation_method", "pearson")

    # ---- EDA ----

    def compute_distributions(
        self, df: pd.DataFrame, columns: Optional[list] = None
    ) -> dict:
        """Compute distribution statistics for numeric columns."""
        cols = columns or df.select_dtypes(include=["number"]).columns.tolist()
        result = {}
        for col in cols:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                continue
            result[col] = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "skew": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "min": float(series.min()),
                "max": float(series.max()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
            }
        logger.info("Computed distributions for %d columns", len(result))
        return result

    def compute_correlations(
        self,
        df: pd.DataFrame,
        method: Optional[str] = None,
        columns: Optional[list] = None,
    ) -> pd.DataFrame:
        """Compute correlation matrix."""
        m = method or self.corr_method
        cols = columns or df.select_dtypes(include=["number"]).columns.tolist()
        corr = df[cols].corr(method=m)
        logger.info("Computed %s correlation matrix (%dx%d)", m, len(cols), len(cols))
        return corr

    # ---- Hypothesis Testing ----

    def t_test(
        self,
        df: pd.DataFrame,
        column: str,
        group_column: str,
        group_a: str,
        group_b: str,
    ) -> dict:
        """Independent samples t-test between two groups."""
        a = df[df[group_column] == group_a][column].dropna()
        b = df[df[group_column] == group_b][column].dropna()

        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

        result = {
            "test": "independent_t_test",
            "column": column,
            "groups": [str(group_a), str(group_b)],
            "group_a_mean": float(a.mean()),
            "group_b_mean": float(b.mean()),
            "group_a_n": int(len(a)),
            "group_b_n": int(len(b)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < self.alpha),
            "alpha": self.alpha,
        }
        logger.info("T-test: t=%.3f, p=%.4f, significant=%s",
                     t_stat, p_value, result["significant"])
        return result

    def anova(self, df: pd.DataFrame, column: str, group_column: str) -> dict:
        """One-way ANOVA across groups."""
        groups = [
            group[column].dropna().values
            for _, group in df.groupby(group_column)
            if len(group[column].dropna()) > 0
        ]

        if len(groups) < 2:
            return {"test": "anova", "error": "Need at least 2 groups", "p_value": 1.0, "significant": False}

        f_stat, p_value = stats.f_oneway(*groups)

        result = {
            "test": "one_way_anova",
            "column": column,
            "group_column": group_column,
            "n_groups": len(groups),
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < self.alpha),
            "alpha": self.alpha,
        }
        logger.info("ANOVA: F=%.3f, p=%.4f, significant=%s",
                     f_stat, p_value, result["significant"])
        return result

    # ---- Machine Learning ----

    def regression(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[list] = None,
        model_type: str = "random_forest",
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> dict:
        """Train a regression model and return metrics."""
        X, y = self._prepare_data(df, target, features)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = self._get_regression_model(model_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cv_scores = cross_val_score(model, X, y, cv=min(cv_folds, len(X)), scoring="r2")

        result = {
            "model_type": model_type,
            "target": target,
            "features": list(X.columns),
            "r2": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(cv_scores.mean()),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "model": model,
        }

        if hasattr(model, "feature_importances_"):
            result["feature_importances"] = dict(
                zip(X.columns, model.feature_importances_.tolist())
            )

        logger.info("Regression (%s): R2=%.3f, RMSE=%.3f", model_type, result["r2"], result["rmse"])
        return result

    def classification(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[list] = None,
        model_type: str = "random_forest",
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> dict:
        """Train a classification model and return metrics."""
        X, y = self._prepare_data(df, target, features)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )

        model = self._get_classification_model(model_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cv_scores = cross_val_score(model, X, y, cv=min(cv_folds, len(X)), scoring="accuracy")

        avg = "binary" if len(y.unique()) == 2 else "weighted"
        result = {
            "model_type": model_type,
            "target": target,
            "features": list(X.columns),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(cv_scores.mean()),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "model": model,
        }

        if hasattr(model, "feature_importances_"):
            result["feature_importances"] = dict(
                zip(X.columns, model.feature_importances_.tolist())
            )

        logger.info("Classification (%s): acc=%.3f, F1=%.3f", model_type, result["accuracy"], result["f1"])
        return result

    def clustering(
        self,
        df: pd.DataFrame,
        features: list,
        method: str = "kmeans",
        n_clusters: int = 4,
    ) -> dict:
        """Perform clustering and return labels and metrics."""
        available = [f for f in features if f in df.columns]
        X = df[available].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
        elif method == "hierarchical":
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        labels = model.fit_predict(X_scaled)

        sil_score = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else 0.0

        result = {
            "method": method,
            "features": available,
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "labels": labels.tolist(),
            "silhouette_score": sil_score,
            "cluster_sizes": pd.Series(labels).value_counts().to_dict(),
        }

        if hasattr(model, "cluster_centers_"):
            centers = scaler.inverse_transform(model.cluster_centers_)
            result["cluster_centers"] = pd.DataFrame(
                centers, columns=available
            ).to_dict("records")

        logger.info("Clustering (%s): %d clusters, silhouette=%.3f",
                     method, result["n_clusters"], sil_score)
        return result

    # ---- Helpers ----

    def _prepare_data(
        self, df: pd.DataFrame, target: str, features: Optional[list]
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix X and target vector y."""
        if features:
            cols = [f for f in features if f in df.columns and f != target]
        else:
            cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target]

        subset = df[cols + [target]].dropna()
        return subset[cols], subset[target]

    @staticmethod
    def _get_regression_model(model_type: str):
        models = {
            "linear": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=1.0),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        if model_type not in models:
            raise ValueError(f"Unknown regression model: {model_type}. Options: {list(models)}")
        return models[model_type]

    @staticmethod
    def _get_classification_model(model_type: str):
        models = {
            "logistic": LogisticRegression(max_iter=1000, random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "svm": SVC(kernel="rbf", random_state=42),
        }
        if model_type not in models:
            raise ValueError(f"Unknown classification model: {model_type}. Options: {list(models)}")
        return models[model_type]
