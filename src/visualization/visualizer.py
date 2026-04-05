"""Visualization Module for the Learning Analytics Platform.

Provides interactive Plotly charts and a Dash-based dashboard.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class Visualizer:
    """Creates interactive visualizations and dashboards."""

    def __init__(self, config: dict):
        self.config = config
        self.theme = config.get("theme", "plotly_white")
        self.colors = config.get("color_palette", px.colors.qualitative.Set2)
        self.dash_config = config.get("dashboard", {})

    def plot_distribution(
        self, df: pd.DataFrame, column: str, title: Optional[str] = None
    ) -> go.Figure:
        """Histogram with KDE overlay for a numeric column."""
        data = df[column].dropna()
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data, nbinsx=30, name="Frequency",
            marker_color=self.colors[0], opacity=0.7,
        ))
        fig.update_layout(
            title=title or f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Frequency",
            template=self.theme,
        )
        return fig

    def plot_correlation_matrix(
        self, corr_matrix: pd.DataFrame, title: Optional[str] = None
    ) -> go.Figure:
        """Heatmap of correlation matrix."""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title=title or "Correlation Matrix",
            template=self.theme,
            width=700, height=600,
        )
        return fig

    def plot_cohort_analysis(
        self,
        df: pd.DataFrame,
        cohort_column: str,
        metric_column: str,
        time_column: str,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Cohort analysis heatmap showing metric by cohort over time."""
        pivot = df.pivot_table(
            values=metric_column,
            index=cohort_column,
            columns=time_column,
            aggfunc="mean",
        )
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale="YlOrRd",
            text=np.round(pivot.values, 2),
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title=title or f"Cohort Analysis: {metric_column}",
            xaxis_title=time_column,
            yaxis_title=cohort_column,
            template=self.theme,
        )
        return fig

    def plot_time_series(
        self,
        df: pd.DataFrame,
        time_column: str,
        value_columns: list[str],
        title: Optional[str] = None,
    ) -> go.Figure:
        """Multi-line time series plot."""
        fig = go.Figure()
        for i, col in enumerate(value_columns):
            if col in df.columns:
                sorted_df = df.sort_values(time_column)
                fig.add_trace(go.Scatter(
                    x=sorted_df[time_column],
                    y=sorted_df[col],
                    mode="lines+markers",
                    name=col,
                    line=dict(color=self.colors[i % len(self.colors)]),
                ))
        fig.update_layout(
            title=title or "Time Series",
            xaxis_title=time_column,
            template=self.theme,
        )
        return fig

    def plot_cluster_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        labels: list,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Scatter plot colored by cluster labels."""
        plot_df = df[[x_col, y_col]].copy()
        plot_df = plot_df.iloc[: len(labels)]
        plot_df["Cluster"] = [str(l) for l in labels[: len(plot_df)]]

        fig = px.scatter(
            plot_df, x=x_col, y=y_col, color="Cluster",
            title=title or f"Cluster Analysis: {x_col} vs {y_col}",
            template=self.theme,
            color_discrete_sequence=self.colors,
        )
        return fig

    def plot_feature_importance(
        self,
        feature_names: list[str],
        importances: list[float],
        title: Optional[str] = None,
    ) -> go.Figure:
        """Horizontal bar chart of feature importances."""
        sorted_idx = np.argsort(importances)
        fig = go.Figure(go.Bar(
            x=[importances[i] for i in sorted_idx],
            y=[feature_names[i] for i in sorted_idx],
            orientation="h",
            marker_color=self.colors[0],
        ))
        fig.update_layout(
            title=title or "Feature Importance",
            xaxis_title="Importance",
            template=self.theme,
        )
        return fig

    def plot_confusion_matrix(
        self,
        cm: list[list[int]],
        class_names: Optional[list[str]] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Annotated heatmap of confusion matrix."""
        cm_array = np.array(cm)
        labels = class_names or [str(i) for i in range(len(cm_array))]

        fig = go.Figure(data=go.Heatmap(
            z=cm_array,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm_array,
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title=title or "Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            template=self.theme,
        )
        return fig

    def create_dashboard(
        self, df: pd.DataFrame, features_df: Optional[pd.DataFrame] = None
    ):
        """Create a Dash app with multiple tabs for exploration.

        Returns:
            A Dash app instance. Call app.run_server() to launch.
        """
        from dash import Dash, dcc, html

        app = Dash(__name__)
        data = features_df if features_df is not None else df
        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()

        # Build figures for dashboard
        figs = []
        for col in numeric_cols[:6]:
            figs.append(self.plot_distribution(data, col))

        if len(numeric_cols) >= 2:
            corr = data[numeric_cols[:10]].corr()
            figs.append(self.plot_correlation_matrix(corr))

        app.layout = html.Div([
            html.H1("Learning Analytics Dashboard",
                     style={"textAlign": "center", "padding": "20px"}),
            html.Hr(),
            html.Div([
                html.H2("Data Overview"),
                html.P(f"Total records: {len(data):,}"),
                html.P(f"Features: {len(data.columns)}"),
                html.P(f"Numeric columns: {len(numeric_cols)}"),
            ], style={"padding": "20px"}),
            html.Div([
                dcc.Graph(figure=fig) for fig in figs
            ]),
        ])

        logger.info("Dashboard created with %d charts", len(figs))
        return app

    def save_figure(
        self, fig: go.Figure, path: str, format: str = "html"
    ) -> str:
        """Save a Plotly figure to file."""
        if format == "html":
            if not path.endswith(".html"):
                path += ".html"
            fig.write_html(path)
        elif format == "png":
            if not path.endswith(".png"):
                path += ".png"
            fig.write_image(path)
        else:
            fig.write_html(path + ".html")

        logger.info("Saved figure to %s", path)
        return path
