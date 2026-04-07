"""
Interactive Dash Dashboard — Learning Analytics Platform.

Multi-tab dashboard with:
  - Executive Overview (KPIs, completion trends, engagement)
  - Learner Explorer (individual profiles, search)
  - Course Analytics (completion, drop-off, time)
  - Engagement & Skills (scoring, skill gaps, distribution)
  - Insights & Predictions (automated insights, ML results)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from pipeline.ingestion import DataIngestor
from pipeline.cleaning import DataCleaner
from pipeline.transformation import DataTransformer
from analytics.learner import LearnerAnalytics
from analytics.course import CourseAnalytics
from analytics.program import ProgramAnalytics
from analytics.video import VideoAnalytics
from analytics.engagement import EngagementScoringEngine
from analytics.skills import SkillIntelligenceEngine
from analytics.predictive import PredictiveAnalytics
from analytics.insights import InsightEngine
from core.config import settings


# ── Load and Process Data ───────────────────────────────────────────

def load_and_process():
    """Run full pipeline and return all analytics objects."""
    ingestor = DataIngestor()
    raw = ingestor.ingest_all(settings.data_dir)

    cleaner = DataCleaner()
    cleaned = {k: cleaner.clean(v) for k, v in raw.items()}

    transformer = DataTransformer()
    transformed = transformer.transform_all(cleaned)

    ca = transformed.get("course_activity", pd.DataFrame())
    va = transformed.get("video_activity", pd.DataFrame())
    pa = transformed.get("program_activity", pd.DataFrame())
    learners = transformed.get("learners", pd.DataFrame())

    # Engagement
    eng_engine = EngagementScoringEngine()
    engagement = eng_engine.compute(ca, va) if not ca.empty else pd.DataFrame()

    # Skills
    skills_engine = SkillIntelligenceEngine(ca) if not ca.empty else None
    skill_scores = skills_engine.compute_skill_scores() if skills_engine else pd.DataFrame()
    skill_gaps = skills_engine.skill_gap_analysis() if skills_engine else pd.DataFrame()
    org_skills = skills_engine.org_skill_distribution() if skills_engine else pd.DataFrame()

    # Insights
    insight_engine = InsightEngine()
    insights = insight_engine.generate_all(ca, engagement, va, pa) if not ca.empty else []

    # Predictive
    predictions = {}
    if not ca.empty:
        try:
            pred_engine = PredictiveAnalytics()
            predictions = pred_engine.get_all_predictions(ca, engagement, va)
        except Exception:
            pass

    # Unified view
    unified = None
    try:
        unified = transformer.get_unified_learner_view()
    except Exception:
        pass

    return {
        "course_activity": ca,
        "video_activity": va,
        "program_activity": pa,
        "learners": learners,
        "engagement": engagement,
        "skill_scores": skill_scores,
        "skill_gaps": skill_gaps,
        "org_skills": org_skills,
        "insights": insights,
        "predictions": predictions,
        "unified": unified,
    }


# ── Initialize ──────────────────────────────────────────────────────

data = load_and_process()
ca = data["course_activity"]
va = data["video_activity"]
pa = data["program_activity"]
engagement = data["engagement"]
skill_gaps = data["skill_gaps"]
org_skills = data["org_skills"]
insights_list = data["insights"]
predictions = data["predictions"]

COLORS = settings.dashboard.color_palette


# ── Dash App ────────────────────────────────────────────────────────

dash_app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="Learning Analytics Platform",
)
server = dash_app.server  # for WSGI


def kpi_card(title, value, subtitle="", color="#1f77b4"):
    return html.Div([
        html.H4(title, style={"color": "#666", "margin": "0", "fontSize": "13px"}),
        html.H2(str(value), style={"color": color, "margin": "5px 0", "fontSize": "28px"}),
        html.P(subtitle, style={"color": "#999", "margin": "0", "fontSize": "11px"}),
    ], style={
        "background": "white", "padding": "20px", "borderRadius": "8px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "textAlign": "center",
        "flex": "1", "minWidth": "180px",
    })


# ── Executive Tab ───────────────────────────────────────────────────

def executive_tab():
    total_learners = ca["email"].nunique() if not ca.empty else 0
    total_courses = ca["course_id"].nunique() if not ca.empty else 0
    completion_rate = round(ca["is_completed"].mean() * 100, 1) if not ca.empty else 0
    total_hours = round(ca["learning_hours"].sum(), 1) if not ca.empty else 0
    avg_engagement = round(engagement["score"].mean(), 1) if not engagement.empty else 0
    at_risk = len(engagement[engagement["category"] == "At-Risk"]) if not engagement.empty else 0

    # Completion by course chart
    course_rates = ca.groupby("course_name")["is_completed"].mean().reset_index()
    course_rates["completion_rate"] = (course_rates["is_completed"] * 100).round(1)
    course_rates = course_rates.sort_values("completion_rate", ascending=True)
    fig_completion = px.bar(
        course_rates, x="completion_rate", y="course_name",
        orientation="h", title="Completion Rate by Course",
        color="completion_rate", color_continuous_scale="RdYlGn",
        labels={"completion_rate": "Completion %", "course_name": ""},
    )
    fig_completion.update_layout(height=450, template="plotly_white", showlegend=False)

    # Engagement distribution
    fig_engagement = px.histogram(
        engagement, x="score", nbins=20, title="Engagement Score Distribution",
        color_discrete_sequence=[COLORS[0]],
        labels={"score": "Engagement Score", "count": "Learners"},
    ) if not engagement.empty else go.Figure()
    fig_engagement.update_layout(height=350, template="plotly_white")

    # Engagement categories pie
    if not engagement.empty:
        cat_counts = engagement["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig_pie = px.pie(
            cat_counts, values="count", names="category",
            title="Engagement Categories",
            color="category",
            color_discrete_map={"High": "#2ca02c", "Medium": "#ff7f0e", "Low": "#d62728", "At-Risk": "#7f0000"},
        )
        fig_pie.update_layout(height=350, template="plotly_white")
    else:
        fig_pie = go.Figure()

    # Progress distribution
    fig_progress = px.histogram(
        ca, x="progress_pct", nbins=20, title="Progress Distribution",
        color_discrete_sequence=[COLORS[1]],
        labels={"progress_pct": "Progress %", "count": "Enrollments"},
    ) if not ca.empty else go.Figure()
    fig_progress.update_layout(height=350, template="plotly_white")

    return html.Div([
        # KPI Cards
        html.Div([
            kpi_card("Total Learners", f"{total_learners:,}", "unique learners", COLORS[0]),
            kpi_card("Courses", total_courses, "unique courses", COLORS[1]),
            kpi_card("Completion Rate", f"{completion_rate}%", "overall", COLORS[2]),
            kpi_card("Learning Hours", f"{total_hours:,.0f}", "total across platform", COLORS[3]),
            kpi_card("Avg Engagement", avg_engagement, "score (0-100)", COLORS[4]),
            kpi_card("At-Risk", at_risk, "learners needing attention", "#d62728"),
        ], style={"display": "flex", "gap": "15px", "flexWrap": "wrap", "marginBottom": "25px"}),

        # Charts row 1
        html.Div([
            html.Div([dcc.Graph(figure=fig_completion)], style={"flex": "1"}),
            html.Div([dcc.Graph(figure=fig_pie)], style={"flex": "0.6"}),
        ], style={"display": "flex", "gap": "15px", "marginBottom": "15px"}),

        # Charts row 2
        html.Div([
            html.Div([dcc.Graph(figure=fig_engagement)], style={"flex": "1"}),
            html.Div([dcc.Graph(figure=fig_progress)], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "15px"}),
    ])


# ── Course Tab ──────────────────────────────────────────────────────

def course_tab():
    if ca.empty:
        return html.Div("No course data available")

    engine = CourseAnalytics(ca)

    # Drop-off analysis
    drop_off = engine.drop_off_analysis()
    if not drop_off.empty:
        fig_dropoff = px.bar(
            drop_off.groupby("progress_bucket", observed=True)["learner_count"].sum().reset_index(),
            x="progress_bucket", y="learner_count",
            title="Where Learners Drop Off (Progress Distribution)",
            color="learner_count", color_continuous_scale="Reds",
            labels={"progress_bucket": "Progress Range", "learner_count": "Learners"},
        )
        fig_dropoff.update_layout(height=400, template="plotly_white")
    else:
        fig_dropoff = go.Figure()

    # Time to completion
    ttc = engine.avg_time_to_completion()
    if not ttc.empty:
        fig_ttc = px.bar(
            ttc.sort_values("avg_days"), x="course_name", y="avg_days",
            title="Average Days to Completion",
            color="avg_days", color_continuous_scale="Viridis",
            labels={"avg_days": "Days", "course_name": "Course"},
        )
        fig_ttc.update_layout(height=400, template="plotly_white", xaxis_tickangle=-45)
    else:
        fig_ttc = go.Figure()

    # Engagement vs completion scatter
    evc = engine.engagement_vs_completion()
    fig_evc = px.scatter(
        evc, x="avg_hours", y="completion_rate", size="avg_progress",
        title="Engagement (Hours) vs Completion Rate",
        labels={"avg_hours": "Avg Learning Hours", "completion_rate": "Completion %"},
        color="completion_rate", color_continuous_scale="RdYlGn",
    )
    fig_evc.update_layout(height=400, template="plotly_white")

    # Course table
    rates = engine.completion_rates()
    table_cols = ["course_name", "enrollments", "completed", "completion_rate", "avg_progress", "avg_hours"]
    available_cols = [c for c in table_cols if c in rates.columns]

    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=fig_dropoff)], style={"flex": "1"}),
            html.Div([dcc.Graph(figure=fig_ttc)], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "15px", "marginBottom": "15px"}),

        dcc.Graph(figure=fig_evc),

        html.H3("Course Performance Table", style={"marginTop": "20px"}),
        dash_table.DataTable(
            data=rates[available_cols].round(1).to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in available_cols],
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
            style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
            page_size=15,
        ),
    ])


# ── Engagement & Skills Tab ────────────────────────────────────────

def skills_tab():
    charts = []

    # Skill gap analysis
    if not skill_gaps.empty:
        fig_gaps = px.bar(
            skill_gaps.sort_values("gap_index", ascending=False).head(15),
            x="gap_index", y="skill_name", orientation="h",
            title="Skill Gap Analysis (Higher = Bigger Gap)",
            color="gap_index", color_continuous_scale="Reds",
            labels={"gap_index": "Gap Index", "skill_name": "Skill"},
        )
        fig_gaps.update_layout(height=450, template="plotly_white")
        charts.append(dcc.Graph(figure=fig_gaps))

    # Org skill distribution
    if not org_skills.empty:
        fig_org = px.bar(
            org_skills.sort_values("total_learners", ascending=True),
            x="total_learners", y="skill_category", orientation="h",
            title="Learners per Skill Category",
            color="avg_score", color_continuous_scale="Viridis",
            labels={"total_learners": "Learners", "skill_category": ""},
        )
        fig_org.update_layout(height=350, template="plotly_white")
        charts.append(dcc.Graph(figure=fig_org))

    # Skill coverage heatmap
    if not skill_gaps.empty:
        fig_coverage = px.scatter(
            skill_gaps, x="avg_score", y="coverage_pct", size="learners_with_skill",
            color="skill_category", text="skill_name",
            title="Skill Score vs Coverage",
            labels={"avg_score": "Avg Skill Score", "coverage_pct": "% Learners With Skill"},
        )
        fig_coverage.update_traces(textposition="top center", textfont_size=9)
        fig_coverage.update_layout(height=450, template="plotly_white")
        charts.append(dcc.Graph(figure=fig_coverage))

    if not charts:
        return html.Div("No skill data available")

    return html.Div(charts)


# ── Insights Tab ────────────────────────────────────────────────────

def insights_tab():
    if not insights_list:
        return html.Div("No insights generated")

    severity_colors = {"critical": "#d62728", "warning": "#ff7f0e", "info": "#1f77b4"}

    cards = []
    for insight in insights_list:
        color = severity_colors.get(insight["severity"], "#666")
        cards.append(html.Div([
            html.Div([
                html.Span(
                    insight["severity"].upper(),
                    style={
                        "background": color, "color": "white", "padding": "2px 8px",
                        "borderRadius": "4px", "fontSize": "11px", "marginRight": "10px",
                    }
                ),
                html.Span(insight["category"], style={"color": "#999", "fontSize": "12px"}),
            ], style={"marginBottom": "8px"}),
            html.H4(insight["title"], style={"margin": "0 0 5px 0", "fontSize": "15px"}),
            html.P(insight["description"], style={"color": "#555", "fontSize": "13px", "margin": 0}),
        ], style={
            "background": "white", "padding": "15px", "borderRadius": "8px",
            "borderLeft": f"4px solid {color}",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.1)", "marginBottom": "10px",
        }))

    # Prediction results
    pred_cards = []
    for task_name, result in predictions.items():
        if isinstance(result, dict) and "accuracy" in result:
            pred_cards.append(html.Div([
                html.H4(f"{task_name.replace('_', ' ').title()} Model", style={"margin": "0 0 10px 0"}),
                html.Div([
                    kpi_card("Accuracy", f"{result['accuracy']:.1%}", "", COLORS[0]),
                    kpi_card("Precision", f"{result['precision']:.1%}", "", COLORS[1]),
                    kpi_card("Recall", f"{result['recall']:.1%}", "", COLORS[2]),
                    kpi_card("F1 Score", f"{result['f1']:.1%}", "", COLORS[3]),
                    kpi_card("AUC-ROC", f"{result.get('auc_roc', 0):.3f}", "", COLORS[4]),
                ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
            ], style={
                "background": "white", "padding": "20px", "borderRadius": "8px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "marginBottom": "15px",
            }))

    return html.Div([
        html.H3(f"Automated Insights ({len(insights_list)} generated)"),
        html.Div(cards),
        html.H3("Predictive Model Performance", style={"marginTop": "30px"}),
        html.Div(pred_cards) if pred_cards else html.P("No predictions computed"),
    ])


# ── Layout ──────────────────────────────────────────────────────────

dash_app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Learning Analytics Platform",
                style={"margin": "0", "color": "white", "fontSize": "24px"}),
        html.P("Coursera Enterprise Analytics Dashboard",
               style={"margin": "0", "color": "rgba(255,255,255,0.7)", "fontSize": "13px"}),
    ], style={
        "background": "linear-gradient(135deg, #1f77b4, #2ca02c)",
        "padding": "20px 30px", "marginBottom": "20px",
    }),

    # Tabs
    dcc.Tabs([
        dcc.Tab(label="Executive Overview", children=[
            html.Div(executive_tab(), style={"padding": "20px"})
        ]),
        dcc.Tab(label="Course Analytics", children=[
            html.Div(course_tab(), style={"padding": "20px"})
        ]),
        dcc.Tab(label="Engagement & Skills", children=[
            html.Div(skills_tab(), style={"padding": "20px"})
        ]),
        dcc.Tab(label="Insights & Predictions", children=[
            html.Div(insights_tab(), style={"padding": "20px"})
        ]),
    ], style={"marginBottom": "20px"}),

], style={"backgroundColor": "#f5f6fa", "minHeight": "100vh", "fontFamily": "Segoe UI, sans-serif"})


def run_dashboard(host="0.0.0.0", port=8050, debug=False):
    dash_app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard(debug=True)
