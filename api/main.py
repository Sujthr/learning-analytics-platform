"""FastAPI application — Learning Analytics Platform API."""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import settings
from core.database import init_db
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── In-memory data store (replaces Redis for simplicity) ────────────

class DataStore:
    """Session-scoped data store for loaded datasets."""
    def __init__(self):
        self.raw: dict = {}
        self.cleaned: dict = {}
        self.transformed: dict = {}
        self.engagement_scores = None
        self.insights = None
        self.predictions = None
        self.unified_learner_view = None
        self.is_loaded = False

store = DataStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialized")
    yield

app = FastAPI(
    title="Learning Analytics Platform API",
    description="Production-grade analytics for Coursera learning data",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health & Status ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/status")
async def status():
    return {
        "data_loaded": store.is_loaded,
        "datasets": list(store.transformed.keys()) if store.transformed else [],
        "has_engagement_scores": store.engagement_scores is not None,
        "has_insights": store.insights is not None,
        "has_predictions": store.predictions is not None,
    }


# ── Data Ingestion ──────────────────────────────────────────────────

@app.post("/api/v1/ingest/upload")
async def upload_file(
    file: UploadFile = File(...),
    source_type: str = Form(...),
):
    """Upload a CSV/Excel file for processing."""
    import tempfile, shutil
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        ingestor = DataIngestor()
        df = ingestor.ingest(tmp_path, source_type)
        store.raw[source_type] = df
        return {
            "source_type": source_type,
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "status": "uploaded",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/api/v1/ingest/load-sample")
async def load_sample_data():
    """Load sample Coursera datasets from data/coursera/."""
    data_dir = settings.data_dir
    ingestor = DataIngestor()
    try:
        datasets = ingestor.ingest_all(data_dir)
        store.raw = datasets
        return {
            "status": "loaded",
            "datasets": {k: len(v) for k, v in datasets.items()},
            "summary": ingestor.load_summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Pipeline ────────────────────────────────────────────────────────

@app.post("/api/v1/pipeline/clean")
async def clean_data():
    """Clean all loaded raw datasets."""
    if not store.raw:
        raise HTTPException(status_code=400, detail="No data loaded. Upload or load sample first.")

    cleaner = DataCleaner()
    store.cleaned = {}
    reports = {}
    for source_type, df in store.raw.items():
        store.cleaned[source_type] = cleaner.clean(df)
        reports[source_type] = cleaner.cleaning_report

    return {"status": "cleaned", "reports": reports}


@app.post("/api/v1/pipeline/transform")
async def transform_data():
    """Transform cleaned data into unified models."""
    data = store.cleaned if store.cleaned else store.raw
    if not data:
        raise HTTPException(status_code=400, detail="No data available to transform.")

    transformer = DataTransformer()
    store.transformed = transformer.transform_all(data)

    try:
        store.unified_learner_view = transformer.get_unified_learner_view()
    except ValueError:
        store.unified_learner_view = None

    store.is_loaded = True
    return {
        "status": "transformed",
        "tables": {k: len(v) for k, v in store.transformed.items()},
    }


@app.post("/api/v1/pipeline/run-all")
async def run_full_pipeline():
    """Run complete pipeline: load sample -> clean -> transform -> analytics."""
    # Load
    load_result = await load_sample_data()
    # Clean
    clean_result = await clean_data()
    # Transform
    transform_result = await transform_data()
    # Analytics
    analytics_result = await run_all_analytics()

    return {
        "status": "complete",
        "load": load_result,
        "clean": clean_result,
        "transform": transform_result,
        "analytics": analytics_result,
    }


# ── Analytics ───────────────────────────────────────────────────────

def _require_data():
    if not store.transformed:
        raise HTTPException(status_code=400, detail="Run the pipeline first (POST /api/v1/pipeline/run-all)")


@app.get("/api/v1/analytics/learner")
async def learner_analytics():
    _require_data()
    ca = store.transformed.get("course_activity")
    if ca is None:
        raise HTTPException(status_code=404, detail="No course activity data")
    learners = store.transformed.get("learners", ca)
    engine = LearnerAnalytics(learners, ca)
    return engine.get_all_metrics()


@app.get("/api/v1/analytics/learner/{email}")
async def learner_detail(email: str):
    _require_data()
    ca = store.transformed["course_activity"]
    record = ca[ca["email"] == email]
    if record.empty:
        raise HTTPException(status_code=404, detail=f"Learner '{email}' not found")

    detail = {
        "email": email,
        "courses": record[["course_id", "course_name", "progress_pct", "is_completed", "learning_hours"]].to_dict(orient="records"),
        "total_hours": round(record["learning_hours"].sum(), 1),
        "avg_progress": round(record["progress_pct"].mean(), 1),
        "courses_completed": int(record["is_completed"].sum()),
    }

    # Add engagement score
    if store.engagement_scores is not None:
        eng = store.engagement_scores[store.engagement_scores["email"] == email]
        if not eng.empty:
            detail["engagement"] = eng.iloc[0].to_dict()

    # Add skill profile
    skills_engine = SkillIntelligenceEngine(ca)
    profile = skills_engine.learner_skill_profile(email)
    if not profile.empty:
        detail["skills"] = profile.to_dict(orient="records")

    return detail


@app.get("/api/v1/analytics/course")
async def course_analytics():
    _require_data()
    ca = store.transformed["course_activity"]
    engine = CourseAnalytics(ca)
    return engine.get_all_metrics()


@app.get("/api/v1/analytics/program")
async def program_analytics():
    _require_data()
    pa = store.transformed.get("program_activity")
    if pa is None or pa.empty:
        return {"message": "No program data available", "metrics": {}}
    engine = ProgramAnalytics(pa)
    return engine.get_all_metrics()


@app.get("/api/v1/analytics/video")
async def video_analytics():
    _require_data()
    va = store.transformed.get("video_activity")
    if va is None or va.empty:
        return {"message": "No video data available", "metrics": {}}
    engine = VideoAnalytics(va)
    return engine.get_all_metrics()


# ── Engagement Scoring ──────────────────────────────────────────────

@app.post("/api/v1/analytics/engagement/compute")
async def compute_engagement():
    _require_data()
    ca = store.transformed["course_activity"]
    va = store.transformed.get("video_activity")

    engine = EngagementScoringEngine()
    store.engagement_scores = engine.compute(ca, va)

    dist = engine.category_distribution(store.engagement_scores)
    return {
        "status": "computed",
        "distribution": dist,
        "sample": store.engagement_scores.head(20).to_dict(orient="records"),
    }


@app.get("/api/v1/analytics/engagement")
async def get_engagement_scores(
    category: str = Query(None, description="Filter by category: High, Medium, Low, At-Risk"),
    limit: int = Query(50, ge=1, le=500),
):
    if store.engagement_scores is None:
        raise HTTPException(status_code=400, detail="Compute engagement scores first")

    result = store.engagement_scores.copy()
    if category:
        result = result[result["category"] == category]

    return {
        "total": len(result),
        "scores": result.head(limit).to_dict(orient="records"),
    }


# ── Skills ──────────────────────────────────────────────────────────

@app.get("/api/v1/analytics/skills/gaps")
async def skill_gaps():
    _require_data()
    engine = SkillIntelligenceEngine(store.transformed["course_activity"])
    gaps = engine.skill_gap_analysis()
    return {
        "gaps": gaps.to_dict(orient="records") if not gaps.empty else [],
        "org_distribution": engine.org_skill_distribution().to_dict(orient="records"),
    }


@app.get("/api/v1/analytics/skills/learner/{email}")
async def learner_skills(email: str):
    _require_data()
    engine = SkillIntelligenceEngine(store.transformed["course_activity"])
    profile = engine.learner_skill_profile(email)
    timeline = engine.skill_progression_timeline(email)
    return {
        "profile": profile.to_dict(orient="records") if not profile.empty else [],
        "timeline": timeline.to_dict(orient="records") if not timeline.empty else [],
    }


# ── Predictive Analytics ───────────────────────────────────────────

@app.post("/api/v1/analytics/predict")
async def run_predictions():
    _require_data()
    ca = store.transformed["course_activity"]
    va = store.transformed.get("video_activity")

    engine = PredictiveAnalytics()
    store.predictions = engine.get_all_predictions(ca, store.engagement_scores, va)
    return store.predictions


# ── Insights ────────────────────────────────────────────────────────

@app.post("/api/v1/insights/generate")
async def generate_insights():
    _require_data()
    ca = store.transformed["course_activity"]
    va = store.transformed.get("video_activity")
    pa = store.transformed.get("program_activity")

    engine = InsightEngine()
    store.insights = engine.generate_all(ca, store.engagement_scores, va, pa)
    return {"total_insights": len(store.insights), "insights": store.insights}


@app.get("/api/v1/insights")
async def get_insights(severity: str = Query(None)):
    if store.insights is None:
        raise HTTPException(status_code=400, detail="Generate insights first")
    results = store.insights
    if severity:
        results = [i for i in results if i["severity"] == severity]
    return {"total": len(results), "insights": results}


# ── Dashboard KPIs ──────────────────────────────────────────────────

@app.get("/api/v1/dashboard/kpis")
async def dashboard_kpis():
    _require_data()
    ca = store.transformed["course_activity"]

    total_learners = ca["email"].nunique()
    total_courses = ca["course_id"].nunique()
    completion_rate = round(ca["is_completed"].mean() * 100, 1) if "is_completed" in ca.columns else 0
    total_hours = round(ca["learning_hours"].sum(), 1)

    active = total_learners  # default
    at_risk = 0
    avg_eng = 0
    if store.engagement_scores is not None:
        at_risk = len(store.engagement_scores[store.engagement_scores["category"] == "At-Risk"])
        avg_eng = round(store.engagement_scores["score"].mean(), 1)

    return {
        "total_learners": total_learners,
        "active_learners": active,
        "total_courses": total_courses,
        "overall_completion_rate": completion_rate,
        "avg_engagement_score": avg_eng,
        "at_risk_count": at_risk,
        "total_learning_hours": total_hours,
        "avg_hours_per_learner": round(total_hours / total_learners, 1) if total_learners else 0,
    }


# ── Reports ─────────────────────────────────────────────────────────

@app.post("/api/v1/reports/generate")
async def generate_report(format: str = Query("pdf", pattern="^(pdf|excel)$")):
    _require_data()
    from reports.generator import ReportGenerator

    gen = ReportGenerator(output_dir=settings.report.output_dir)

    analytics_data = {
        "kpis": (await dashboard_kpis()),
        "insights": store.insights or [],
        "engagement": store.engagement_scores.to_dict(orient="records") if store.engagement_scores is not None else [],
        "predictions": store.predictions or {},
    }

    if format == "pdf":
        path = gen.generate_pdf_report(store.transformed, analytics_data)
    else:
        path = gen.generate_excel_report(store.transformed, analytics_data)

    return FileResponse(path, filename=Path(path).name)


@app.post("/api/v1/reports/export/{table_name}")
async def export_table(table_name: str, format: str = Query("csv", pattern="^(csv|excel)$")):
    _require_data()
    df = store.transformed.get(table_name)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

    from reports.generator import ReportGenerator
    gen = ReportGenerator(output_dir=settings.report.export_dir)
    path = gen.export_dataframe(df, table_name, format)
    return FileResponse(path, filename=Path(path).name)


# ── Run all analytics at once ───────────────────────────────────────

@app.post("/api/v1/analytics/run-all")
async def run_all_analytics():
    _require_data()
    results = {}

    results["learner"] = await learner_analytics()
    results["course"] = await course_analytics()
    results["program"] = await program_analytics()
    results["video"] = await video_analytics()

    eng_result = await compute_engagement()
    results["engagement"] = eng_result

    insight_result = await generate_insights()
    results["insights"] = insight_result

    try:
        pred_result = await run_predictions()
        results["predictions"] = pred_result
    except Exception as e:
        results["predictions"] = {"error": str(e)}

    return results
