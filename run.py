"""
Learning Analytics Platform — Main entry point.

Usage:
    python run.py pipeline     # Run full data pipeline
    python run.py api          # Start FastAPI server
    python run.py dashboard    # Start Dash dashboard
    python run.py all          # Run pipeline + start both servers
    python run.py test         # Run test suite
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def run_pipeline():
    """Execute the full data pipeline and print analytics summary."""
    print("=" * 60)
    print("  LEARNING ANALYTICS PLATFORM - Full Pipeline")
    print("=" * 60)

    from pipeline.ingestion import DataIngestor
    from pipeline.cleaning import DataCleaner
    from pipeline.transformation import DataTransformer
    from analytics.learner import LearnerAnalytics
    from analytics.course import CourseAnalytics
    from analytics.engagement import EngagementScoringEngine
    from analytics.skills import SkillIntelligenceEngine
    from analytics.predictive import PredictiveAnalytics
    from analytics.insights import InsightEngine
    from reports.generator import ReportGenerator
    from core.config import settings

    # Step 1: Ingest
    print("\n[1/7] Ingesting data...")
    ingestor = DataIngestor()
    raw = ingestor.ingest_all(settings.data_dir)
    for entry in ingestor.load_summary:
        print(f"  Loaded {entry['file']}: {entry['rows']} rows")

    # Step 2: Clean
    print("\n[2/7] Cleaning data...")
    cleaner = DataCleaner()
    cleaned = {}
    for k, v in raw.items():
        cleaned[k] = cleaner.clean(v)
        r = cleaner.cleaning_report
        print(f"  {k}: {r['original_rows']} -> {r['cleaned_rows']} rows")

    # Step 3: Transform
    print("\n[3/7] Transforming to unified models...")
    transformer = DataTransformer()
    transformed = transformer.transform_all(cleaned)
    for k, v in transformed.items():
        print(f"  {k}: {len(v)} rows")

    ca = transformed["course_activity"]
    va = transformed.get("video_activity")
    pa = transformed.get("program_activity")
    learners = transformed.get("learners")

    # Step 4: Analytics
    print("\n[4/7] Running analytics...")
    la = LearnerAnalytics(learners, ca)
    comp = la.completion_summary()
    print(f"  Completion rate: {comp['completion_rate']}%")
    print(f"  Total enrollments: {comp['total_enrollments']}")

    course_eng = CourseAnalytics(ca)
    metrics = course_eng.get_all_metrics()
    print(f"  Courses tracked: {metrics['total_courses']}")

    # Step 5: Engagement Scoring
    print("\n[5/7] Computing engagement scores...")
    eng_engine = EngagementScoringEngine()
    engagement = eng_engine.compute(ca, va)
    dist = eng_engine.category_distribution(engagement)
    print(f"  Avg engagement: {dist['avg_score']}")
    print(f"  Distribution: {dist['distribution']}")

    # Step 6: Skills + Predictions
    print("\n[6/7] Running skill analysis & predictions...")
    skills_engine = SkillIntelligenceEngine(ca)
    skill_metrics = skills_engine.get_all_metrics()
    print(f"  Skills tracked: {skill_metrics['total_skills_tracked']}")
    print(f"  Avg skill score: {skill_metrics['avg_skill_score']}")

    pred_engine = PredictiveAnalytics()
    try:
        predictions = pred_engine.get_all_predictions(ca, engagement, va)
        for task, result in predictions.items():
            if isinstance(result, dict) and "accuracy" in result:
                print(f"  {task}: accuracy={result['accuracy']:.3f}, f1={result['f1']:.3f}")
    except Exception as e:
        print(f"  Predictions skipped: {e}")
        predictions = {}

    # Step 7: Insights + Reports
    print("\n[7/7] Generating insights & reports...")
    insight_engine = InsightEngine()
    insights = insight_engine.generate_all(ca, engagement, va, pa)
    print(f"  Generated {len(insights)} insights")

    for i in insights[:5]:
        print(f"    [{i['severity'].upper()}] {i['title']}")

    # Export reports
    gen = ReportGenerator(output_dir=settings.report.output_dir)
    analytics_data = {
        "kpis": {
            "total_learners": ca["email"].nunique(),
            "total_courses": ca["course_id"].nunique(),
            "overall_completion_rate": comp["completion_rate"],
            "avg_engagement_score": dist["avg_score"],
        },
        "insights": insights,
        "engagement": engagement.to_dict(orient="records"),
        "predictions": predictions,
    }

    pdf_path = gen.generate_pdf_report(transformed, analytics_data)
    excel_path = gen.generate_excel_report(transformed, analytics_data)
    print(f"\n  PDF Report: {pdf_path}")
    print(f"  Excel Report: {excel_path}")

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


def run_api():
    """Start FastAPI server."""
    import uvicorn
    print("Starting FastAPI server at http://localhost:8000")
    print("API docs at http://localhost:8000/docs")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


def run_dashboard():
    """Start Dash dashboard."""
    print("Starting Dash dashboard at http://localhost:8050")
    from dashboard.app import run_dashboard as _run
    _run(debug=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()

    if command == "pipeline":
        run_pipeline()
    elif command == "api":
        run_api()
    elif command == "dashboard":
        run_dashboard()
    elif command == "all":
        run_pipeline()
        print("\nStarting servers...")
        run_api()
    elif command == "test":
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], cwd=str(ROOT))
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
