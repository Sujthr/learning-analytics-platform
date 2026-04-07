# Learning Analytics Platform - Complete Runbook

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Data Setup](#3-data-setup)
4. [Running the Platform](#4-running-the-platform)
5. [API Reference](#5-api-reference)
6. [Dashboard Guide](#6-dashboard-guide)
7. [Using Your Own Data](#7-using-your-own-data)
8. [Architecture Overview](#8-architecture-overview)
9. [Analytics Modules](#9-analytics-modules)
10. [Reports & Exports](#10-reports--exports)
11. [Testing](#11-testing)
12. [Troubleshooting](#12-troubleshooting)
13. [Data Privacy](#13-data-privacy)

---

## 1. Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| pip | Latest |
| Git | 2.30+ |
| OS | Windows 10/11, macOS, Linux |

Optional (for production deployment):
- PostgreSQL 14+
- Redis 7+

---

## 2. Installation

```bash
# Clone the repository
git clone https://github.com/Sujthr/learning-analytics-platform.git
cd learning-analytics-platform

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies installed

| Package | Purpose |
|---------|---------|
| pandas, numpy, scipy | Data processing |
| scikit-learn | ML models (predictive analytics) |
| matplotlib, plotly | Visualization |
| dash | Interactive dashboard |
| fastapi, uvicorn | REST API server |
| sqlalchemy | Database ORM |
| openpyxl | Excel export |
| fpdf2 | PDF report generation |
| pydantic | API schemas |
| pytest, httpx | Testing |

---

## 3. Data Setup

### Option A: Generate synthetic sample data

```bash
python data/generate_coursera_data.py
```

This creates 5 files in `data/coursera/`:

| File | Rows | Description |
|------|------|-------------|
| `course_activity.csv` | ~2,500 | Per-learner course enrollment and progress |
| `program_activity.csv` | ~350 | Per-learner program-level progress |
| `specialization_activity.csv` | ~300 | Per-learner specialization progress |
| `video_clip_activity.csv` | ~12,500 | Per-learner video watch events |
| `learners_reference.csv` | 500 | Learner profiles |

### Option B: Use your own Coursera data

Place your CSV exports in `data/coursera/` with these filenames:
- `course_activity.csv`
- `program_activity.csv`
- `specialization_activity.csv`
- `video_clip_activity.csv`

Required columns per dataset - see [Section 7](#7-using-your-own-data) for details.

---

## 4. Running the Platform

### 4.1 Full Pipeline (recommended first run)

```bash
python run.py pipeline
```

This runs all 7 steps sequentially:
1. **Ingest** - Load and validate all 4 CSV datasets
2. **Clean** - Handle missing values, duplicates, normalize timestamps
3. **Transform** - Build unified data models (learner, course, program, video tables)
4. **Analyze** - Run learner, course, program analytics
5. **Engagement Scoring** - Compute 0-100 engagement scores per learner
6. **Skills & Predictions** - Skill gap analysis + ML dropout/completion models
7. **Insights & Reports** - Generate automated insights + PDF/Excel reports

Output:
- PDF report saved to `output/reports/analytics_report_<timestamp>.pdf`
- Excel report saved to `output/reports/analytics_report_<timestamp>.xlsx`
- Console prints key metrics and top insights

### 4.2 FastAPI Server

```bash
python run.py api
```

- Server starts at: **http://localhost:8000**
- Swagger docs at: **http://localhost:8000/docs**
- ReDoc at: **http://localhost:8000/redoc**

### 4.3 Interactive Dashboard

```bash
python run.py dashboard
```

- Dashboard opens at: **http://localhost:8050**
- Auto-loads and processes data on startup
- 4 tabs: Executive Overview, Course Analytics, Engagement & Skills, Insights & Predictions

### 4.4 Run Tests

```bash
python run.py test
# or directly:
python -m pytest tests/ -v
```

---

## 5. API Reference

### Pipeline Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/ingest/load-sample` | Load sample data from `data/coursera/` |
| POST | `/api/v1/ingest/upload` | Upload a CSV file (multipart form: `file` + `source_type`) |
| POST | `/api/v1/pipeline/clean` | Clean loaded raw data |
| POST | `/api/v1/pipeline/transform` | Transform to unified models |
| POST | `/api/v1/pipeline/run-all` | Run complete pipeline (load + clean + transform + analytics) |

### Analytics Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/analytics/learner` | Learner-level analytics summary |
| GET | `/api/v1/analytics/learner/{email}` | Individual learner detail + skills |
| GET | `/api/v1/analytics/course` | Course completion rates, drop-off, time-to-completion |
| GET | `/api/v1/analytics/program` | Program success rates |
| GET | `/api/v1/analytics/video` | Video watch frequency, completion, rewatch patterns |
| POST | `/api/v1/analytics/engagement/compute` | Compute engagement scores |
| GET | `/api/v1/analytics/engagement` | Get engagement scores (filter by `?category=High`) |
| GET | `/api/v1/analytics/skills/gaps` | Org-wide skill gap analysis |
| GET | `/api/v1/analytics/skills/learner/{email}` | Individual skill profile + timeline |
| POST | `/api/v1/analytics/predict` | Run ML prediction models |
| POST | `/api/v1/analytics/run-all` | Run all analytics at once |

### Insights & Reports

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/insights/generate` | Generate automated insights |
| GET | `/api/v1/insights` | Get insights (filter by `?severity=critical`) |
| GET | `/api/v1/dashboard/kpis` | Dashboard KPI summary |
| POST | `/api/v1/reports/generate?format=pdf` | Download PDF report |
| POST | `/api/v1/reports/generate?format=excel` | Download Excel report |
| POST | `/api/v1/reports/export/{table_name}?format=csv` | Export a specific table |

### Typical API workflow

```bash
# 1. Run full pipeline (loads data, cleans, transforms, runs analytics)
curl -X POST http://localhost:8000/api/v1/pipeline/run-all

# 2. Get KPIs
curl http://localhost:8000/api/v1/dashboard/kpis

# 3. Get insights
curl http://localhost:8000/api/v1/insights

# 4. Get specific learner detail
curl http://localhost:8000/api/v1/analytics/learner/sanjay.patel1@company.com

# 5. Download PDF report
curl -X POST http://localhost:8000/api/v1/reports/generate?format=pdf -o report.pdf
```

---

## 6. Dashboard Guide

### Tab 1: Executive Overview
- **KPI Cards**: Total learners, courses, completion rate, learning hours, avg engagement, at-risk count
- **Completion by Course**: Horizontal bar chart ranked by completion %
- **Engagement Categories**: Pie chart (High/Medium/Low/At-Risk)
- **Score Distribution**: Histogram of engagement scores
- **Progress Distribution**: Histogram of progress %

### Tab 2: Course Analytics
- **Drop-off Analysis**: Where learners stall (0-10%, 10-25%, etc.)
- **Time to Completion**: Average days per course
- **Engagement vs Completion**: Scatter plot showing hours vs completion rate
- **Course Table**: Sortable/filterable table with all metrics

### Tab 3: Engagement & Skills
- **Skill Gap Analysis**: Top skills with largest gaps
- **Learners per Category**: Bar chart by skill category
- **Skill Score vs Coverage**: Scatter showing which skills are widespread vs rare

### Tab 4: Insights & Predictions
- **Automated Insights**: Color-coded cards (CRITICAL/WARNING/INFO)
- **ML Model Performance**: Accuracy, precision, recall, F1, AUC-ROC for each model

---

## 7. Using Your Own Data

### Required CSV schemas

#### course_activity.csv
```
Email,External ID,Course Id,Course Name,Progress (%)
```
Optional columns: `Name`, `Business Unit`, `Role`, `Location`, `Course Slug`, `Institution`, `Enrollment Timestamp`, `Completion Timestamp`, `Grade (%)`, `Learning Hours`, `Completed`, `Last Activity Timestamp`

#### program_activity.csv
```
Email,External ID,Program Id,Program Name,Progress (%)
```
Optional: `Name`, `Business Unit`, `Program Slug`, `Total Courses in Program`, `Courses Completed`, `Learning Hours`, `Completed`, `Enrollment Timestamp`, `Last Activity Timestamp`

#### specialization_activity.csv
```
Email,External ID,Specialization Id,Specialization Name,Progress (%)
```
Optional: `Name`, `Business Unit`, `Specialization Slug`, `Total Courses`, `Courses Completed`, `Learning Hours`, `Completed`, `Enrollment Timestamp`, `Last Activity Timestamp`

#### video_clip_activity.csv
```
Email,External ID,Course Id,Video Id
```
Optional: `Course Name`, `Video Name`, `Watch Duration (seconds)`, `Total Duration (seconds)`, `Completion (%)`, `Watch Count`, `Last Watch Timestamp`

### Steps to use your data

1. Place your CSVs in `data/coursera/`
2. If your columns differ, update `pipeline/ingestion.py` SCHEMAS dict
3. Run: `python run.py pipeline`
4. If you need custom skill mappings, edit `analytics/skills.py` COURSE_SKILL_MAP

### Uploading via API

```bash
curl -X POST http://localhost:8000/api/v1/ingest/upload \
  -F "file=@/path/to/your/course_activity.csv" \
  -F "source_type=course"
```

---

## 8. Architecture Overview

```
learning_analytics_platform/
|
|-- core/                          # Infrastructure
|   |-- config.py                  # Dataclass-based settings (env overrides)
|   |-- database.py                # SQLAlchemy engine, session, init_db()
|   |-- models.py                  # 10 ORM models (Learner, Course, etc.)
|
|-- pipeline/                      # Data Pipeline
|   |-- ingestion.py               # DataIngestor - CSV/Excel load + schema validation
|   |-- cleaning.py                # DataCleaner - missing values, duplicates, timestamps
|   |-- transformation.py          # DataTransformer - unified models, learner view
|
|-- analytics/                     # Analytics Engines
|   |-- learner.py                 # LearnerAnalytics - active/inactive, velocity, completion
|   |-- course.py                  # CourseAnalytics - rates, drop-off, time-to-completion
|   |-- program.py                 # ProgramAnalytics - success rates, distribution
|   |-- video.py                   # VideoAnalytics - frequency, completion, rewatch
|   |-- engagement.py              # EngagementScoringEngine - 0-100 composite scores
|   |-- skills.py                  # SkillIntelligenceEngine - taxonomy, gaps, profiles
|   |-- predictive.py              # PredictiveAnalytics - dropout, completion, high-performer ML
|   |-- insights.py                # InsightEngine - automated insight generation
|
|-- api/                           # FastAPI Backend
|   |-- main.py                    # App with 20+ endpoints, in-memory data store
|   |-- schemas.py                 # Pydantic request/response models
|
|-- dashboard/                     # Interactive Dashboard
|   |-- app.py                     # 4-tab Dash app with Plotly charts
|
|-- reports/                       # Report Generation
|   |-- generator.py               # PDF (matplotlib) and Excel (openpyxl) export
|
|-- tests/                         # Test Suite
|   |-- test_pipeline.py           # 16 tests - ingestion, cleaning, transformation
|   |-- test_analytics.py          # 21 tests - all analytics engines
|   |-- test_api.py                # 11 tests - API endpoints
|
|-- data/
|   |-- coursera/                  # Your data goes here (git-ignored)
|   |-- generate_coursera_data.py  # Synthetic data generator
|
|-- run.py                         # CLI entry point
|-- requirements.txt               # Python dependencies
```

### Data Flow

```
CSV Files --> Ingest --> Clean --> Transform --> Analytics --> Insights
                                     |              |            |
                                     v              v            v
                                  Dashboard      API         Reports
                                 (port 8050)  (port 8000)   (PDF/Excel)
```

---

## 9. Analytics Modules

### 9.1 Learner Analytics
| Metric | Method | Description |
|--------|--------|-------------|
| Active vs Inactive | `active_vs_inactive(days=90)` | Learners with activity in last N days |
| Learning Velocity | `learning_velocity()` | Hours per week per learner |
| Completion Summary | `completion_summary()` | Overall completion rate and stats |
| Completion by Dimension | `completion_by_dimension("business_unit")` | Break down by BU/Role/Location |
| Top Learners | `top_learners(n=20)` | Ranked by composite score |

### 9.2 Course Analytics
| Metric | Method | Description |
|--------|--------|-------------|
| Completion Rates | `completion_rates()` | Per-course completion % |
| Drop-off Analysis | `drop_off_analysis()` | Where learners stall |
| Time to Completion | `avg_time_to_completion()` | Days from enrollment to completion |
| Engagement vs Completion | `engagement_vs_completion()` | Hours vs outcome correlation |

### 9.3 Engagement Scoring (0-100)

Components (configurable weights):
| Component | Weight | Source |
|-----------|--------|--------|
| Progress % | 0.25 | Course activity avg progress |
| Learning Hours | 0.25 | Normalized total hours |
| Video Interactions | 0.25 | Completion % + watch count |
| Recency | 0.25 | Days since last activity |

Categories:
| Score Range | Category |
|-------------|----------|
| 75-100 | High |
| 50-74 | Medium |
| 25-49 | Low |
| 0-24 | At-Risk |

### 9.4 Skill Intelligence

- **Taxonomy**: 7 categories, 38 skills mapped from 15 courses
- **Skill Score**: 0-100 per learner per skill (70% progress + 30% completion bonus)
- **Gap Analysis**: Identifies organization-wide skill deficiencies
- **Progression Timeline**: Shows skill acquisition over time

### 9.5 Predictive Models

| Model | Target | Algorithm |
|-------|--------|-----------|
| Dropout | avg_progress < 25% and 0 completions | Random Forest |
| Completion | Will complete at least 1 course | Random Forest |
| High Performer | Top 25th percentile progress | Random Forest |

Output includes: accuracy, precision, recall, F1, AUC-ROC, feature importances, top flagged learners.

### 9.6 Insight Engine

Auto-generates insights like:
- "Low overall completion rate: 13.0%"
- "Course X has only Y% completion"
- "N learners (Z%) are At-Risk"
- "High engagement learners are Nx more likely to complete"
- "Business Unit X has Y% lower completion than average"
- "Average video completion is only Z%"

---

## 10. Reports & Exports

### PDF Report includes:
- Title page with KPI summary
- Completion rate by course (bar chart)
- Progress and learning hours distributions
- Engagement score distribution and category pie chart
- Key insights (up to 12)

### Excel Report includes:
- Sheet per data table (course_activity, video_activity, etc.)
- KPIs sheet
- Engagement Scores sheet
- Insights sheet
- Prediction Model Results sheet

### Export specific tables
```bash
# Via API
curl -X POST http://localhost:8000/api/v1/reports/export/course_activity?format=csv -o courses.csv
curl -X POST http://localhost:8000/api/v1/reports/export/video_activity?format=excel -o videos.xlsx
```

---

## 11. Testing

```bash
# Run all 48 tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_pipeline.py -v
python -m pytest tests/test_analytics.py -v
python -m pytest tests/test_api.py -v

# Run with coverage (install pytest-cov first)
python -m pytest tests/ --cov=pipeline --cov=analytics --cov=api -v
```

### Test breakdown
| File | Tests | Covers |
|------|-------|--------|
| test_pipeline.py | 16 | Ingestion, cleaning, transformation, schema validation, edge cases |
| test_analytics.py | 21 | All 8 analytics engines end-to-end |
| test_api.py | 11 | API health, pipeline, analytics, edge cases |

---

## 12. Troubleshooting

### "No data loaded" error
Run the pipeline first:
```bash
python run.py pipeline
# or via API:
curl -X POST http://localhost:8000/api/v1/pipeline/run-all
```

### "No module named X" error
Install dependencies:
```bash
pip install -r requirements.txt
```

### "File not found" for data
Generate sample data first:
```bash
python data/generate_coursera_data.py
```

### Dashboard shows empty charts
Ensure data files exist in `data/coursera/`. The dashboard loads data on startup.

### Port already in use
Change ports in `core/config.py`:
```python
class DashboardConfig:
    port: int = 8050  # change this

class APIConfig:
    port: int = 8000  # change this
```

### Database errors
Delete and regenerate:
```bash
rm output/analytics.db
python run.py pipeline
```

---

## 13. Data Privacy

**IMPORTANT: CSV data files are git-ignored and must NEVER be committed.**

- `.gitignore` blocks: `data/**/*.csv`, `data/**/*.xlsx`, `data/**/*.xls`, `*.db`
- All CSV files in `data/coursera/` and `data/sample/` exist locally only
- The data generator creates synthetic data — safe for demos
- When using real Coursera data, ensure it stays in `data/coursera/` (local only)
- Reports in `output/` are also git-ignored
- The API stores data in-memory (not persisted between restarts)

To verify no data is tracked:
```bash
git ls-files "*.csv"    # should return nothing
git ls-files "*.xlsx"   # should return nothing
```
