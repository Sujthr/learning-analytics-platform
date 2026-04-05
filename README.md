# Learning Analytics Platform

A modular, config-driven analytics platform designed for PhD research on learner engagement and performance across multiple educational data sources (Coursera, LMS, Academic records).

## Architecture

```
learning_analytics_platform/
├── config/
│   └── pipeline_config.yaml      # All pipeline settings
├── src/
│   ├── ingestion/ingestor.py     # Module 1: Data Ingestion
│   ├── cleaning/cleaner.py       # Module 2: Data Cleaning
│   ├── integration/integrator.py # Module 3: Data Integration
│   ├── features/engineer.py      # Module 4: Feature Engineering
│   ├── analytics/analyzer.py     # Module 5: Analytics Engine
│   ├── visualization/visualizer.py # Module 6: Visualization
│   └── reporting/reporter.py     # Module 7: Report Generator
├── data/sample/                  # Sample datasets & generator
├── demo_pipeline.py              # End-to-end demo
└── requirements.txt
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample data
python data/sample/generate_sample_data.py

# 3. Run the full demo pipeline
python demo_pipeline.py
```

## Modules

### 1. Data Ingestion (`src/ingestion/ingestor.py`)
- **Class:** `DataIngestor`
- Reads CSV and Excel files with schema validation
- Handles large datasets via chunked reading
- Validates against expected schemas per source type

```python
from src.ingestion.ingestor import DataIngestor
ingestor = DataIngestor(config["ingestion"])
df = ingestor.ingest("data/coursera.csv", source_type="coursera")
```

### 2. Data Cleaning (`src/cleaning/cleaner.py`)
- **Class:** `DataCleaner`
- Missing value handling (mean, median, mode, drop, zero, unknown)
- Duplicate removal with configurable strategy
- Automatic timestamp normalization
- Intelligent data type inference

```python
from src.cleaning.cleaner import DataCleaner
cleaner = DataCleaner(config["cleaning"])
cleaned_df = cleaner.clean(raw_df)
report = cleaner.get_cleaning_report(raw_df, cleaned_df)
```

### 3. Data Integration (`src/integration/integrator.py`)
- **Class:** `DataIntegrator`
- Joins datasets across sources using mapping tables
- Conflict resolution (latest timestamp, source priority)
- Referential integrity validation

```python
from src.integration.integrator import DataIntegrator
integrator = DataIntegrator(config["integration"])
integrator.load_mapping_table("data/student_mapping.csv")
merged_df = integrator.integrate({"coursera": df1, "lms": df2, "academic": df3})
```

### 4. Feature Engineering (`src/features/engineer.py`)
- **Class:** `FeatureEngineer`
- Computes: engagement score, session frequency, video completion %, assessment improvement rate
- Supports custom feature definitions via `add_custom_feature(name, func)`
- All features computed per-student with configurable parameters

```python
from src.features.engineer import FeatureEngineer
engineer = FeatureEngineer(config["features"])
features_df = engineer.generate_features(integrated_df)
```

### 5. Analytics Engine (`src/analytics/analyzer.py`)
- **Class:** `AnalyticsEngine`
- **EDA:** Distributions, correlation matrices
- **Hypothesis Testing:** Independent t-test, one-way ANOVA
- **ML Regression:** Linear, Ridge, Lasso, Random Forest, Gradient Boosting
- **ML Classification:** Logistic Regression, Random Forest, Gradient Boosting, SVM
- **Clustering:** K-Means, DBSCAN, Hierarchical

```python
from src.analytics.analyzer import AnalyticsEngine
engine = AnalyticsEngine(config["analytics"])
reg = engine.regression(df, target="final_score", features=["engagement_score", "session_freq"])
cls = engine.classification(df, target="dropout", model_type="random_forest")
clusters = engine.clustering(df, features=["engagement_score", "gpa"], n_clusters=4)
```

### 6. Visualization (`src/visualization/visualizer.py`)
- **Class:** `Visualizer`
- Interactive Plotly charts (distributions, correlations, clusters, time-series)
- Dash-based dashboard with multiple tabs
- Export to HTML and PNG

```python
from src.visualization.visualizer import Visualizer
viz = Visualizer(config["visualization"])
fig = viz.plot_distribution(df, "engagement_score")
app = viz.create_dashboard(df)  # Launch with app.run_server()
```

### 7. Report Generator (`src/reporting/reporter.py`)
- **Class:** `ReportGenerator`
- PDF reports with formatted tables and sections
- CSV export for data and statistics
- Research-ready tables with significance stars

```python
from src.reporting.reporter import ReportGenerator
reporter = ReportGenerator(config["reporting"])
reporter.export_pdf(analytics_results, "report.pdf")
reporter.export_csv(df, "results.csv")
```

## Configuration

All pipeline behavior is controlled via `config/pipeline_config.yaml`. Key sections:

| Section | Controls |
|---------|----------|
| `ingestion` | File formats, schemas, chunk sizes |
| `cleaning` | Missing value strategies, duplicate handling |
| `integration` | Join keys, conflict resolution, source priority |
| `features` | Engagement weights, computation windows |
| `analytics` | ML model types, test parameters, significance levels |
| `visualization` | Themes, color palettes, dashboard settings |
| `reporting` | Output formats, decimal places |

## Sample Data

Run `python data/sample/generate_sample_data.py` to generate:
- `coursera_activity.csv` - ~15K Coursera activity logs
- `lms_sessions.csv` - ~15K LMS session records
- `academic_records.csv` - ~2.5K academic grades
- `student_mapping.csv` - Cross-system ID mapping (200 students)

The sample data includes realistic patterns: varied engagement levels, a 12% dropout rate, intentional missing values, and duplicate records for testing the cleaning pipeline.

## Extending the Platform

**Custom features:**
```python
engineer.add_custom_feature("study_intensity", lambda df: df["duration_minutes"] / df["session_count"])
```

**Custom pipeline steps:** Each module is independent. Import and use any module standalone or compose custom pipelines by chaining modules in any order.

## Requirements

- Python 3.10+
- See `requirements.txt` for package dependencies
