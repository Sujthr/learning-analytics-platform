"""
Learning Analytics Platform - Demo Pipeline
=============================================
End-to-end demonstration of the platform capabilities:
1. Data Ingestion -> 2. Cleaning -> 3. Integration -> 4. Feature Engineering
5. Analytics -> 6. Visualization -> 7. Reporting
"""

import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("demo_pipeline")

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config" / "pipeline_config.yaml"
DATA_DIR = BASE_DIR / "data" / "sample"
OUTPUT_DIR = BASE_DIR / "output"


def load_config():
    """Load pipeline configuration."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def step1_ingest(config):
    """Step 1: Ingest data from multiple sources."""
    from src.ingestion.ingestor import DataIngestor

    logger.info("=" * 60)
    logger.info("STEP 1: DATA INGESTION")
    logger.info("=" * 60)

    ingestor = DataIngestor(config["ingestion"])

    datasets = {}
    files = {
        "coursera": DATA_DIR / "coursera_activity.csv",
        "lms": DATA_DIR / "lms_sessions.csv",
        "academic": DATA_DIR / "academic_records.csv",
    }

    for source, path in files.items():
        if path.exists():
            df = ingestor.ingest(str(path), source_type=source)
            datasets[source] = df
            logger.info(f"  Ingested {source}: {len(df)} rows, {len(df.columns)} columns")
        else:
            logger.warning(f"  File not found: {path}")

    return datasets


def step2_clean(config, datasets):
    """Step 2: Clean all datasets."""
    from src.cleaning.cleaner import DataCleaner

    logger.info("=" * 60)
    logger.info("STEP 2: DATA CLEANING")
    logger.info("=" * 60)

    cleaner = DataCleaner(config["cleaning"])

    cleaned = {}
    for source, df in datasets.items():
        cleaned_df = cleaner.clean(df)
        report = cleaner.get_cleaning_report(df, cleaned_df)
        cleaned[source] = cleaned_df
        logger.info(f"  {source}: {report.get('rows_removed', 0)} rows removed, "
                     f"{report.get('missing_filled', 0)} missing values handled")

    return cleaned


def step3_integrate(config, datasets):
    """Step 3: Integrate datasets using mapping table."""
    from src.integration.integrator import DataIntegrator

    logger.info("=" * 60)
    logger.info("STEP 3: DATA INTEGRATION")
    logger.info("=" * 60)

    integrator = DataIntegrator(config["integration"])

    mapping_path = DATA_DIR / "student_mapping.csv"
    if mapping_path.exists():
        integrator.load_mapping_table(str(mapping_path))

    integrated_df = integrator.integrate(datasets)
    report = integrator.get_integration_report()
    logger.info(f"  Integrated dataset: {len(integrated_df)} rows, {len(integrated_df.columns)} columns")
    logger.info(f"  Merge stats: {report}")

    return integrated_df


def step4_features(config, df):
    """Step 4: Engineer features."""
    from src.features.engineer import FeatureEngineer

    logger.info("=" * 60)
    logger.info("STEP 4: FEATURE ENGINEERING")
    logger.info("=" * 60)

    engineer = FeatureEngineer(config["features"])
    features_df = engineer.generate_features(df)
    summary = engineer.get_feature_summary(features_df)

    logger.info(f"  Generated {len(summary)} features")
    for feat, stats in summary.items():
        if isinstance(stats, dict) and "mean" in stats:
            logger.info(f"    {feat}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    return features_df


def step5_analytics(config, df):
    """Step 5: Run analytics."""
    from src.analytics.analyzer import AnalyticsEngine

    logger.info("=" * 60)
    logger.info("STEP 5: ANALYTICS")
    logger.info("=" * 60)

    engine = AnalyticsEngine(config["analytics"])
    results = {}

    # EDA
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        results["distributions"] = engine.compute_distributions(df, numeric_cols[:10])
        results["correlations"] = engine.compute_correlations(df, columns=numeric_cols[:10])
        logger.info(f"  EDA complete: {len(numeric_cols)} numeric columns analyzed")

    # Hypothesis testing - compare active vs withdrawn if available
    if "enrollment_status" in df.columns and "engagement_score" in df.columns:
        statuses = df["enrollment_status"].dropna().unique()
        if len(statuses) >= 2:
            result = engine.t_test(df, "engagement_score", "enrollment_status",
                                   statuses[0], statuses[1])
            results["t_test_engagement"] = result
            logger.info(f"  T-test (engagement by status): p={result['p_value']:.4f}, "
                        f"significant={result['significant']}")

    if "engagement_score" in df.columns and "enrollment_status" in df.columns:
        result = engine.anova(df, "engagement_score", "enrollment_status")
        results["anova_engagement"] = result
        logger.info(f"  ANOVA: p={result['p_value']:.4f}")

    # ML - Regression
    ml_config = config["analytics"]["ml"]
    regression_target = ml_config["regression"]["target"]
    if regression_target in df.columns:
        feature_cols = [c for c in numeric_cols if c != regression_target][:10]
        if feature_cols:
            reg_result = engine.regression(
                df.dropna(subset=[regression_target] + feature_cols),
                target=regression_target,
                features=feature_cols,
                model_type=ml_config["regression"]["model"],
            )
            results["regression"] = reg_result
            logger.info(f"  Regression R2={reg_result['r2']:.3f}, RMSE={reg_result['rmse']:.3f}")

    # ML - Classification
    if "dropout" in df.columns:
        feature_cols = [c for c in numeric_cols if c != "dropout"][:10]
        if feature_cols:
            cls_result = engine.classification(
                df.dropna(subset=["dropout"] + feature_cols),
                target="dropout",
                features=feature_cols,
                model_type=ml_config["classification"]["model"],
            )
            results["classification"] = cls_result
            logger.info(f"  Classification accuracy={cls_result['accuracy']:.3f}, "
                        f"F1={cls_result['f1']:.3f}")

    # ML - Clustering
    cluster_features = [c for c in ml_config["clustering"]["features"] if c in df.columns]
    if len(cluster_features) >= 2:
        cluster_df = df[cluster_features].dropna()
        if len(cluster_df) > ml_config["clustering"]["n_clusters"]:
            clust_result = engine.clustering(
                cluster_df,
                features=cluster_features,
                method=ml_config["clustering"]["method"],
                n_clusters=ml_config["clustering"]["n_clusters"],
            )
            results["clustering"] = clust_result
            logger.info(f"  Clustering silhouette={clust_result['silhouette_score']:.3f}")

    return results


def step6_visualize(config, df, analytics_results):
    """Step 6: Create visualizations."""
    from src.visualization.visualizer import Visualizer

    logger.info("=" * 60)
    logger.info("STEP 6: VISUALIZATION")
    logger.info("=" * 60)

    viz = Visualizer(config["visualization"])
    output_viz = OUTPUT_DIR / "visualizations"
    output_viz.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Distribution plots
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols[:4]:
        fig = viz.plot_distribution(df, col)
        viz.save_figure(fig, str(output_viz / f"dist_{col}.html"))
        figures[f"dist_{col}"] = fig

    # Correlation matrix
    if "correlations" in analytics_results:
        fig = viz.plot_correlation_matrix(analytics_results["correlations"])
        viz.save_figure(fig, str(output_viz / "correlation_matrix.html"))
        figures["correlation_matrix"] = fig

    # Cluster visualization
    if "clustering" in analytics_results:
        cluster_features = config["analytics"]["ml"]["clustering"]["features"]
        available = [c for c in cluster_features if c in df.columns]
        if len(available) >= 2:
            fig = viz.plot_cluster_scatter(
                df.dropna(subset=available[:2]),
                available[0], available[1],
                analytics_results["clustering"]["labels"],
            )
            viz.save_figure(fig, str(output_viz / "clusters.html"))
            figures["clusters"] = fig

    logger.info(f"  Saved {len(figures)} visualizations to {output_viz}")
    return figures


def step7_report(config, df, analytics_results):
    """Step 7: Generate reports."""
    from src.reporting.reporter import ReportGenerator

    logger.info("=" * 60)
    logger.info("STEP 7: REPORT GENERATION")
    logger.info("=" * 60)

    reporter = ReportGenerator(config["reporting"])
    output_reports = OUTPUT_DIR / "reports"
    output_reports.mkdir(parents=True, exist_ok=True)

    # Export data as CSV
    reporter.export_csv(df, "integrated_dataset.csv", str(output_reports))

    # Generate summary stats
    summary = reporter.generate_summary_stats(df)
    reporter.export_csv(summary, "summary_statistics.csv", str(output_reports))

    # Generate PDF report
    reporter.export_pdf(analytics_results, "analytics_report.pdf", str(output_reports))

    # Full report generation
    reporter.generate_report(analytics_results, str(output_reports))

    logger.info(f"  Reports saved to {output_reports}")


def main():
    """Run the complete demo pipeline."""
    logger.info("Learning Analytics Platform - Demo Pipeline")
    logger.info("=" * 60)

    # Load config
    config = load_config()
    logger.info("Configuration loaded")

    # Generate sample data if not present
    if not (DATA_DIR / "coursera_activity.csv").exists():
        logger.info("Generating sample data...")
        import subprocess
        import sys
        subprocess.run([sys.executable, str(DATA_DIR / "generate_sample_data.py")], check=True)

    # Run pipeline
    datasets = step1_ingest(config)
    if not datasets:
        logger.error("No data ingested. Exiting.")
        return

    cleaned = step2_clean(config, datasets)
    integrated = step3_integrate(config, cleaned)

    if integrated.empty:
        logger.error("Integration produced empty dataset. Exiting.")
        return

    featured = step4_features(config, integrated)
    analytics_results = step5_analytics(config, featured)
    step6_visualize(config, featured, analytics_results)
    step7_report(config, featured, analytics_results)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Output saved to: {OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
