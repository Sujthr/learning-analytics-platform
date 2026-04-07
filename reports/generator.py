"""Report Generator — PDF and Excel export with analytics summaries."""

import os
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate downloadable PDF and Excel reports."""

    def __init__(self, output_dir: str = "output/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_pdf_report(
        self,
        transformed_data: dict[str, pd.DataFrame],
        analytics_data: dict,
    ) -> str:
        """Generate a comprehensive PDF report.

        Uses matplotlib for charts embedded in PDF via matplotlib's PdfPages.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"analytics_report_{timestamp}.pdf"

        ca = transformed_data.get("course_activity", pd.DataFrame())
        va = transformed_data.get("video_activity", pd.DataFrame())

        with PdfPages(str(filepath)) as pdf:
            # Title page
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.text(0.5, 0.7, "Learning Analytics Platform", transform=ax.transAxes,
                    fontsize=28, ha="center", fontweight="bold", color="#1f77b4")
            ax.text(0.5, 0.6, "Analytics Report", transform=ax.transAxes,
                    fontsize=20, ha="center", color="#666")
            ax.text(0.5, 0.5, f"Generated: {datetime.now().strftime('%B %d, %Y %I:%M %p')}",
                    transform=ax.transAxes, fontsize=12, ha="center", color="#999")

            kpis = analytics_data.get("kpis", {})
            if kpis:
                kpi_text = (
                    f"Total Learners: {kpis.get('total_learners', 'N/A')}  |  "
                    f"Courses: {kpis.get('total_courses', 'N/A')}  |  "
                    f"Completion: {kpis.get('overall_completion_rate', 'N/A')}%  |  "
                    f"Avg Engagement: {kpis.get('avg_engagement_score', 'N/A')}"
                )
                ax.text(0.5, 0.35, kpi_text, transform=ax.transAxes,
                        fontsize=11, ha="center", color="#333",
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0"))
            pdf.savefig(fig)
            plt.close(fig)

            # Completion by course
            if not ca.empty:
                fig, ax = plt.subplots(figsize=(11, 7))
                rates = ca.groupby("course_name")["is_completed"].mean().sort_values() * 100
                rates.plot(kind="barh", ax=ax, color="#1f77b4")
                ax.set_title("Completion Rate by Course", fontsize=16, pad=15)
                ax.set_xlabel("Completion %")
                ax.set_ylabel("")
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # Progress distribution
            if not ca.empty:
                fig, axes = plt.subplots(1, 2, figsize=(11, 5))
                ca["progress_pct"].hist(bins=20, ax=axes[0], color="#ff7f0e", edgecolor="white")
                axes[0].set_title("Progress Distribution")
                axes[0].set_xlabel("Progress %")
                axes[0].set_ylabel("Learners")

                if "learning_hours" in ca.columns:
                    ca.groupby("email")["learning_hours"].sum().hist(
                        bins=20, ax=axes[1], color="#2ca02c", edgecolor="white"
                    )
                    axes[1].set_title("Learning Hours Distribution")
                    axes[1].set_xlabel("Total Hours")
                    axes[1].set_ylabel("Learners")
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # Engagement distribution
            engagement_data = analytics_data.get("engagement", [])
            if engagement_data:
                eng_df = pd.DataFrame(engagement_data)
                if "score" in eng_df.columns:
                    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
                    eng_df["score"].hist(bins=20, ax=axes[0], color="#9467bd", edgecolor="white")
                    axes[0].set_title("Engagement Score Distribution")

                    if "category" in eng_df.columns:
                        cat_counts = eng_df["category"].value_counts()
                        colors = {"High": "#2ca02c", "Medium": "#ff7f0e", "Low": "#d62728", "At-Risk": "#7f0000"}
                        cat_counts.plot(
                            kind="pie", ax=axes[1], autopct="%1.1f%%",
                            colors=[colors.get(c, "#999") for c in cat_counts.index]
                        )
                        axes[1].set_title("Engagement Categories")
                        axes[1].set_ylabel("")
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

            # Insights page
            insights = analytics_data.get("insights", [])
            if insights:
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis("off")
                ax.text(0.5, 0.95, "Key Insights", transform=ax.transAxes,
                        fontsize=20, ha="center", fontweight="bold")

                y_pos = 0.88
                for i, insight in enumerate(insights[:12]):
                    severity = insight.get("severity", "info").upper()
                    title = insight.get("title", "")
                    color = {"CRITICAL": "red", "WARNING": "orange", "INFO": "blue"}.get(severity, "gray")
                    ax.text(0.05, y_pos, f"[{severity}]", transform=ax.transAxes,
                            fontsize=9, color=color, fontweight="bold")
                    ax.text(0.15, y_pos, title[:80], transform=ax.transAxes, fontsize=9)
                    y_pos -= 0.06
                    if y_pos < 0.05:
                        break

                pdf.savefig(fig)
                plt.close(fig)

        logger.info(f"PDF report saved to {filepath}")
        return str(filepath)

    def generate_excel_report(
        self,
        transformed_data: dict[str, pd.DataFrame],
        analytics_data: dict,
    ) -> str:
        """Generate a multi-sheet Excel workbook with all data and analytics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"analytics_report_{timestamp}.xlsx"

        with pd.ExcelWriter(str(filepath), engine="openpyxl") as writer:
            # Raw data sheets
            for name, df in transformed_data.items():
                if not df.empty:
                    sheet_name = name[:31]  # Excel sheet name limit
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            # KPIs
            kpis = analytics_data.get("kpis", {})
            if kpis:
                kpi_df = pd.DataFrame([kpis])
                kpi_df.to_excel(writer, sheet_name="KPIs", index=False)

            # Engagement scores
            engagement = analytics_data.get("engagement", [])
            if engagement:
                pd.DataFrame(engagement).to_excel(
                    writer, sheet_name="Engagement Scores", index=False
                )

            # Insights
            insights = analytics_data.get("insights", [])
            if insights:
                pd.DataFrame(insights).to_excel(
                    writer, sheet_name="Insights", index=False
                )

            # Predictions summary
            predictions = analytics_data.get("predictions", {})
            if predictions:
                pred_rows = []
                for task, result in predictions.items():
                    if isinstance(result, dict) and "accuracy" in result:
                        pred_rows.append({
                            "Task": task,
                            "Model": result.get("model_type"),
                            "Accuracy": result.get("accuracy"),
                            "Precision": result.get("precision"),
                            "Recall": result.get("recall"),
                            "F1": result.get("f1"),
                            "AUC-ROC": result.get("auc_roc"),
                            "CV Accuracy": result.get("cv_mean_accuracy"),
                        })
                if pred_rows:
                    pd.DataFrame(pred_rows).to_excel(
                        writer, sheet_name="Prediction Models", index=False
                    )

        logger.info(f"Excel report saved to {filepath}")
        return str(filepath)

    def export_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        format: str = "csv",
    ) -> str:
        """Export a single DataFrame as CSV or Excel."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if format == "excel":
            filepath = self.output_dir / f"{name}_{timestamp}.xlsx"
            df.to_excel(str(filepath), index=False)
        else:
            filepath = self.output_dir / f"{name}_{timestamp}.csv"
            df.to_csv(str(filepath), index=False)

        logger.info(f"Exported {name} to {filepath}")
        return str(filepath)
