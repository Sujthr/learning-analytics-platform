"""Report Generator for the Learning Analytics Platform.

Exports analytics results as PDF reports and CSV files with
research-ready formatting.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from fpdf import FPDF

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates PDF and CSV reports from analytics results."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config.get("output_dir", "output/reports")
        self.decimal_places = config.get("decimal_places", 3)

    def generate_report(self, results: dict, output_dir: Optional[str] = None) -> str:
        """Generate a full report package from analytics results.

        Creates both PDF and CSV exports.
        """
        out = Path(output_dir or self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Export individual result tables as CSV
        if "distributions" in results:
            dist_df = pd.DataFrame(results["distributions"]).T
            self.export_csv(dist_df, "distribution_stats.csv", str(out))

        if "correlations" in results:
            self.export_csv(results["correlations"], "correlations.csv", str(out))

        # Generate PDF
        self.export_pdf(results, "full_report.pdf", str(out))

        logger.info("Full report generated in %s", out)
        return str(out)

    def export_csv(
        self,
        data: Union[pd.DataFrame, dict],
        filename: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """Export DataFrame or dict to CSV."""
        out = Path(output_dir or self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename

        if isinstance(data, dict):
            data = pd.DataFrame(data)

        data.to_csv(path, index=True)
        logger.info("Exported CSV: %s (%d rows)", path, len(data))
        return str(path)

    def export_pdf(
        self,
        results: dict,
        filename: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """Generate a formatted PDF report."""
        out = Path(output_dir or self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title page
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 24)
        pdf.cell(0, 60, "", ln=True)
        pdf.cell(0, 15, "Learning Analytics Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 14)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        pdf.cell(0, 10, "Learning Analytics Platform", ln=True, align="C")

        # Executive Summary
        pdf.add_page()
        self._section_header(pdf, "1. Executive Summary")
        summary_lines = self._build_summary(results)
        for line in summary_lines:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, line)
            pdf.ln(2)

        # Data Overview
        if "distributions" in results:
            self._section_header(pdf, "2. Data Overview")
            self._add_dict_table(pdf, results["distributions"])

        # Statistical Tests
        test_section = False
        for key in ["t_test_engagement", "anova_engagement"]:
            if key in results:
                if not test_section:
                    self._section_header(pdf, "3. Statistical Tests")
                    test_section = True
                self._add_test_result(pdf, results[key])

        # ML Results
        ml_section = False
        for key in ["regression", "classification", "clustering"]:
            if key in results:
                if not ml_section:
                    self._section_header(pdf, "4. Machine Learning Results")
                    ml_section = True
                self._add_ml_result(pdf, key, results[key])

        pdf.output(str(path))
        logger.info("Exported PDF: %s", path)
        return str(path)

    def format_table(self, df: pd.DataFrame, decimal_places: Optional[int] = None) -> pd.DataFrame:
        """Format DataFrame for research publication.

        Rounds numeric columns and adds significance stars for p-values.
        """
        dp = decimal_places or self.decimal_places
        formatted = df.copy()

        for col in formatted.select_dtypes(include=["number"]).columns:
            if "p_value" in col or col == "p_value":
                formatted[col] = formatted[col].apply(lambda p: self._format_pvalue(p, dp))
            else:
                formatted[col] = formatted[col].round(dp)

        return formatted

    def generate_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create publication-ready descriptive statistics table."""
        numeric = df.select_dtypes(include=["number"])
        stats = numeric.describe().T
        stats["skew"] = numeric.skew()
        stats["kurtosis"] = numeric.kurtosis()
        stats = stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skew", "kurtosis"]]
        stats = stats.round(self.decimal_places)
        return stats

    # ---- Private helpers ----

    def _section_header(self, pdf: FPDF, title: str):
        pdf.ln(8)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_draw_color(41, 128, 185)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

    def _build_summary(self, results: dict) -> list[str]:
        lines = []
        if "distributions" in results:
            n_features = len(results["distributions"])
            lines.append(f"Analyzed {n_features} numeric features.")

        for key in ["t_test_engagement", "anova_engagement"]:
            if key in results:
                r = results[key]
                sig = "statistically significant" if r.get("significant") else "not significant"
                lines.append(f"{r.get('test', key)}: p={r['p_value']:.4f} ({sig} at alpha={r.get('alpha', 0.05)}).")

        if "regression" in results:
            r = results["regression"]
            lines.append(f"Regression ({r['model_type']}): R2={r['r2']:.3f}, RMSE={r['rmse']:.3f}.")

        if "classification" in results:
            r = results["classification"]
            lines.append(f"Classification ({r['model_type']}): Accuracy={r['accuracy']:.3f}, F1={r['f1']:.3f}.")

        if "clustering" in results:
            r = results["clustering"]
            lines.append(f"Clustering ({r['method']}): {r['n_clusters']} clusters, silhouette={r['silhouette_score']:.3f}.")

        if not lines:
            lines.append("No analytics results available.")

        return lines

    def _add_dict_table(self, pdf: FPDF, data: dict):
        """Add a dictionary-based table to the PDF."""
        if not data:
            return

        pdf.set_font("Helvetica", "", 8)
        first_key = next(iter(data))
        if isinstance(data[first_key], dict):
            cols = ["Feature"] + list(data[first_key].keys())
        else:
            cols = ["Key", "Value"]

        # Header
        col_width = min(180 / len(cols), 35)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(41, 128, 185)
        pdf.set_text_color(255, 255, 255)
        for col in cols:
            pdf.cell(col_width, 7, str(col)[:15], border=1, fill=True, align="C")
        pdf.ln()

        # Rows
        pdf.set_text_color(0, 0, 0)
        for i, (key, value) in enumerate(list(data.items())[:20]):
            pdf.set_fill_color(240, 240, 240) if i % 2 else pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "", 8)

            if isinstance(value, dict):
                pdf.cell(col_width, 6, str(key)[:15], border=1, fill=True)
                for v in value.values():
                    pdf.cell(col_width, 6, self._fmt(v), border=1, fill=True, align="R")
            else:
                pdf.cell(col_width, 6, str(key)[:15], border=1, fill=True)
                pdf.cell(col_width, 6, self._fmt(value), border=1, fill=True, align="R")
            pdf.ln()

    def _add_test_result(self, pdf: FPDF, result: dict):
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, result.get("test", "Test"), ln=True)
        pdf.set_font("Helvetica", "", 9)
        for key in ["column", "groups", "t_statistic", "f_statistic", "p_value", "significant"]:
            if key in result:
                val = result[key]
                pdf.cell(0, 5, f"  {key}: {self._fmt(val)}", ln=True)
        pdf.ln(3)

    def _add_ml_result(self, pdf: FPDF, name: str, result: dict):
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, f"{name.title()} ({result.get('model_type', result.get('method', ''))})", ln=True)
        pdf.set_font("Helvetica", "", 9)

        skip = {"model", "feature_importances", "cv_scores", "labels", "cluster_centers", "confusion_matrix", "features"}
        for key, val in result.items():
            if key in skip:
                continue
            pdf.cell(0, 5, f"  {key}: {self._fmt(val)}", ln=True)
        pdf.ln(3)

    def _fmt(self, value) -> str:
        if isinstance(value, float):
            return f"{value:.{self.decimal_places}f}"
        if isinstance(value, (list, dict)):
            return str(value)[:40]
        return str(value)

    @staticmethod
    def _format_pvalue(p: float, dp: int) -> str:
        if p < 0.001:
            return f"{p:.{dp}f}***"
        elif p < 0.01:
            return f"{p:.{dp}f}**"
        elif p < 0.05:
            return f"{p:.{dp}f}*"
        return f"{p:.{dp}f}"
