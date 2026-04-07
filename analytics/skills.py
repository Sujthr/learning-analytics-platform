"""Skill Intelligence Engine — taxonomy, mapping, scoring, gap analysis."""

import pandas as pd
import numpy as np
from typing import Optional


# Skill taxonomy: category -> skills
SKILL_TAXONOMY = {
    "Data Science": ["Machine Learning", "Statistics", "Data Visualization", "Data Analysis", "R Programming"],
    "Programming": ["Python", "Programming Fundamentals", "SQL", "Databases"],
    "AI & Deep Learning": ["Deep Learning", "Neural Networks", "TensorFlow", "NLP"],
    "Cloud & Infrastructure": ["AWS", "Cloud Computing", "Infrastructure", "Cybersecurity", "Network Security", "Risk Assessment"],
    "Business": ["Project Management", "Agile", "Leadership", "Strategy", "Business Analysis", "Excel", "Business Intelligence"],
    "Marketing & Design": ["Digital Marketing", "SEO", "Analytics", "UX Design", "Prototyping", "User Research"],
    "Finance": ["Finance", "Risk Management", "Investment"],
    "Data Analytics": ["Data Analytics", "Spreadsheets", "Tableau"],
}

# Course -> skills mapping
COURSE_SKILL_MAP = {
    "CRS-001": ["Machine Learning", "Python", "Statistics"],
    "CRS-002": ["Deep Learning", "Neural Networks", "TensorFlow"],
    "CRS-003": ["Python", "Programming Fundamentals"],
    "CRS-004": ["R Programming", "Statistics", "Data Visualization"],
    "CRS-005": ["SQL", "Data Analytics", "Spreadsheets", "Tableau"],
    "CRS-006": ["AWS", "Cloud Computing", "Infrastructure"],
    "CRS-007": ["Digital Marketing", "SEO", "Analytics"],
    "CRS-008": ["Project Management", "Agile", "Leadership"],
    "CRS-009": ["Finance", "Risk Management", "Investment"],
    "CRS-010": ["UX Design", "Prototyping", "User Research"],
    "CRS-011": ["Cybersecurity", "Network Security", "Risk Assessment"],
    "CRS-012": ["Strategy", "Business Analysis", "Leadership"],
    "CRS-013": ["SQL", "Databases", "Data Analysis"],
    "CRS-014": ["NLP", "Deep Learning", "Python"],
    "CRS-015": ["Excel", "Data Analysis", "Business Intelligence"],
}


def _get_skill_category(skill: str) -> str:
    for cat, skills in SKILL_TAXONOMY.items():
        if skill in skills:
            return cat
    return "Other"


class SkillIntelligenceEngine:
    """Map courses to skills, compute learner skill profiles, and analyze gaps."""

    def __init__(
        self,
        course_activity_df: pd.DataFrame,
        course_skill_map: dict | None = None,
    ):
        self.activity = course_activity_df.copy()
        self.skill_map = course_skill_map or COURSE_SKILL_MAP

    def compute_skill_scores(self) -> pd.DataFrame:
        """Compute skill score per learner based on course completion/progress.

        Skill score = weighted average of progress across courses teaching that skill.
        """
        rows = []
        for _, record in self.activity.iterrows():
            course_id = record.get("course_id", "")
            skills = self.skill_map.get(course_id, [])
            for skill in skills:
                rows.append({
                    "email": record["email"],
                    "skill_name": skill,
                    "skill_category": _get_skill_category(skill),
                    "course_id": course_id,
                    "progress_pct": record.get("progress_pct", 0),
                    "is_completed": record.get("is_completed", False),
                    "grade": record.get("grade", None),
                })

        if not rows:
            return pd.DataFrame()

        skill_df = pd.DataFrame(rows)

        # Skill score = mean progress across contributing courses, boosted by completions
        per_learner_skill = skill_df.groupby(["email", "skill_name", "skill_category"]).agg(
            avg_progress=("progress_pct", "mean"),
            courses_contributing=("course_id", "count"),
            courses_completed=("is_completed", "sum"),
        ).reset_index()

        # Normalize: progress gives base score, completion bonus
        per_learner_skill["skill_score"] = (
            per_learner_skill["avg_progress"] * 0.7
            + (per_learner_skill["courses_completed"] / per_learner_skill["courses_contributing"]) * 30
        ).clip(0, 100).round(1)

        return per_learner_skill

    def learner_skill_profile(self, email: str) -> pd.DataFrame:
        """Full skill profile for a specific learner."""
        scores = self.compute_skill_scores()
        return scores[scores["email"] == email].sort_values("skill_score", ascending=False)

    def skill_gap_analysis(self, target_skills: list[str] | None = None) -> pd.DataFrame:
        """Identify skill gaps across the organization.

        Returns each skill with its average score and coverage (% of learners with it).
        """
        scores = self.compute_skill_scores()
        if scores.empty:
            return pd.DataFrame()

        total_learners = scores["email"].nunique()

        gap = scores.groupby(["skill_name", "skill_category"]).agg(
            avg_score=("skill_score", "mean"),
            learners_with_skill=("email", "nunique"),
        ).reset_index()
        gap["coverage_pct"] = (gap["learners_with_skill"] / total_learners * 100).round(1)
        gap["avg_score"] = gap["avg_score"].round(1)
        gap["gap_index"] = (100 - gap["avg_score"]).round(1)

        if target_skills:
            gap = gap[gap["skill_name"].isin(target_skills)]

        return gap.sort_values("gap_index", ascending=False)

    def org_skill_distribution(self) -> pd.DataFrame:
        """Skill distribution by category across the org."""
        scores = self.compute_skill_scores()
        if scores.empty:
            return pd.DataFrame()

        dist = scores.groupby("skill_category").agg(
            total_learners=("email", "nunique"),
            avg_score=("skill_score", "mean"),
            total_skill_entries=("skill_name", "count"),
            unique_skills=("skill_name", "nunique"),
        ).reset_index()
        dist["avg_score"] = dist["avg_score"].round(1)
        return dist.sort_values("total_learners", ascending=False)

    def skill_progression_timeline(self, email: str) -> pd.DataFrame:
        """Show how a learner's skills have developed over time (course-by-course)."""
        learner_act = self.activity[self.activity["email"] == email].copy()
        if learner_act.empty:
            return pd.DataFrame()

        if "enrollment_ts" in learner_act.columns:
            learner_act = learner_act.sort_values("enrollment_ts")

        rows = []
        cumulative_skills = {}
        for _, record in learner_act.iterrows():
            skills = self.skill_map.get(record.get("course_id", ""), [])
            for skill in skills:
                score = record.get("progress_pct", 0) * 0.7
                if record.get("is_completed"):
                    score += 30
                cumulative_skills[skill] = max(cumulative_skills.get(skill, 0), score)
                rows.append({
                    "course_id": record.get("course_id"),
                    "enrollment_ts": record.get("enrollment_ts"),
                    "skill_name": skill,
                    "skill_score": round(min(cumulative_skills[skill], 100), 1),
                })

        return pd.DataFrame(rows)

    def get_all_metrics(self) -> dict:
        scores = self.compute_skill_scores()
        gaps = self.skill_gap_analysis()
        dist = self.org_skill_distribution()
        return {
            "total_skills_tracked": int(scores["skill_name"].nunique()) if not scores.empty else 0,
            "avg_skill_score": round(scores["skill_score"].mean(), 1) if not scores.empty else 0,
            "skill_gaps": gaps.head(10).to_dict(orient="records") if not gaps.empty else [],
            "org_distribution": dist.to_dict(orient="records") if not dist.empty else [],
        }
