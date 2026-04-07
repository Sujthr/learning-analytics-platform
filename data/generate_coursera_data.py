"""
Generate realistic Coursera-style datasets for the Learning Analytics Platform.

Produces 4 CSV files matching real Coursera enterprise export schemas:
  1. course_activity.csv       — per-learner course enrollment & progress
  2. program_activity.csv      — per-learner program-level progress
  3. specialization_activity.csv — per-learner specialization progress
  4. video_clip_activity.csv   — per-learner video watch events
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = Path(__file__).parent / "coursera"
OUT_DIR.mkdir(exist_ok=True)

# ── Reference data ──────────────────────────────────────────────────

NUM_LEARNERS = 500

BUSINESS_UNITS = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Product", "Data Science", "Operations"]
ROLES = ["Individual Contributor", "Manager", "Director", "VP", "Associate", "Senior Engineer", "Analyst", "Lead"]
LOCATIONS = ["New York", "San Francisco", "London", "Bangalore", "Singapore", "Toronto", "Berlin", "Sydney"]

COURSES = [
    {"id": "CRS-001", "name": "Machine Learning", "slug": "machine-learning", "institution": "Stanford", "category": "Data Science", "hours": 60, "skills": ["Machine Learning", "Python", "Statistics"]},
    {"id": "CRS-002", "name": "Deep Learning Specialization", "slug": "deep-learning", "institution": "deeplearning.ai", "category": "AI", "hours": 80, "skills": ["Deep Learning", "Neural Networks", "TensorFlow"]},
    {"id": "CRS-003", "name": "Python for Everybody", "slug": "python-for-everybody", "institution": "U-Michigan", "category": "Programming", "hours": 40, "skills": ["Python", "Programming Fundamentals"]},
    {"id": "CRS-004", "name": "Data Science with R", "slug": "data-science-r", "institution": "Johns Hopkins", "category": "Data Science", "hours": 50, "skills": ["R Programming", "Statistics", "Data Visualization"]},
    {"id": "CRS-005", "name": "Google Data Analytics", "slug": "google-data-analytics", "institution": "Google", "category": "Data Analytics", "hours": 45, "skills": ["SQL", "Data Analytics", "Spreadsheets", "Tableau"]},
    {"id": "CRS-006", "name": "AWS Cloud Practitioner", "slug": "aws-cloud-practitioner", "institution": "AWS", "category": "Cloud", "hours": 30, "skills": ["AWS", "Cloud Computing", "Infrastructure"]},
    {"id": "CRS-007", "name": "Digital Marketing", "slug": "digital-marketing", "institution": "U-Illinois", "category": "Marketing", "hours": 35, "skills": ["Digital Marketing", "SEO", "Analytics"]},
    {"id": "CRS-008", "name": "Project Management Professional", "slug": "pmp-cert", "institution": "Google", "category": "Management", "hours": 50, "skills": ["Project Management", "Agile", "Leadership"]},
    {"id": "CRS-009", "name": "Financial Markets", "slug": "financial-markets", "institution": "Yale", "category": "Finance", "hours": 25, "skills": ["Finance", "Risk Management", "Investment"]},
    {"id": "CRS-010", "name": "UX Design", "slug": "ux-design", "institution": "Google", "category": "Design", "hours": 40, "skills": ["UX Design", "Prototyping", "User Research"]},
    {"id": "CRS-011", "name": "Cybersecurity Fundamentals", "slug": "cybersecurity-fund", "institution": "IBM", "category": "Security", "hours": 35, "skills": ["Cybersecurity", "Network Security", "Risk Assessment"]},
    {"id": "CRS-012", "name": "Business Strategy", "slug": "business-strategy", "institution": "UVA", "category": "Business", "hours": 20, "skills": ["Strategy", "Business Analysis", "Leadership"]},
    {"id": "CRS-013", "name": "SQL for Data Science", "slug": "sql-data-science", "institution": "UC Davis", "category": "Data Science", "hours": 15, "skills": ["SQL", "Databases", "Data Analysis"]},
    {"id": "CRS-014", "name": "Natural Language Processing", "slug": "nlp-specialization", "institution": "deeplearning.ai", "category": "AI", "hours": 55, "skills": ["NLP", "Deep Learning", "Python"]},
    {"id": "CRS-015", "name": "Excel Skills for Business", "slug": "excel-business", "institution": "Macquarie", "category": "Business", "hours": 25, "skills": ["Excel", "Data Analysis", "Business Intelligence"]},
]

PROGRAMS = [
    {"id": "PRG-001", "name": "Data Science Career Track", "slug": "ds-career-track", "courses": ["CRS-001", "CRS-004", "CRS-005", "CRS-013"]},
    {"id": "PRG-002", "name": "AI Engineering Program", "slug": "ai-engineering", "courses": ["CRS-001", "CRS-002", "CRS-014"]},
    {"id": "PRG-003", "name": "Cloud & DevOps", "slug": "cloud-devops", "courses": ["CRS-006", "CRS-011"]},
    {"id": "PRG-004", "name": "Business Leadership", "slug": "business-leadership", "courses": ["CRS-008", "CRS-009", "CRS-012"]},
    {"id": "PRG-005", "name": "Digital Skills", "slug": "digital-skills", "courses": ["CRS-003", "CRS-007", "CRS-010", "CRS-015"]},
]

SPECIALIZATIONS = [
    {"id": "SPC-001", "name": "Applied Data Science", "slug": "applied-ds", "courses": ["CRS-001", "CRS-004", "CRS-013"]},
    {"id": "SPC-002", "name": "Deep Learning Specialization", "slug": "deep-learning-spec", "courses": ["CRS-002", "CRS-014"]},
    {"id": "SPC-003", "name": "Google IT & Analytics", "slug": "google-it-analytics", "courses": ["CRS-005", "CRS-006"]},
    {"id": "SPC-004", "name": "Business Essentials", "slug": "business-essentials", "courses": ["CRS-008", "CRS-012", "CRS-015"]},
]

VIDEOS_PER_COURSE = 12  # average


def _random_ts(start: datetime, end: datetime) -> datetime:
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=random.uniform(0, delta))


def generate_learners() -> pd.DataFrame:
    rows = []
    for i in range(1, NUM_LEARNERS + 1):
        first = random.choice(["Alex", "Jordan", "Taylor", "Morgan", "Casey",
                               "Riley", "Avery", "Quinn", "Harper", "Drew",
                               "Sanjay", "Priya", "Wei", "Yuki", "Ahmed"])
        last = random.choice(["Smith", "Patel", "Kim", "Garcia", "Chen",
                              "Johnson", "Williams", "Brown", "Lee", "Kumar"])
        rows.append({
            "email": f"{first.lower()}.{last.lower()}{i}@company.com",
            "external_id": f"EXT-{i:04d}",
            "name": f"{first} {last}",
            "business_unit": random.choice(BUSINESS_UNITS),
            "role": random.choice(ROLES),
            "location": random.choice(LOCATIONS),
        })
    return pd.DataFrame(rows)


def generate_course_activity(learners: pd.DataFrame) -> pd.DataFrame:
    """Each learner enrols in 2-8 courses."""
    rows = []
    start_window = datetime(2024, 1, 1)
    end_window = datetime(2025, 12, 31)

    for _, learner in learners.iterrows():
        n_courses = random.randint(2, 8)
        chosen = random.sample(COURSES, n_courses)
        for course in chosen:
            enroll = _random_ts(start_window, end_window - timedelta(days=60))
            # determine learner "type" for realistic distributions
            learner_type = random.choices(
                ["high", "medium", "low", "dropout"],
                weights=[0.25, 0.35, 0.25, 0.15],
            )[0]

            if learner_type == "high":
                progress = round(random.uniform(80, 100), 1)
            elif learner_type == "medium":
                progress = round(random.uniform(40, 79), 1)
            elif learner_type == "low":
                progress = round(random.uniform(10, 39), 1)
            else:
                progress = round(random.uniform(0, 15), 1)

            is_completed = progress >= 90
            grade = round(random.uniform(70, 100), 1) if is_completed else None
            hours = round(course["hours"] * (progress / 100) * random.uniform(0.8, 1.3), 1)
            completion_ts = enroll + timedelta(days=random.randint(14, 120)) if is_completed else None
            last_activity = (
                completion_ts if completion_ts
                else _random_ts(enroll, min(enroll + timedelta(days=180), end_window))
            )

            rows.append({
                "Name": learner["name"],
                "Email": learner["email"],
                "External ID": learner["external_id"],
                "Business Unit": learner["business_unit"],
                "Role": learner["role"],
                "Location": learner["location"],
                "Course Id": course["id"],
                "Course Name": course["name"],
                "Course Slug": course["slug"],
                "Institution": course["institution"],
                "Enrollment Timestamp": enroll.strftime("%Y-%m-%d %H:%M:%S"),
                "Completion Timestamp": completion_ts.strftime("%Y-%m-%d %H:%M:%S") if completion_ts else "",
                "Progress (%)": progress,
                "Grade (%)": grade if grade else "",
                "Learning Hours": hours,
                "Completed": "Yes" if is_completed else "No",
                "Last Activity Timestamp": last_activity.strftime("%Y-%m-%d %H:%M:%S"),
            })

    return pd.DataFrame(rows)


def generate_program_activity(learners: pd.DataFrame, course_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, learner in learners.iterrows():
        n_programs = random.randint(0, 2)
        if n_programs == 0:
            continue
        chosen = random.sample(PROGRAMS, n_programs)
        for prog in chosen:
            # Check which courses in this program the learner enrolled in
            learner_courses = course_df[course_df["Email"] == learner["email"]]["Course Id"].tolist()
            prog_courses_enrolled = [c for c in prog["courses"] if c in learner_courses]
            if not prog_courses_enrolled:
                continue
            completed_count = 0
            total_hours = 0.0
            for cid in prog_courses_enrolled:
                rec = course_df[(course_df["Email"] == learner["email"]) & (course_df["Course Id"] == cid)]
                if not rec.empty:
                    if rec.iloc[0]["Completed"] == "Yes":
                        completed_count += 1
                    total_hours += float(rec.iloc[0]["Learning Hours"])

            total_in_prog = len(prog["courses"])
            progress = round((completed_count / total_in_prog) * 100, 1)
            enroll_ts = _random_ts(datetime(2024, 1, 1), datetime(2025, 6, 1))
            last_ts = _random_ts(enroll_ts, datetime(2025, 12, 31))

            rows.append({
                "Name": learner["name"],
                "Email": learner["email"],
                "External ID": learner["external_id"],
                "Business Unit": learner["business_unit"],
                "Program Id": prog["id"],
                "Program Name": prog["name"],
                "Program Slug": prog["slug"],
                "Total Courses in Program": total_in_prog,
                "Courses Completed": completed_count,
                "Progress (%)": progress,
                "Learning Hours": round(total_hours, 1),
                "Completed": "Yes" if completed_count == total_in_prog else "No",
                "Enrollment Timestamp": enroll_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "Last Activity Timestamp": last_ts.strftime("%Y-%m-%d %H:%M:%S"),
            })

    return pd.DataFrame(rows)


def generate_specialization_activity(learners: pd.DataFrame, course_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, learner in learners.iterrows():
        n_specs = random.randint(0, 2)
        if n_specs == 0:
            continue
        chosen = random.sample(SPECIALIZATIONS, min(n_specs, len(SPECIALIZATIONS)))
        for spec in chosen:
            learner_courses = course_df[course_df["Email"] == learner["email"]]["Course Id"].tolist()
            spec_enrolled = [c for c in spec["courses"] if c in learner_courses]
            if not spec_enrolled:
                continue

            completed_count = 0
            total_hours = 0.0
            for cid in spec_enrolled:
                rec = course_df[(course_df["Email"] == learner["email"]) & (course_df["Course Id"] == cid)]
                if not rec.empty:
                    if rec.iloc[0]["Completed"] == "Yes":
                        completed_count += 1
                    total_hours += float(rec.iloc[0]["Learning Hours"])

            total_in_spec = len(spec["courses"])
            progress = round((completed_count / total_in_spec) * 100, 1)
            enroll_ts = _random_ts(datetime(2024, 1, 1), datetime(2025, 6, 1))
            last_ts = _random_ts(enroll_ts, datetime(2025, 12, 31))

            rows.append({
                "Name": learner["name"],
                "Email": learner["email"],
                "External ID": learner["external_id"],
                "Business Unit": learner["business_unit"],
                "Specialization Id": spec["id"],
                "Specialization Name": spec["name"],
                "Specialization Slug": spec["slug"],
                "Total Courses": total_in_spec,
                "Courses Completed": completed_count,
                "Progress (%)": progress,
                "Learning Hours": round(total_hours, 1),
                "Completed": "Yes" if completed_count == total_in_spec else "No",
                "Enrollment Timestamp": enroll_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "Last Activity Timestamp": last_ts.strftime("%Y-%m-%d %H:%M:%S"),
            })

    return pd.DataFrame(rows)


def generate_video_activity(learners: pd.DataFrame, course_df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-video watch events for courses learners are enrolled in."""
    rows = []
    for _, learner in learners.iterrows():
        enrolled = course_df[course_df["Email"] == learner["email"]]
        for _, enrollment in enrolled.iterrows():
            n_videos = random.randint(6, VIDEOS_PER_COURSE + 6)
            progress = float(enrollment["Progress (%)"])
            # learners with higher progress watch more videos
            videos_watched = max(1, int(n_videos * (progress / 100) * random.uniform(0.7, 1.1)))
            for v in range(1, videos_watched + 1):
                total_secs = random.randint(180, 1200)  # 3-20 min clips
                watch_pct = min(1.0, random.betavariate(
                    2 + progress / 30, 2 + (100 - progress) / 50
                ))
                watch_secs = round(total_secs * watch_pct, 1)
                watch_count = random.choices([1, 2, 3, 4], weights=[0.5, 0.3, 0.15, 0.05])[0]
                ts = _random_ts(datetime(2024, 3, 1), datetime(2025, 12, 1))

                rows.append({
                    "Email": learner["email"],
                    "External ID": learner["external_id"],
                    "Course Id": enrollment["Course Id"],
                    "Course Name": enrollment["Course Name"],
                    "Video Name": f"{enrollment['Course Name']} - Lecture {v}",
                    "Video Id": f"VID-{enrollment['Course Id']}-{v:03d}",
                    "Watch Duration (seconds)": watch_secs,
                    "Total Duration (seconds)": total_secs,
                    "Completion (%)": round(watch_pct * 100, 1),
                    "Watch Count": watch_count,
                    "Last Watch Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                })

    return pd.DataFrame(rows)


def main():
    print("Generating learner profiles...")
    learners = generate_learners()

    print("Generating course activity...")
    course_df = generate_course_activity(learners)
    course_df.to_csv(OUT_DIR / "course_activity.csv", index=False)
    print(f"  > course_activity.csv: {len(course_df)} rows")

    print("Generating program activity...")
    program_df = generate_program_activity(learners, course_df)
    program_df.to_csv(OUT_DIR / "program_activity.csv", index=False)
    print(f"  > program_activity.csv: {len(program_df)} rows")

    print("Generating specialization activity...")
    spec_df = generate_specialization_activity(learners, course_df)
    spec_df.to_csv(OUT_DIR / "specialization_activity.csv", index=False)
    print(f"  > specialization_activity.csv: {len(spec_df)} rows")

    print("Generating video clip activity...")
    video_df = generate_video_activity(learners, course_df)
    video_df.to_csv(OUT_DIR / "video_clip_activity.csv", index=False)
    print(f"  > video_clip_activity.csv: {len(video_df)} rows")

    # Save learner reference
    learners.to_csv(OUT_DIR / "learners_reference.csv", index=False)
    print(f"  > learners_reference.csv: {len(learners)} rows")

    print("\nAll datasets generated in", OUT_DIR)


if __name__ == "__main__":
    main()
