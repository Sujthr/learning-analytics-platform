"""Generate realistic sample datasets for the Learning Analytics Platform."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent
NUM_STUDENTS = 200
NUM_COURSES = 10
SEMESTERS = ["2024-Fall", "2025-Spring"]


def generate_student_mapping():
    """Generate student ID mapping table across systems."""
    records = []
    for i in range(1, NUM_STUDENTS + 1):
        records.append({
            "student_id": f"STU-{i:04d}",
            "coursera_id": f"CR-{1000 + i}",
            "lms_id": f"LMS-{2000 + i}",
            "academic_id": f"AC-{3000 + i}",
        })
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "student_mapping.csv", index=False)
    return df


def generate_coursera_data(mapping_df):
    """Generate Coursera activity logs."""
    activities = ["video_watch", "quiz_attempt", "assignment_submit", "forum_post", "reading"]
    records = []
    start_date = datetime(2024, 9, 1)

    for _, row in mapping_df.iterrows():
        n_activities = np.random.randint(20, 150)
        for _ in range(n_activities):
            activity = np.random.choice(activities, p=[0.35, 0.2, 0.15, 0.1, 0.2])
            ts = start_date + timedelta(
                days=np.random.randint(0, 240),
                hours=np.random.randint(8, 23),
                minutes=np.random.randint(0, 60),
            )
            record = {
                "student_id": row["coursera_id"],
                "course_id": f"COURSE-{np.random.randint(1, NUM_COURSES + 1):03d}",
                "timestamp": ts.strftime("%m/%d/%Y %I:%M %p"),  # US format intentionally
                "activity_type": activity,
                "duration_minutes": round(np.random.exponential(20), 1) if activity != "forum_post" else round(np.random.exponential(5), 1),
            }
            if activity == "video_watch":
                record["video_id"] = f"VID-{np.random.randint(1, 50):03d}"
                record["completion_pct"] = round(min(np.random.beta(5, 2), 1.0), 2)
            elif activity == "quiz_attempt":
                record["score"] = round(np.random.normal(72, 15), 1)
            records.append(record)

    df = pd.DataFrame(records)
    # Introduce some missing values
    mask = np.random.random(len(df)) < 0.03
    df.loc[mask, "duration_minutes"] = np.nan
    # Add some duplicates
    dupes = df.sample(n=int(len(df) * 0.02), random_state=42)
    df = pd.concat([df, dupes], ignore_index=True)
    df.to_csv(OUTPUT_DIR / "coursera_activity.csv", index=False)
    print(f"Coursera data: {len(df)} records")
    return df


def generate_lms_data(mapping_df):
    """Generate LMS session logs."""
    records = []
    start_date = datetime(2024, 9, 1)

    for _, row in mapping_df.iterrows():
        n_sessions = np.random.randint(30, 120)
        for _ in range(n_sessions):
            session_start = start_date + timedelta(
                days=np.random.randint(0, 240),
                hours=np.random.randint(7, 22),
                minutes=np.random.randint(0, 60),
            )
            duration = timedelta(minutes=np.random.randint(5, 180))
            records.append({
                "student_id": row["lms_id"],
                "course_id": f"COURSE-{np.random.randint(1, NUM_COURSES + 1):03d}",
                "session_start": session_start.strftime("%Y-%m-%d %H:%M:%S"),
                "session_end": (session_start + duration).strftime("%Y-%m-%d %H:%M:%S"),
                "page_views": np.random.randint(1, 50),
                "downloads": np.random.randint(0, 10),
                "forum_posts": np.random.randint(0, 5),
            })

    df = pd.DataFrame(records)
    # Some missing values
    mask = np.random.random(len(df)) < 0.02
    df.loc[mask, "page_views"] = np.nan
    df.to_csv(OUTPUT_DIR / "lms_sessions.csv", index=False)
    print(f"LMS data: {len(df)} records")
    return df


def generate_academic_data(mapping_df):
    """Generate academic records."""
    records = []
    for _, row in mapping_df.iterrows():
        # Simulate a student performance profile
        base_gpa = np.random.normal(3.0, 0.6)
        base_gpa = np.clip(base_gpa, 1.0, 4.0)
        dropout = np.random.random() < 0.12  # 12% dropout rate

        for semester in SEMESTERS:
            n_courses = np.random.randint(3, 6)
            for _ in range(n_courses):
                grade_points = np.clip(base_gpa + np.random.normal(0, 0.3), 0, 4.0)
                letter_grades = {4.0: "A", 3.7: "A-", 3.3: "B+", 3.0: "B", 2.7: "B-", 2.3: "C+", 2.0: "C", 1.7: "C-", 1.0: "D", 0.0: "F"}
                closest = min(letter_grades.keys(), key=lambda x: abs(x - grade_points))
                records.append({
                    "student_id": row["academic_id"],
                    "course_id": f"COURSE-{np.random.randint(1, NUM_COURSES + 1):03d}",
                    "grade": letter_grades[closest],
                    "semester": semester,
                    "credits": np.random.choice([3, 4]),
                    "gpa": round(grade_points, 2),
                    "enrollment_status": "withdrawn" if dropout and semester == "2025-Spring" else "active",
                })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "academic_records.csv", index=False)
    print(f"Academic data: {len(df)} records")
    return df


if __name__ == "__main__":
    print("Generating sample data...")
    mapping = generate_student_mapping()
    generate_coursera_data(mapping)
    generate_lms_data(mapping)
    generate_academic_data(mapping)
    print(f"\nSample data generated in {OUTPUT_DIR}")
