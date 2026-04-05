"""Generate realistic sample datasets for the Learning Analytics Platform.

Student profiles drive all downstream data:
  - Each student has a latent 'engagement_trait' (0-1)
  - High engagement -> more activities, higher scores, more sessions, lower dropout
  - Low engagement -> fewer activities, lower scores, higher dropout risk
  - GPA correlates with engagement (r ~ 0.6)

This ensures the analytics pipeline finds meaningful, statistically significant
patterns — just as real educational data would.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent
NUM_STUDENTS = 200
NUM_COURSES = 10
SEMESTERS = ["2024-Fall", "2025-Spring"]


def _generate_student_profiles():
    """Create latent student profiles that drive all data generation.

    Returns:
        DataFrame with columns: student_id, engagement_trait, ability,
        dropout_risk, dropout
    """
    profiles = []
    for i in range(1, NUM_STUDENTS + 1):
        # Latent engagement trait: bimodal — most students engaged, some not
        if np.random.random() < 0.25:
            engagement = np.clip(np.random.beta(2, 5), 0.05, 0.95)   # low-engagement group
        else:
            engagement = np.clip(np.random.beta(5, 2), 0.05, 0.95)   # high-engagement group

        # Academic ability correlates with engagement (r ~ 0.5) but has its own variance
        ability = np.clip(0.5 * engagement + 0.5 * np.random.beta(3, 2), 0.05, 0.95)

        # Dropout risk: inversely related to engagement — low-engagement students
        # are 4-5x more likely to drop out
        dropout_prob = np.clip(0.40 - 0.45 * engagement + np.random.normal(0, 0.05), 0.02, 0.60)
        dropout = np.random.random() < dropout_prob

        profiles.append({
            "idx": i,
            "engagement": engagement,
            "ability": ability,
            "dropout_prob": dropout_prob,
            "dropout": dropout,
        })

    return pd.DataFrame(profiles)


def generate_student_mapping(profiles):
    """Generate student ID mapping table across systems."""
    records = []
    for _, p in profiles.iterrows():
        i = p["idx"]
        records.append({
            "student_id": f"STU-{i:04d}",
            "coursera_id": f"CR-{1000 + i}",
            "lms_id": f"LMS-{2000 + i}",
            "academic_id": f"AC-{3000 + i}",
        })
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "student_mapping.csv", index=False)
    return df


def generate_coursera_data(mapping_df, profiles):
    """Generate Coursera activity logs driven by student engagement profiles."""
    activities = ["video_watch", "quiz_attempt", "assignment_submit", "forum_post", "reading"]
    records = []
    start_date = datetime(2024, 9, 1)

    for (_, row), (_, prof) in zip(mapping_df.iterrows(), profiles.iterrows()):
        eng = prof["engagement"]
        abil = prof["ability"]

        # High engagement -> more activities (30-180), low -> fewer (10-60)
        n_activities = int(np.clip(eng * 160 + np.random.normal(0, 15), 10, 200))

        # Engagement affects activity distribution
        # High engagement: more quizzes and assignments; low: more passive reading/watching
        if eng > 0.5:
            probs = [0.30, 0.25, 0.20, 0.10, 0.15]
        else:
            probs = [0.40, 0.12, 0.08, 0.05, 0.35]

        for _ in range(n_activities):
            activity = np.random.choice(activities, p=probs)
            ts = start_date + timedelta(
                days=np.random.randint(0, 240),
                hours=np.random.randint(8, 23),
                minutes=np.random.randint(0, 60),
            )
            # Duration: engaged students spend more time
            base_dur = 15 + eng * 20
            duration = round(np.random.exponential(base_dur), 1) if activity != "forum_post" else round(np.random.exponential(5 + eng * 8), 1)

            record = {
                "student_id": row["coursera_id"],
                "course_id": f"COURSE-{np.random.randint(1, NUM_COURSES + 1):03d}",
                "timestamp": ts.strftime("%m/%d/%Y %I:%M %p"),
                "activity_type": activity,
                "duration_minutes": duration,
            }
            if activity == "video_watch":
                record["video_id"] = f"VID-{np.random.randint(1, 50):03d}"
                # High engagement -> higher completion (mean ~85%), low -> mean ~45%
                completion = np.clip(eng * 0.6 + np.random.beta(3, 2) * 0.4, 0.0, 1.0)
                record["completion_pct"] = round(completion, 2)
            elif activity == "quiz_attempt":
                # Score driven by ability (mean 55-90) with noise
                base_score = 55 + abil * 35
                record["score"] = round(np.clip(base_score + np.random.normal(0, 8), 0, 100), 1)
            records.append(record)

    df = pd.DataFrame(records)
    # Introduce some missing values (3%)
    mask = np.random.random(len(df)) < 0.03
    df.loc[mask, "duration_minutes"] = np.nan
    # Add some duplicates (2%)
    dupes = df.sample(n=int(len(df) * 0.02), random_state=42)
    df = pd.concat([df, dupes], ignore_index=True)
    df.to_csv(OUTPUT_DIR / "coursera_activity.csv", index=False)
    print(f"Coursera data: {len(df)} records")
    return df


def generate_lms_data(mapping_df, profiles):
    """Generate LMS session logs driven by student engagement profiles."""
    records = []
    start_date = datetime(2024, 9, 1)

    for (_, row), (_, prof) in zip(mapping_df.iterrows(), profiles.iterrows()):
        eng = prof["engagement"]

        # High engagement -> 60-150 sessions, low -> 15-50
        n_sessions = int(np.clip(eng * 120 + np.random.normal(20, 10), 15, 160))

        for _ in range(n_sessions):
            session_start = start_date + timedelta(
                days=np.random.randint(0, 240),
                hours=np.random.randint(7, 22),
                minutes=np.random.randint(0, 60),
            )
            # Session length: engaged students have longer sessions
            dur_minutes = int(np.clip(np.random.exponential(20 + eng * 60), 5, 240))
            duration = timedelta(minutes=dur_minutes)

            # Page views and forum posts scale with engagement
            page_views = int(np.clip(np.random.poisson(10 + eng * 30), 1, 80))
            forum_posts = int(np.clip(np.random.poisson(eng * 3), 0, 12))
            downloads = np.random.randint(0, int(3 + eng * 8))

            records.append({
                "student_id": row["lms_id"],
                "course_id": f"COURSE-{np.random.randint(1, NUM_COURSES + 1):03d}",
                "session_start": session_start.strftime("%Y-%m-%d %H:%M:%S"),
                "session_end": (session_start + duration).strftime("%Y-%m-%d %H:%M:%S"),
                "page_views": page_views,
                "downloads": downloads,
                "forum_posts": forum_posts,
            })

    df = pd.DataFrame(records)
    # Some missing values (2%)
    mask = np.random.random(len(df)) < 0.02
    df.loc[mask, "page_views"] = np.nan
    df.to_csv(OUTPUT_DIR / "lms_sessions.csv", index=False)
    print(f"LMS data: {len(df)} records")
    return df


def generate_academic_data(mapping_df, profiles):
    """Generate academic records driven by student ability and dropout profiles."""
    records = []
    for (_, row), (_, prof) in zip(mapping_df.iterrows(), profiles.iterrows()):
        abil = prof["ability"]
        is_dropout = prof["dropout"]

        # Base GPA driven by ability: high ability -> 3.2-4.0, low -> 1.5-2.8
        base_gpa = np.clip(1.5 + abil * 2.5 + np.random.normal(0, 0.2), 0.5, 4.0)

        for semester in SEMESTERS:
            # Dropout students show GPA decline in 2nd semester
            if is_dropout and semester == "2025-Spring":
                semester_gpa = np.clip(base_gpa - np.random.uniform(0.3, 0.8), 0.5, 4.0)
            else:
                semester_gpa = np.clip(base_gpa + np.random.normal(0, 0.15), 0.5, 4.0)

            n_courses = np.random.randint(3, 6)
            for _ in range(n_courses):
                grade_points = np.clip(semester_gpa + np.random.normal(0, 0.25), 0, 4.0)
                letter_grades = {
                    4.0: "A", 3.7: "A-", 3.3: "B+", 3.0: "B", 2.7: "B-",
                    2.3: "C+", 2.0: "C", 1.7: "C-", 1.0: "D", 0.0: "F",
                }
                closest = min(letter_grades.keys(), key=lambda x: abs(x - grade_points))
                records.append({
                    "student_id": row["academic_id"],
                    "course_id": f"COURSE-{np.random.randint(1, NUM_COURSES + 1):03d}",
                    "grade": letter_grades[closest],
                    "semester": semester,
                    "credits": np.random.choice([3, 4]),
                    "gpa": round(grade_points, 2),
                    "enrollment_status": "withdrawn" if is_dropout and semester == "2025-Spring" else "active",
                })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "academic_records.csv", index=False)
    print(f"Academic data: {len(df)} records")
    return df


if __name__ == "__main__":
    print("Generating sample data with realistic correlations...")
    print("  - Engagement drives activity volume, session length, quiz scores")
    print("  - Low engagement -> 4-5x higher dropout risk")
    print("  - Ability correlates with engagement (r ~ 0.5)")
    print("  - Dropout students show GPA decline in 2nd semester")
    print()

    profiles = _generate_student_profiles()

    # Print profile stats
    print(f"Student profiles (n={len(profiles)}):")
    print(f"  Engagement:  mean={profiles['engagement'].mean():.2f}, std={profiles['engagement'].std():.2f}")
    print(f"  Ability:     mean={profiles['ability'].mean():.2f}, std={profiles['ability'].std():.2f}")
    print(f"  Dropout:     {profiles['dropout'].sum()} students ({profiles['dropout'].mean()*100:.1f}%)")
    print()

    mapping = generate_student_mapping(profiles)
    generate_coursera_data(mapping, profiles)
    generate_lms_data(mapping, profiles)
    generate_academic_data(mapping, profiles)
    print(f"\nSample data generated in {OUTPUT_DIR}")
