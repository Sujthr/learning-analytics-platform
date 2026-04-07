"""SQLAlchemy ORM models — PostgreSQL-compatible schema."""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text,
    ForeignKey, Index, JSON,
)
from sqlalchemy.orm import relationship
from core.database import Base


# ── Dimension Tables ────────────────────────────────────────────────

class Learner(Base):
    __tablename__ = "learners"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    external_id = Column(String(100), unique=True, index=True)
    name = Column(String(255))
    business_unit = Column(String(100))
    role = Column(String(100))
    location = Column(String(100))
    enrollment_date = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    course_activities = relationship("CourseActivity", back_populates="learner")
    video_activities = relationship("VideoActivity", back_populates="learner")
    engagement_scores = relationship("EngagementScore", back_populates="learner")
    skill_profiles = relationship("SkillProfile", back_populates="learner")


class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(String(100), unique=True, nullable=False, index=True)
    course_name = Column(String(500))
    course_slug = Column(String(500))
    institution = Column(String(255))
    category = Column(String(100))
    estimated_hours = Column(Float)
    skills = Column(JSON)  # list of skill tags
    created_at = Column(DateTime, default=datetime.utcnow)

    activities = relationship("CourseActivity", back_populates="course")


class Program(Base):
    __tablename__ = "programs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    program_id = Column(String(100), unique=True, nullable=False, index=True)
    program_name = Column(String(500))
    program_slug = Column(String(500))
    total_courses = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class Specialization(Base):
    __tablename__ = "specializations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    spec_id = Column(String(100), unique=True, nullable=False, index=True)
    spec_name = Column(String(500))
    spec_slug = Column(String(500))
    total_courses = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Fact Tables ─────────────────────────────────────────────────────

class CourseActivity(Base):
    __tablename__ = "course_activities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    learner_id = Column(Integer, ForeignKey("learners.id"), index=True)
    course_id_ref = Column(String(100), ForeignKey("courses.course_id"), index=True)
    course_name = Column(String(500))
    enrollment_ts = Column(DateTime)
    completion_ts = Column(DateTime)
    progress_pct = Column(Float, default=0)
    grade = Column(Float)
    learning_hours = Column(Float, default=0)
    is_completed = Column(Boolean, default=False)
    last_activity_ts = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    learner = relationship("Learner", back_populates="course_activities")
    course = relationship("Course", back_populates="activities")

    __table_args__ = (
        Index("ix_course_activity_learner_course", "learner_id", "course_id_ref"),
    )


class VideoActivity(Base):
    __tablename__ = "video_activities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    learner_id = Column(Integer, ForeignKey("learners.id"), index=True)
    course_id_ref = Column(String(100), index=True)
    video_name = Column(String(500))
    video_id = Column(String(100))
    watch_seconds = Column(Float, default=0)
    total_seconds = Column(Float, default=0)
    completion_pct = Column(Float, default=0)
    watch_count = Column(Integer, default=1)
    last_watch_ts = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    learner = relationship("Learner", back_populates="video_activities")


class ProgramActivity(Base):
    __tablename__ = "program_activities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    learner_id = Column(Integer, ForeignKey("learners.id"), index=True)
    program_id = Column(String(100), index=True)
    program_name = Column(String(500))
    progress_pct = Column(Float, default=0)
    courses_completed = Column(Integer, default=0)
    total_courses = Column(Integer, default=0)
    learning_hours = Column(Float, default=0)
    is_completed = Column(Boolean, default=False)
    enrollment_ts = Column(DateTime)
    last_activity_ts = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


class SpecializationActivity(Base):
    __tablename__ = "specialization_activities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    learner_id = Column(Integer, ForeignKey("learners.id"), index=True)
    spec_id = Column(String(100), index=True)
    spec_name = Column(String(500))
    progress_pct = Column(Float, default=0)
    courses_completed = Column(Integer, default=0)
    total_courses = Column(Integer, default=0)
    learning_hours = Column(Float, default=0)
    is_completed = Column(Boolean, default=False)
    enrollment_ts = Column(DateTime)
    last_activity_ts = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Derived / Analytics Tables ──────────────────────────────────────

class EngagementScore(Base):
    __tablename__ = "engagement_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    learner_id = Column(Integer, ForeignKey("learners.id"), index=True)
    score = Column(Float)  # 0-100
    category = Column(String(20))  # High / Medium / Low / At-Risk
    progress_component = Column(Float)
    hours_component = Column(Float)
    video_component = Column(Float)
    recency_component = Column(Float)
    computed_at = Column(DateTime, default=datetime.utcnow)

    learner = relationship("Learner", back_populates="engagement_scores")


class SkillProfile(Base):
    __tablename__ = "skill_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    learner_id = Column(Integer, ForeignKey("learners.id"), index=True)
    skill_name = Column(String(200))
    skill_category = Column(String(100))
    skill_score = Column(Float)  # 0-100
    courses_contributing = Column(Integer)
    acquired_at = Column(DateTime)
    computed_at = Column(DateTime, default=datetime.utcnow)

    learner = relationship("Learner", back_populates="skill_profiles")


class Insight(Base):
    __tablename__ = "insights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(50))  # learner, course, program, engagement, skill
    severity = Column(String(20))  # info, warning, critical
    title = Column(String(500))
    description = Column(Text)
    metric_value = Column(Float)
    metadata_ = Column("metadata", JSON)
    generated_at = Column(DateTime, default=datetime.utcnow)
