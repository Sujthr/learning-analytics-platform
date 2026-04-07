"""Database engine, session, and table creation."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from core.config import settings


engine = create_engine(settings.db.url, echo=settings.db.echo)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency for DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables."""
    from core.models import (  # noqa: F401
        Learner, Course, Program, Specialization,
        CourseActivity, VideoActivity, ProgramActivity,
        SpecializationActivity, EngagementScore, SkillProfile,
    )
    Base.metadata.create_all(bind=engine)
