"""
NeuralKarma — Database Layer
SQLAlchemy ORM models and database session management.
Uses SQLite for zero-config operation.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime, Boolean,
    ForeignKey, Index, create_engine, JSON,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Database file in project root
DB_PATH = Path(__file__).parent.parent / "neuralkarma.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    """A user who performs actions and accumulates karma."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200), default="")
    created_at = Column(DateTime, default=utcnow)
    total_actions = Column(Integer, default=0)
    aggregate_karma = Column(Float, default=50.0)

    actions = relationship("Action", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "total_actions": self.total_actions,
            "aggregate_karma": round(self.aggregate_karma, 2),
        }


class Action(Base):
    """An action (text input) that has been scored by the karma engine."""
    __tablename__ = "actions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utcnow, index=True)

    # 5-axis scores (0-100)
    prosociality_score = Column(Float, default=50.0)
    harm_avoidance_score = Column(Float, default=50.0)
    fairness_score = Column(Float, default=50.0)
    virtue_score = Column(Float, default=50.0)
    duty_score = Column(Float, default=50.0)

    # Aggregate
    aggregate_score = Column(Float, default=50.0, index=True)
    confidence = Column(Float, default=0.0)

    # Decay
    decayed_score = Column(Float, default=50.0)
    last_decay_update = Column(DateTime, default=utcnow)

    # Karma chain
    parent_action_id = Column(Integer, ForeignKey("actions.id"), nullable=True)
    chain_modifier = Column(Float, default=1.0)

    # Ripple
    ripple_total_impact = Column(Float, default=0.0)
    ripple_people_reached = Column(Integer, default=0)

    # Full score breakdown (JSON)
    raw_scores = Column(JSON, nullable=True)

    user = relationship("User", back_populates="actions")
    ripple_effects = relationship("RippleEffect", back_populates="source_action",
                                  cascade="all, delete-orphan", foreign_keys="RippleEffect.source_action_id")

    __table_args__ = (
        Index("idx_action_user_time", "user_id", "created_at"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "text": self.text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "axis_scores": {
                "prosociality": round(self.prosociality_score, 2),
                "harm_avoidance": round(self.harm_avoidance_score, 2),
                "fairness": round(self.fairness_score, 2),
                "virtue": round(self.virtue_score, 2),
                "duty": round(self.duty_score, 2),
            },
            "aggregate_score": round(self.aggregate_score, 2),
            "confidence": round(self.confidence, 4),
            "decayed_score": round(self.decayed_score, 2),
            "parent_action_id": self.parent_action_id,
            "chain_modifier": round(self.chain_modifier, 4),
            "ripple_total_impact": round(self.ripple_total_impact, 2),
            "ripple_people_reached": self.ripple_people_reached,
        }


class RippleEffect(Base):
    """Tracks ripple effect propagation from one action."""
    __tablename__ = "ripple_effects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_action_id = Column(Integer, ForeignKey("actions.id"), nullable=False, index=True)
    depth = Column(Integer, nullable=False)
    impact_per_person = Column(Float, default=0.0)
    people_affected = Column(Integer, default=0)
    depth_total_impact = Column(Float, default=0.0)
    cumulative_people = Column(Integer, default=0)
    cumulative_impact = Column(Float, default=0.0)

    source_action = relationship("Action", back_populates="ripple_effects",
                                 foreign_keys=[source_action_id])

    def to_dict(self):
        return {
            "id": self.id,
            "source_action_id": self.source_action_id,
            "depth": self.depth,
            "impact_per_person": round(self.impact_per_person, 2),
            "people_affected": self.people_affected,
            "depth_total_impact": round(self.depth_total_impact, 2),
            "cumulative_people": self.cumulative_people,
            "cumulative_impact": round(self.cumulative_impact, 2),
        }


class KarmaSnapshot(Base):
    """Periodic snapshot of a user's aggregate karma for history charting."""
    __tablename__ = "karma_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    aggregate_karma = Column(Float, default=50.0)
    total_actions = Column(Integer, default=0)
    snapshot_at = Column(DateTime, default=utcnow, index=True)

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "aggregate_karma": round(self.aggregate_karma, 2),
            "total_actions": self.total_actions,
            "snapshot_at": self.snapshot_at.isoformat() if self.snapshot_at else None,
        }


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    print("  [OK] Database initialized")


def get_db():
    """Get a database session (for FastAPI dependency injection)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
