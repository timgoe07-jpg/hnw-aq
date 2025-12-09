from pathlib import Path
from typing import Generator
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = f"sqlite:///{BASE_DIR / 'app.db'}"

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Investor(Base):
    __tablename__ = "investors"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    aum = Column(Float, nullable=False)  # assets under management
    risk_tolerance = Column(String, nullable=False)  # low/medium/high
    engagement_score = Column(Float, nullable=False)  # 0-100
    email_open_rate = Column(Float, nullable=False)  # 0-1
    call_frequency = Column(Float, nullable=False)  # calls per quarter
    inactivity_days = Column(Integer, nullable=False, default=0)
    redemption_intent = Column(Boolean, nullable=False, default=False)
    distribution_yield = Column(Float, nullable=False, default=0.0)
    meetings_last_quarter = Column(Integer, nullable=False, default=0)
    churn_label = Column(Integer)  # optional historical label
    churn_risk_score = Column(Float)  # latest predicted probability (legacy single-horizon)
    churn_risk_score_3m = Column(Float)
    churn_risk_score_6m = Column(Float)
    churn_risk_score_12m = Column(Float)
    churn_risk_bucket_3m = Column(String)
    churn_risk_bucket_6m = Column(String)
    churn_risk_bucket_12m = Column(String)

    loans = relationship("Loan", back_populates="investor", cascade="all, delete-orphan")


class Loan(Base):
    __tablename__ = "loans"

    id = Column(Integer, primary_key=True)
    investor_id = Column(Integer, ForeignKey("investors.id"), nullable=False)
    amount = Column(Float, nullable=False)
    ltv_ratio = Column(Float, nullable=False)  # loan-to-value
    term_months = Column(Integer, nullable=False)
    sector = Column(String, nullable=False)
    arrears_flag = Column(Boolean, nullable=False, default=False)
    dscr = Column(Float, nullable=False, default=1.0)  # debt service coverage ratio
    covenants_flag = Column(Boolean, nullable=False, default=False)
    collateral_score = Column(Float, nullable=False, default=0.5)
    default_label = Column(Integer)  # optional historical label
    default_risk_score = Column(Float)  # latest predicted probability (legacy single-horizon)
    default_risk_score_3m = Column(Float)
    default_risk_score_6m = Column(Float)
    default_risk_score_12m = Column(Float)
    default_risk_bucket_3m = Column(String)
    default_risk_bucket_6m = Column(String)
    default_risk_bucket_12m = Column(String)

    investor = relationship("Investor", back_populates="loans")


class DailySnapshot(Base):
    __tablename__ = "daily_snapshots"

    id = Column(Integer, primary_key=True)
    snapshot_date = Column(String, unique=True, nullable=False)
    avg_churn_risk_3m = Column(Float)
    avg_churn_risk_6m = Column(Float)
    avg_churn_risk_12m = Column(Float)
    avg_default_risk_3m = Column(Float)
    avg_default_risk_6m = Column(Float)
    avg_default_risk_12m = Column(Float)
    high_risk_investor_count_6m = Column(Integer)
    high_risk_loan_count_6m = Column(Integer)
    high_risk_exposure_amount_6m = Column(Float)


class Intervention(Base):
    __tablename__ = "interventions"

    id = Column(Integer, primary_key=True)
    entity_type = Column(String, nullable=False)  # investor or loan
    entity_id = Column(Integer, nullable=False)
    action_type = Column(String, nullable=False)
    expected_effect = Column(String, nullable=True)
    engagement_delta = Column(Float, nullable=True)
    inactivity_delta = Column(Float, nullable=True)
    ltv_delta = Column(Float, nullable=True)
    dscr_delta = Column(Float, nullable=True)
    created_at = Column(String, nullable=False, default=lambda: datetime.now(timezone.utc).isoformat())


class RiskAlert(Base):
    __tablename__ = "risk_alerts"

    id = Column(Integer, primary_key=True)
    entity_type = Column(String, nullable=False)  # investor or loan
    entity_id = Column(Integer, nullable=False)
    first_high_at = Column(String, nullable=False)  # ISO timestamp when first marked high
    last_seen_at = Column(String, nullable=False)  # ISO timestamp of last refresh check
    last_bucket = Column(String, nullable=True)
    last_score = Column(Float, nullable=True)
    sla_days = Column(Integer, nullable=False, default=5)
    resolved_at = Column(String, nullable=True)

    def mark_seen(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.last_seen_at = now
        if not self.first_high_at:
            self.first_high_at = now


def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_session() -> Generator:
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
