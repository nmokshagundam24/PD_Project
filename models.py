from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from database import Base


class Participant(Base):
    __tablename__ = "participants"

    id = Column(Integer, primary_key=True, index=True)
    participant_id = Column(String, unique=True, index=True, nullable=False)
    final_prediction = Column(Integer, nullable=True)  # 0=Healthy, 1=Parkinson's, null=pending
    final_confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    recordings = relationship("Recording", back_populates="participant")


class Recording(Base):
    __tablename__ = "recordings"

    id = Column(Integer, primary_key=True, index=True)
    participant_id = Column(String, ForeignKey("participants.participant_id"), index=True, nullable=False)
    task_id = Column(Integer, nullable=False)  # 1 to 6
    task_name = Column(String, nullable=False)  # vowels, numbers, ddk, words, sentences, spontaneous
    audio_file_path = Column(String, nullable=False)
    label = Column(Integer, nullable=True)  # 0=Healthy, 1=Parkinson's
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    participant = relationship("Participant", back_populates="recordings")

    __table_args__ = (
        UniqueConstraint("participant_id", "task_id", name="uq_participant_task"),
    )
