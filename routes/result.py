from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import Recording, Participant
from config import TOTAL_TASKS
from services.majority_vote import compute_majority_vote

router = APIRouter()


@router.get("/final-result/{participant_id}")
def get_final_result(participant_id: str, db: Session = Depends(get_db)):
    """
    Compute and return the final prediction using majority voting.
    Requires all 6 tasks to be completed.
    """
    # Check participant exists
    participant = db.query(Participant).filter(Participant.participant_id == participant_id).first()
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found.")

    # Get all recordings for this participant
    recordings = (
        db.query(Recording)
        .filter(Recording.participant_id == participant_id)
        .order_by(Recording.task_id)
        .all()
    )

    completed = len(recordings)

    if completed < TOTAL_TASKS:
        return {
            "status": "pending",
            "completed_tasks": completed,
            "total_tasks": TOTAL_TASKS,
            "message": f"Only {completed}/{TOTAL_TASKS} tasks completed. Complete all tasks first.",
        }

    # Compute majority vote
    labels = [r.label for r in recordings]
    final_label, breakdown = compute_majority_vote(labels)

    # Compute average confidence across tasks
    # Average confidence of the tasks that agree with the final label
    confidences = [r.confidence for r in recordings if r.label == final_label and r.confidence is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Store final prediction in participant record
    participant.final_prediction = final_label
    participant.final_confidence = avg_confidence
    db.commit()

    # Build task-level breakdown
    task_results = [
        {
            "task_id": r.task_id,
            "task_name": r.task_name,
            "label": r.label,
            "label_text": "Parkinson's" if r.label == 1 else "Healthy",
            "confidence": round(r.confidence, 4) if r.confidence else None,
        }
        for r in recordings
    ]

    return {
        "status": "complete",
        "participant_id": participant_id,
        "final_result": "Parkinson's Indicators Detected" if final_label == 1 else "No Parkinson's Indicators Detected",
        "final_label": final_label,
        "final_confidence": round(avg_confidence, 4),
        "majority_vote": breakdown,
        "task_results": task_results,
    }
