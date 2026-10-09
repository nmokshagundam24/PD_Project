from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import uuid

from database import get_db
from models import Participant

router = APIRouter()


@router.post("/start-session")
def start_session(db: Session = Depends(get_db)):
    """Generate a new anonymous participant and return the UUID."""
    participant_id = str(uuid.uuid4())

    participant = Participant(participant_id=participant_id)
    db.add(participant)
    db.commit()

    return {
        "participant_id": participant_id,
        "status": "session_started",
    }
