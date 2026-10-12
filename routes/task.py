import os
import threading
import time

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import Recording, Participant
from config import ALLOWED_TASKS, ALLOWED_LANGUAGES, MAX_UPLOAD_BYTES, MAX_AUDIO_SECONDS
from services.audio_storage import save_audio, load_audio
from services.feature_extraction import extract_features_from_audio
from services.inference import predict

router = APIRouter()

# Only one recording is analysed at a time. Free hosting has very little memory,
# so extra requests simply wait in line instead of crashing the server.
_gate = threading.Semaphore(1)


def _result_payload(participant_id, task_id, task_name, language, label, confidence):
    return {
        "status": "success",
        "participant_id": participant_id,
        "task_id": task_id,
        "task_name": task_name,
        "language_model_used": language,
        "label": label,
        "confidence": round(confidence, 4) if confidence is not None else None,
    }


# NOTE: plain "def" (not "async def") so FastAPI runs this heavy work in a worker thread
@router.post("/process-task")
def process_task(
    participant_id: str = Form(...),
    task_id: int = Form(...),
    task_name: str = Form(...),
    language: str = Form("english"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    request_started = time.perf_counter()
    task_name_lower = task_name.lower()
    if task_name_lower not in ALLOWED_TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task_name. Must be one of: {ALLOWED_TASKS}")

    language = language.lower().strip()
    if language not in ALLOWED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Invalid language. Must be one of: {ALLOWED_LANGUAGES}")

    if not 1 <= task_id <= 6:
        raise HTTPException(status_code=400, detail="task_id must be between 1 and 6")

    db_start = time.perf_counter()
    participant = db.query(Participant).filter(Participant.participant_id == participant_id).first()
    print(
        f"[timing] Participant DB lookup: "
        f"{time.perf_counter() - db_start:.3f}s"
    )
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found. Call /start-session first.")

    # If this task was already saved (e.g. the reply got lost and the user pressed Retry),
    # return the saved result instead of an error.
    existing = (
        db.query(Recording)
        .filter(Recording.participant_id == participant_id, Recording.task_id == task_id)
        .first()
    )
    if existing:
        return _result_payload(participant_id, task_id, existing.task_name, language,
                               existing.label, existing.confidence)

    audio_path = None
    try:
        read_start = time.perf_counter()
        file_content = file.file.read()
        print(
            f"[timing] Read uploaded file: "
            f"{time.perf_counter() - read_start:.3f}s"
        )
        
        if len(file_content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio file is too large.")
        print(f"[process-task] task={task_id}/{task_name_lower} lang={language} size={len(file_content)} bytes")

        # with _gate:
        #     audio_path = save_audio(participant_id, task_id, task_name_lower, file_content)
        #     y, sr = load_audio(audio_path)
        #     y = y[: sr * MAX_AUDIO_SECONDS]  # safety cap (tasks are at most 60 s)
        #     print(f"[process-task] Audio loaded: {len(y)} samples, sr={sr}, duration={len(y)/sr:.2f}s")

        #     features = extract_features_from_audio(y, sr)
        #     if features is None:
        #         raise HTTPException(status_code=422, detail="Audio too short or feature extraction failed.")

        #     features["Task"] = task_name_lower  # for the Telugu model's OneHotEncoder
        #     label, confidence = predict(features, language)

        gate_start = time.perf_counter()

        with _gate:
            print(
                f"[timing] Semaphore wait: "
                f"{time.perf_counter() - gate_start:.3f}s"
            )

            stage_start = time.perf_counter()
            audio_path = save_audio(
                participant_id, task_id, task_name_lower, file_content
            )
            print(
                f"[timing] Save temporary audio: "
                f"{time.perf_counter() - stage_start:.3f}s"
            )

            stage_start = time.perf_counter()
            y, sr = load_audio(audio_path)
            y = y[: sr * MAX_AUDIO_SECONDS]

            print(
                f"[timing] Load/decode audio: "
                f"{time.perf_counter() - stage_start:.3f}s"
            )
            print(
                f"[process-task] Audio loaded: {len(y)} samples, "
                f"sr={sr}, duration={len(y)/sr:.2f}s"
            )

            stage_start = time.perf_counter()
            features = extract_features_from_audio(y, sr)

            print(
                f"[timing] Feature extraction call: "
                f"{time.perf_counter() - stage_start:.3f}s"
            )

            if features is None:
                raise HTTPException(
                    status_code=422,
                    detail="Audio too short or feature extraction failed."
                )

            features["Task"] = task_name_lower

            stage_start = time.perf_counter()
            label, confidence = predict(features, language)

            print(
                f"[timing] Model prediction: "
                f"{time.perf_counter() - stage_start:.3f}s"
            )
        print(f"[process-task] Prediction: label={label}, confidence={confidence:.4f} (model: {language})")

        recording = Recording(
            participant_id=participant_id,
            task_id=task_id,
            task_name=task_name_lower,
            audio_file_path="not_stored",
            label=label,
            confidence=confidence,
        )

        db_start = time.perf_counter()
        
        db.add(recording)
        db.commit()
        print(
            f"[timing] Save recording to DB: "
            f"{time.perf_counter() - db_start:.3f}s"
        )

        return _result_payload(participant_id, task_id, task_name_lower, language, label, confidence)

    except HTTPException:
        raise
    except ValueError as e:
        # Audio could not be decoded (corrupt / unsupported file)
        db.rollback()
        raise HTTPException(status_code=422, detail="Could not read this audio. Please record again.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Privacy: the recording is never kept
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        print(
            f"[timing] TOTAL endpoint time: "
            f"{time.perf_counter() - request_started:.3f}s"
        )
