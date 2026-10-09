import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database (hosted Postgres gives "postgresql://..." - SQLAlchemy needs the driver name added)
_raw_db = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:password@localhost:5432/parkinson_db")
if _raw_db.startswith("postgres://"):
    _raw_db = _raw_db.replace("postgres://", "postgresql+psycopg2://", 1)
elif _raw_db.startswith("postgresql://"):
    _raw_db = _raw_db.replace("postgresql://", "postgresql+psycopg2://", 1)
DATABASE_URL = _raw_db

# Audio is only used as a temporary file and deleted right after processing
AUDIO_STORAGE_DIR = os.getenv("AUDIO_STORAGE_DIR", os.path.join(BASE_DIR, "audio_storage"))

# Folder that CONTAINS the english_model and telugu_models folders
MODEL_ROOT = os.getenv("MODEL_ROOT", BASE_DIR)
NUM_FOLDS = int(os.getenv("NUM_FOLDS", "5"))

# App Settings - comma-separated list of website addresses allowed to call this API
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",") if o.strip()]
ALLOWED_TASKS = ["vowels", "numbers", "ddk", "words", "sentences", "spontaneous"]
ALLOWED_LANGUAGES = ["english", "telugu"]
TOTAL_TASKS = 6
TARGET_SAMPLE_RATE = 16000
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_AUDIO_SECONDS = 75
