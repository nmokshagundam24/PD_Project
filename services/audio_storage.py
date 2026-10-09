import os
import io
import traceback
import librosa
import soundfile as sf
import numpy as np
from config import AUDIO_STORAGE_DIR, TARGET_SAMPLE_RATE


def ensure_storage_dir(participant_id: str) -> str:
    """Create participant directory if it doesn't exist."""
    participant_dir = os.path.join(AUDIO_STORAGE_DIR, participant_id)
    os.makedirs(participant_dir, exist_ok=True)
    return participant_dir


def _load_audio_bytes(file_content: bytes, sr: int) -> np.ndarray:
    """
    Try multiple strategies to decode uploaded audio bytes.
    Returns the mono audio numpy array at the requested sample rate.
    """
    header = file_content[:12]
    print(f"  [audio_storage] Received {len(file_content)} bytes, header={header}")

    # Strategy 1: soundfile (handles WAV, FLAC, OGG)
    try:
        y, orig_sr = sf.read(io.BytesIO(file_content), dtype="float32", always_2d=False)
        print(f"  [audio_storage] soundfile decoded: sr={orig_sr}, samples={len(y)}")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        return y
    except Exception as e:
        print(f"  [audio_storage] soundfile failed: {e}")

    # Strategy 2: librosa (may use audioread / ffmpeg backend)
    try:
        y, _ = librosa.load(io.BytesIO(file_content), sr=sr, mono=True)
        print(f"  [audio_storage] librosa decoded: samples={len(y)}")
        return y
    except Exception as e:
        print(f"  [audio_storage] librosa failed: {e}")

    # Strategy 3: Save raw bytes to a temp file and try loading from disk
    #             (some decoders work better with file paths)
    import tempfile
    for ext in (".wav", ".webm", ".ogg"):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            y, _ = librosa.load(tmp_path, sr=sr, mono=True)
            print(f"  [audio_storage] librosa file-based decode ({ext}): samples={len(y)}")
            return y
        except Exception as e:
            print(f"  [audio_storage] file-based decode ({ext}) failed: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    raise ValueError(
        f"Cannot decode audio. Received {len(file_content)} bytes with header {header!r}. "
        "Ensure the browser sends WAV (PCM) format or install ffmpeg for WebM support."
    )


def save_audio(participant_id: str, task_id: int, task_name: str, file_content: bytes) -> str:
    """
    Save uploaded audio to filesystem as a normalized .wav file.

    - Decodes audio bytes (WAV, WebM, OGG, etc.)
    - Resamples to 16 kHz mono
    - Saves as .wav
    - Returns the saved file path
    """
    participant_dir = ensure_storage_dir(participant_id)
    filename = f"task_{task_id}_{task_name}.wav"
    file_path = os.path.join(participant_dir, filename)

    print(f"  [audio_storage] save_audio: participant={participant_id}, task={task_id}_{task_name}")

    y = _load_audio_bytes(file_content, TARGET_SAMPLE_RATE)

    # Trim silence from edges
    y, _ = librosa.effects.trim(y)
    print(f"  [audio_storage] Trimmed audio: {len(y)} samples, {len(y)/TARGET_SAMPLE_RATE:.2f}s")

    # Save as WAV
    sf.write(file_path, y, TARGET_SAMPLE_RATE)
    print(f"  [audio_storage] Saved to {file_path}")

    return file_path


def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """Load a stored .wav file and return (audio_array, sample_rate)."""
    y, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE, mono=True)
    return y, sr
