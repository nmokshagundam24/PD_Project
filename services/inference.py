import os
import joblib
import numpy as np
import pandas as pd
from config import NUM_FOLDS, MODEL_ROOT

MODEL_DIRS = {
    "english": os.path.join(MODEL_ROOT, "english_model"),
    "telugu": os.path.join(MODEL_ROOT, "telugu_models"),
}

# --- 49 Features for English Model ---
FEATURE_COLS_ENGLISH = [f"mfcc{i}" for i in range(1, 40)] + [
    "jitter", "shimmer", "pitch", "formant1", "formant2", "formant3",
    "spectral_centroid", "spectral_bandwidth", "spectral_flatness", "spectral_rolloff"
]

# --- 104 Features + Task for Telugu Model ---
FEATURE_COLS_TELUGU = ["Task"]
for i in range(1, 14):
    FEATURE_COLS_TELUGU.extend([
        f"mfcc{i}_mean", f"mfcc{i}_std",
        f"mfcc{i}_delta_mean", f"mfcc{i}_delta_std",
        f"mfcc{i}_delta2_mean", f"mfcc{i}_delta2_std"
    ])
FEATURE_COLS_TELUGU.extend([
    "jitter", "shimmer", "hnr",
    "pitch_mean", "pitch_std", "pitch_median", "pitch_min", "pitch_max", "pitch_range",
    "formant1", "formant2", "formant3",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_flatness_mean", "spectral_flatness_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "zcr_mean", "zcr_std",
    "rms_mean", "rms_std",
    "duration"
])

_models = {"english": [], "telugu": []}

def load_models():
    global _models
    for lang, m_dir in MODEL_DIRS.items():
        _models[lang] = []
        for i in range(1, NUM_FOLDS + 1):
            m_path = os.path.join(m_dir, f"xgb_fold{i}.joblib")
            if os.path.exists(m_path):
                _models[lang].append(joblib.load(m_path))
                print(f"  Loaded {lang} model: xgb_fold{i}.joblib")
            else:
                print(f"  Warning: {lang} model not found at {m_path}")
        if not _models[lang]:
            raise RuntimeError(f"No {lang} models found in {m_dir}")
        print(f"  {lang.capitalize()} inference ready: {len(_models[lang])} fold models loaded")

def predict(features_dict: dict, language: str) -> tuple[int, float]:
    language = language.lower()
    if language not in _models or not _models[language]:
        raise RuntimeError(f"No models loaded for language: {language}")

    if language == "english":
        # Filter to exact 49 columns, removing "Task"
        df_feat = pd.DataFrame([features_dict])
        X = df_feat.reindex(columns=FEATURE_COLS_ENGLISH).fillna(0)
        # English XGBoost expects a pure numerical array
        X_input = X.values 
    else:
        # Provide 104 columns including the "Task" string
        if "Task" not in features_dict:
            features_dict["Task"] = "vowels"
        df_feat = pd.DataFrame([features_dict])
        X = df_feat.reindex(columns=FEATURE_COLS_TELUGU)
        # Telugu XGBoost has a ColumnTransformer that requires a DataFrame
        X_input = X 

    all_probs = [m.predict_proba(X_input)[:, 1] for m in _models[language]]
    avg_prob = float(np.mean(all_probs, axis=0)[0])

    label = 1 if avg_prob >= 0.5 else 0
    # Confidence = how sure the model is about the label it chose (always 0.5 - 1.0)
    confidence = avg_prob if label == 1 else 1.0 - avg_prob
    return label, confidence
