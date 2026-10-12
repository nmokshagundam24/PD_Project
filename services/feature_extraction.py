import warnings
import time
import numpy as np
import librosa
import parselmouth

warnings.filterwarnings("ignore")

def extract_features_from_audio(y: np.ndarray, sr: int) -> dict | None:
    total_start = time.perf_counter()
    stage_start = total_start
    
    try:
        y, _ = librosa.effects.trim(y)
        duration = float(librosa.get_duration(y=y, sr=sr))
        if duration < 0.1: return None

        res = {}

        # 1. Base MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        if mfcc.shape[1] >= 9:
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
        else:
            delta = np.zeros_like(mfcc)
            delta2 = np.zeros_like(mfcc)

        # Build 49-feature MFCCs (Means of concatenated arrays)
        mfcc_feats = np.concatenate([mfcc, delta, delta2], axis=0).mean(axis=1)
        for i, v in enumerate(mfcc_feats):
            res[f"mfcc{i + 1}"] = float(v)
        print(f"[timing] MFCC extraction and statistics: " f"{time.perf_counter() - stage_start:.3f}s")
        stage_start = time.perf_counter()

        # Build 104-feature MFCCs (Explicit mean/std stats)
        for i in range(13):
            idx = i + 1
            res[f"mfcc{idx}_mean"] = float(np.mean(mfcc[i]))
            res[f"mfcc{idx}_std"] = float(np.std(mfcc[i]))
            res[f"mfcc{idx}_delta_mean"] = float(np.mean(delta[i]))
            res[f"mfcc{idx}_delta_std"] = float(np.std(delta[i]))
            res[f"mfcc{idx}_delta2_mean"] = float(np.mean(delta2[i]))
            res[f"mfcc{idx}_delta2_std"] = float(np.std(delta2[i]))

        snd = parselmouth.Sound(y, sampling_frequency=sr)

        # Jitter, Shimmer, HNR
        try:
            pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitter = float(parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
            shimmer = float(parselmouth.praat.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        except Exception:
            jitter = shimmer = np.nan
            
        try:
            hnr = float(parselmouth.praat.call(snd, "Get harmonics-to-noise ratio", 0, 0))
        except Exception:
            hnr = np.nan

        res["jitter"] = jitter
        res["shimmer"] = shimmer
        res["hnr"] = hnr
        print(
            f"[timing] Jitter, shimmer, HNR: "
            f"{time.perf_counter() - stage_start:.3f}s"
        )
        stage_start = time.perf_counter()

        # Pitch (Midpoint for 49-feature, Stats for 104-feature)
        try:
            pitch_obj = snd.to_pitch()
            res["pitch"] = float(pitch_obj.get_value_at_time(duration / 2)) # 49-feature pitch
            
            pitch_values = pitch_obj.selected_array['frequency']
            voiced = pitch_values[pitch_values > 0]
            if len(voiced) > 0:
                res["pitch_mean"] = float(np.mean(voiced))
                res["pitch_std"] = float(np.std(voiced))
                res["pitch_median"] = float(np.median(voiced))
                res["pitch_min"] = float(np.min(voiced))
                res["pitch_max"] = float(np.max(voiced))
                res["pitch_range"] = float(np.max(voiced) - np.min(voiced))
            else:
                res["pitch_mean"] = res["pitch_std"] = res["pitch_median"] = res["pitch_min"] = res["pitch_max"] = res["pitch_range"] = np.nan
        except Exception:
            res["pitch"] = res["pitch_mean"] = res["pitch_std"] = res["pitch_median"] = res["pitch_min"] = res["pitch_max"] = res["pitch_range"] = np.nan

        print(
            f"[timing] Pitch extraction: "
            f"{time.perf_counter() - stage_start:.3f}s"
        )
        stage_start = time.perf_counter()

        # Formants
        try:
            formants = snd.to_formant_burg()
            res["formant1"] = float(formants.get_value_at_time(1, duration / 2))
            res["formant2"] = float(formants.get_value_at_time(2, duration / 2))
            res["formant3"] = float(formants.get_value_at_time(3, duration / 2))
        except Exception:
            res["formant1"] = res["formant2"] = res["formant3"] = np.nan
                print(
            f"[timing] Formant extraction: "
            f"{time.perf_counter() - stage_start:.3f}s"
        )
        stage_start = time.perf_counter()

        
        # Spectral
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        flat = librosa.feature.spectral_flatness(y=y)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        rms = librosa.feature.rms(y=y)

        # 49-feature spectral
        res["spectral_centroid"] = float(np.mean(cent))
        res["spectral_bandwidth"] = float(np.mean(bw))
        res["spectral_flatness"] = float(np.mean(flat))
        res["spectral_rolloff"] = float(np.mean(rolloff))

        # 104-feature spectral
        res["spectral_centroid_mean"] = float(np.mean(cent))
        res["spectral_centroid_std"] = float(np.std(cent))
        res["spectral_bandwidth_mean"] = float(np.mean(bw))
        res["spectral_bandwidth_std"] = float(np.std(bw))
        res["spectral_flatness_mean"] = float(np.mean(flat))
        res["spectral_flatness_std"] = float(np.std(flat))
        res["spectral_rolloff_mean"] = float(np.mean(rolloff))
        res["spectral_rolloff_std"] = float(np.std(rolloff))
        res["zcr_mean"] = float(np.mean(zcr))
        res["zcr_std"] = float(np.std(zcr))
        res["rms_mean"] = float(np.mean(rms))
        res["rms_std"] = float(np.std(rms))
        res["duration"] = duration

        print(
            f"[timing] Spectral feature extraction: "
            f"{time.perf_counter() - stage_start:.3f}s"
        )
        print(
            f"[timing] TOTAL feature extraction: "
            f"{time.perf_counter() - total_start:.3f}s"
        )

        return res
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None
