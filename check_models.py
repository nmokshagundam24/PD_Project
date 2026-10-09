"""Run:  python check_models.py     (from inside the backend folder)
Checks that all 10 model files load and shows what each one expects."""
import glob, os
import joblib
from sklearn.pipeline import Pipeline

here = os.path.dirname(os.path.abspath(__file__))
ok = True
for folder in ("english_model", "telugu_models"):
    files = sorted(glob.glob(os.path.join(here, folder, "xgb_fold*.joblib")))
    print(f"\n== {folder}: found {len(files)} files (need 5)")
    if len(files) != 5:
        ok = False
    for f in files:
        try:
            m = joblib.load(f)
        except Exception as e:
            ok = False
            print("  FAILED to load", os.path.basename(f), "->", e)
            continue
        print(" ", os.path.basename(f), type(m).__name__, "features:", getattr(m, "n_features_in_", "?"))
        if isinstance(m, Pipeline):
            for name, step in m.steps:
                for tn, t, cols in getattr(step, "transformers_", []):
                    if hasattr(t, "categories_"):
                        print("     Task categories the model knows:", [list(c) for c in t.categories_])
print("\nRESULT:", "ALL GOOD" if ok else "PROBLEM - send me this whole output")
