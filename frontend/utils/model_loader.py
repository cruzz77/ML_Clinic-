from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)