from pathlib import Path

# Project root 
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Data
DATA_PATH = BASE_DIR / "data" / "raw" / "KaggleV2-May-2016.csv"

# Artifacts
ARTIFACTS_DIR = BASE_DIR / "backend" / "app" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"