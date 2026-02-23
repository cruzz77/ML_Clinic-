import joblib
import pandas as pd

from app.core.config import MODEL_PATH, PREPROCESSOR_PATH
from app.ml.preprocess import feature_engineering

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


def predict(data: dict):

    df = pd.DataFrame([data])

    df = feature_engineering(df)

    X_processed = preprocessor.transform(df)

    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }