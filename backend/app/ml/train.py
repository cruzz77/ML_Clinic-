import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from app.core.config import DATA_PATH, MODEL_PATH, PREPROCESSOR_PATH
from app.ml.preprocess import feature_engineering, build_preprocessor


def train():
    df = pd.read_csv(DATA_PATH)

    # TARGET CLEANING
    df["No-show"] = df["No-show"].replace({"No": 0, "Yes": 1})
    df = df.dropna(subset=["No-show"])
    df["No-show"] = df["No-show"].astype(int)

    df = feature_engineering(df)

    X = df.drop("No-show", axis=1)
    y = df["No-show"]

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(X)

    # XGBoost Model 
    xgb_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        ))
    ])

    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    print("XGBoost ROC-AUC:", auc)

    # Save full pipeline
    joblib.dump(xgb_model, MODEL_PATH)

    print("Model saved successfully.")


if __name__ == "__main__":
    train()