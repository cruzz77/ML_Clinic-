import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["No-show"] = df["No-show"].replace({"No": 0, "Yes": 1})

    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"]).dt.normalize()
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"]).dt.normalize()

    # Calculate lead time 
    df["lead_time"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days
    df = df[df["lead_time"] >= 0]

    df["appointment_day_of_week"] = df["AppointmentDay"].dt.day_name()

    df = df[df["Age"] >= 0]

    df = df.drop(columns=["PatientId", "AppointmentID"], errors="ignore")

    return df


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return preprocessor