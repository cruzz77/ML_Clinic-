import streamlit as st
import pandas as pd

from utils.model_loader import load_model
from components.upload_section import upload_csv
from components.prediction_display import show_predictions
from components.feature_importance import show_feature_importance


st.set_page_config(page_title="ML Clinic", layout="wide")

st.title("🏥 Clinical Appointment No-Show Prediction System")
st.markdown("Built with XGBoost | Production ML Pipeline")

# Load trained pipeline
model = load_model()

# Upload section
df = upload_csv()

if df is not None:

    try:
        # -----------------------------
        # Feature Engineering (MUST MATCH TRAINING)
        # -----------------------------

        if "ScheduledDay" in df.columns:
            df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"]).dt.normalize()

        if "AppointmentDay" in df.columns:
            df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"]).dt.normalize()

        if "ScheduledDay" in df.columns and "AppointmentDay" in df.columns:
            df["lead_time"] = (
                df["AppointmentDay"] - df["ScheduledDay"]
            ).dt.days

        # 🔥 REQUIRED COLUMN
        if "AppointmentDay" in df.columns:
            df["appointment_day_of_week"] = (
                df["AppointmentDay"].dt.day_name()
            )

        if "Age" in df.columns:
            df = df[df["Age"] >= 0]

        df = df.drop(columns=["PatientId", "AppointmentID"], errors="ignore")

        # -----------------------------
        # Prediction
        # -----------------------------
        show_predictions(model, df)

        # Feature Importance
        show_feature_importance(model)

    except Exception as e:
        st.error(f"Error during prediction: {e}")