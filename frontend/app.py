import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="ML Clinic", layout="wide")

st.title("🏥 Clinical Appointment No-Show Prediction System")
st.markdown("Milestone 1 Demo Version")

st.subheader("📂 Upload Patient Data (CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    try:
        # Basic feature engineering (safe operations)
        if "ScheduledDay" in df.columns:
            df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"]).dt.normalize()

        if "AppointmentDay" in df.columns:
            df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"]).dt.normalize()

        if "AppointmentDay" in df.columns and "ScheduledDay" in df.columns:
            df["lead_time"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days

        # 🔥 Dummy prediction (stable for deployment)
        df["Predicted_No_Show"] = np.random.randint(0, 2, size=len(df))
        df["No_Show_Probability"] = np.random.uniform(0.2, 0.8, size=len(df))

        st.subheader("📊 Prediction Results")
        st.dataframe(df.head())

        st.metric(
            label="Average No-Show Risk",
            value=f"{df['No_Show_Probability'].mean():.2%}"
        )

        st.success("Demo predictions generated successfully!")

    except Exception as e:
        st.error(f"Error during processing: {e}")