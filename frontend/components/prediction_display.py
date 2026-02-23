import streamlit as st
import pandas as pd

def show_predictions(model, df):
    st.subheader("📊 Prediction Results")

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    result_df = df.copy()
    result_df["No_Show_Prediction"] = predictions
    result_df["No_Show_Probability"] = probabilities

    st.dataframe(result_df.head())

    st.metric(
        label="Average No-Show Risk",
        value=f"{result_df['No_Show_Probability'].mean():.2%}"
    )

    st.success("Predictions generated successfully!")