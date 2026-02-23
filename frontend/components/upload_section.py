import streamlit as st
import pandas as pd

def upload_csv():
    st.subheader("📂 Upload Patient Data (CSV)")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())
        return df

    return None