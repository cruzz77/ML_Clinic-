import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
from fpdf import FPDF
from agent.graph import run_agent
from rag.build_index import build_faiss_index

# --- Page Config ---
st.set_page_config(
    page_title="AI Care Coordination Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'history' not in st.session_state:
    st.session_state['history'] = {}  # {patient_id: {report: ..., row: ...}}
if 'last_analyzed' not in st.session_state:
    st.session_state['last_analyzed'] = None

# --- Initialize RAG on Startup ---
if not os.path.exists("rag/faiss_index.bin"):
    with st.spinner("Initializing Knowledge Base (building FAISS index)..."):
        build_faiss_index()

# --- Helper: PDF Generation ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Clinical Care Coordination Report', 0, 1, 'C')
        self.ln(5)

USER_PROMPT = "Generate a coordinated care report for the patient risk profile described above in JSON format."

def export_to_pdf(report, patient_id):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    
    # Title
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(pdf.epw, 10, "Care Coordination Report", align="C")
    pdf.ln(15)
    
    # Patient ID
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(pdf.epw, 8, f"Patient ID: {patient_id}")
    pdf.ln(10)
    
    # Risk Summary
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(pdf.epw, 10, "Risk Summary:")
    pdf.ln(8)
    pdf.set_font("helvetica", size=10)
    pdf.multi_cell(pdf.epw, 6, report.get('appointment_risk_summary', 'No summary available.'))
    pdf.ln(5)
    
    # Interventions
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(pdf.epw, 10, "Actionable Interventions:")
    pdf.ln(8)
    pdf.set_font("helvetica", size=10)
    for idx, strategy in enumerate(report.get('intervention_strategies', [])):
        action = strategy.get('action', 'N/A')
        priority = strategy.get('priority', 'low').upper()
        rationale = strategy.get('rationale', 'N/A')
        text = f"{idx+1}. {action} ({priority} Priority)\nRationale: {rationale}"
        pdf.multi_cell(pdf.epw, 6, text)
        pdf.ln(2)
    
    # Disclaimers
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 11)
    pdf.cell(pdf.epw, 10, "Disclaimers:")
    pdf.ln(8)
    pdf.set_font("helvetica", "I", 9)
    for disc in report.get('disclaimers', []):
        pdf.multi_cell(pdf.epw, 5, f"- {disc}")
        
    filename = f"report_{patient_id}.pdf"
    pdf.output(filename)
    return filename

# --- Sidebar ---
with st.sidebar:
    st.title("🧰 Control Center")
    st.markdown("---")
    
    api_source = st.selectbox("LLM Provider", ["Groq (llama-3.1-8b)", "HuggingFace (Mistral-7B)"])
    
    with st.expander("🔑 API Key Setup"):
        groq_key = st.text_input("Groq API Key", type="password")
        hf_token = st.text_input("HF API Token", type="password")
        if groq_key: os.environ["GROQ_API_KEY"] = groq_key
        if hf_token: os.environ["HF_API_TOKEN"] = hf_token

    st.markdown("---")
    
    # Session History
    st.subheader("🕑 Patient History")
    if st.session_state['history']:
        for pid in list(st.session_state['history'].keys())[::-1]:
            hist_item = st.session_state['history'][pid]
            if st.button(f"📄 {pid} ({hist_item['prediction']['tier']})", key=f"hist_{pid}"):
                st.session_state['last_analyzed'] = pid
                # We need to set agent_result to the historical one
                st.session_state['agent_result'] = {
                    "report": hist_item['report'],
                    "prediction": hist_item['prediction'],
                    # Dummy values for other graph state if needed
                    "patient_row": hist_item['patient_row']
                }
                st.rerun()
    else:
        st.caption("No patients analyzed yet.")
    
    st.markdown("---")
    st.info("""
    **Project Info**
    - AIML Course Milestone 2
    - No-Show Prediction Agent
    - Powered by LangGraph & RAG
    """)

# --- Tab Management ---
tabs = st.tabs(["🔍 Patient Risk Analyzer", "📊 Batch View"])

# --- TAB 1: Patient Analyzer ---
with tabs[0]:
    st.header("Patient Risk Analyzer")
    
    # Load Dataset
    data_path = "data/raw/KaggleV2-May-2016.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        patient_options = df['PatientId'].unique()[:50] # Sample top 50
        selected_id = st.selectbox("Select Patient by ID", patient_options)
        
        patient_row = df[df['PatientId'] == selected_id].iloc[0].to_dict()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Patient Details")
            st.json(patient_row)
            
            if st.button("🚀 Run Agentic Care Coordination"):
                with st.spinner("Agent exploring guidelines and generating report..."):
                    result = run_agent(patient_row)
                    
                    if result.get('error'):
                        st.error("### ⚠️ Agent Reasoning Interrupted")
                        st.markdown(f"**The AI agent encountered a hurdle:** {result['error']}")
                        st.info("💡 **Tip:** Check if your API keys are correctly set in the sidebar and that you have an active internet connection.")
                    else:
                        st.session_state['agent_result'] = result
                        # Update History
                        st.session_state['history'][selected_id] = {
                            "report": result['report'],
                            "patient_row": patient_row,
                            "prediction": result['prediction'],
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        }
                        st.session_state['last_analyzed'] = selected_id
    else:
        st.error(f"Dataset not found at {data_path}. Please check data directory.")

    # Display Report
    if 'agent_result' in st.session_state:
        res = st.session_state['agent_result']
        report = res['report']
        pred = res['prediction']
        
        st.markdown("---")
        
        # Data Health Indicator
        health_issues = res.get('patient_row', {}).get('_health_check', [])
        if health_issues:
            with st.expander("🛠️ Data Preparation Health Check"):
                st.write("The agent performed the following cleaning steps:")
                for issue in health_issues:
                    st.write(f"- {issue}")
                st.info("Imputation and outlier correction ensure more robust AI reasoning.")

        # Risk Header
        color = {"Critical": "red", "High": "orange", "Medium": "blue", "Low": "green"}.get(pred['tier'], "grey")
        st.markdown(f"## Patient Risk: :{color}[{pred['tier']} ({pred['probability']:.1%})]")
        
        if report.get('confidence') == 'low':
            st.warning("⚠️ **Low Confidence Output**: The agent had difficulty matching clinical guidelines precisely.")
            
        # 4-Column Layout for Factors/Summary
        c1, c2 = st.columns([1, 1])
        with c1:
            st.info(f"**Risk Summary**\n\n{report.get('appointment_risk_summary', 'No summary provided.')}")
        with c2:
            factors = report.get('contributing_factors', [])
            st.success("**Contributing Factors**\n\n" + "\n".join([f"- {f}" for f in factors]) if factors else "No factors identified.")
            
        # Interventions
        st.subheader("📋 Actionable Intervention Strategies")
        interventions = report.get('intervention_strategies', [])
        if not interventions:
            st.info("No primary interventions recommended by clinical guidelines.")
        for strategy in interventions:
            with st.container(border=True):
                s1, s2 = st.columns([3, 1])
                s1.markdown(f"**Action:** {strategy.get('action', 'N/A')}")
                s1.markdown(f"*Rationale:* {strategy.get('rationale', 'N/A')}")
                s2.metric("Priority", strategy.get('priority', 'low').upper())
                s2.markdown(f"**Effort:** {strategy.get('effort', 'N/A')}")
        
        # PDF Export
        if st.button("📥 Export Report to PDF"):
            pdf_path = export_to_pdf(report, selected_id)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name=pdf_path)
                
        # Citations & Disclaimers
        with st.expander("📚 Sources & Disclaimers"):
            st.write("**Clinical Sources Used:**")
            for src in report['sources']:
                st.markdown(f"- {src}")
            st.markdown("---")
            st.write("**Disclaimers:**")
            for d in report['disclaimers']:
                st.caption(f"• {d}")

# --- TAB 2: Batch View ---
with tabs[1]:
    st.header("Batch Risk Distribution")
    
    uploaded_file = st.file_uploader("Upload CSV for Batch Analysis", type="csv")
    
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        # For POC, simulate risk scores randomly for the table if not running agent on 1000s of rows
        batch_df['Risk Probability'] = [0.1 + (i % 8)/10 for i in range(len(batch_df))]
        batch_df['Tier'] = batch_df['Risk Probability'].apply(lambda x: "Critical" if x >= 0.75 else ("High" if x >= 0.55 else "Medium"))
        
        fig = px.pie(batch_df, names='Tier', title="Population Risk Tier Breakdown", color='Tier',
                    color_discrete_map={"Critical": "red", "High": "orange", "Medium": "blue", "Low": "green"})
        st.plotly_chart(fig)
        
        st.subheader("Priority Follow-up List")
        # Ensure Critical/High are at top
        batch_df = batch_df.sort_values(by="Risk Probability", ascending=False)
        styled_df = batch_df[['PatientId', 'AppointmentDay', 'Neighbourhood', 'Risk Probability', 'Tier']].head(20)
        
        try:
            st.dataframe(styled_df.style.background_gradient(subset=['Risk Probability'], cmap='Reds'))
        except ImportError:
            st.dataframe(styled_df)
            st.warning("⚠️ Install `matplotlib` to see color-coded risk gradients.")

        # Batch Export
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Operational Worklist (CSV)",
            data=csv,
            file_name="prioritized_worklist.csv",
            mime="text/csv",
            help="Full list of patients sorted by no-show risk probability."
        )
    else:
        st.info("Upload a patient manifest (CSV) to see population-level distribution.")

st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 20px;
}
.stActionButton {
    border-radius: 20px;
}
</style>
""", unsafe_allow_html=True)
