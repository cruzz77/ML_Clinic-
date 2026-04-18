# 🏥 ML Clinic: Agentic Care Coordination System

**AIML Course Project | End-to-End ML & Agentic AI**

ML Clinic is a comprehensive healthcare intelligence system that evolves from baseline machine learning predictions to a fully autonomous **Agentic Care Coordination Assistant**. The system predicts patient no-show risks and generates validated, evidence-based intervention strategies using RAG and self-correcting AI workflows.

---

## 🚀 Milestone 2 - Agentic Care Coordination (Latest Update)

The project has been scaled to an advanced **Agentic AI** platform found in the `noshow_project/` directory.

### 🌟 Latest Agentic Features
- **5-Node Agentic Workflow**: Orchestrated via **LangGraph** (`Ingestor` → `RAG` → `Reasoner` → `Builder` → `Critic`).
- **Clinical Self-Correction**: A dedicated **Critic Node** audits AI recommendations against medical guidelines to ensure 100% evidence-based advice.
- **Session-Based History**: Persistent history sidebar allowing clinical staff to switch between analyzed patients seamlessly.
- **Operational Worklist**: Population-level batch analysis with prioritized CSV export for hospital outreach.
- **Semantic RAG**: Real-time retrieval from a clinical knowledge base using **FAISS** and **Sentence-Transformers**.
- **Official PDF Reports**: Generates professional care coordination summaries with clinical citations.

### 🔗 Live URL (Agentic AI App)
**Hosted App**: [https://ft5vbrt6kryvjzsynxzkf4.streamlit.app/](https://ft5vbrt6kryvjzsynxzkf4.streamlit.app/)

---

## 🏗 Project Architecture

```
ML CLINIC /
│
├── noshow_project/          # [Milestone 2] Agentic AI Platform (Latest)
│   ├── agent/               # LangGraph state & node orchestration
│   ├── rag/                 # FAISS vector database & retrievers
│   ├── schemas/             # Pydantic structured output models
│   ├── utils/               # LLM wrappers & robust preprocessing
│   └── app.py               # Main Agentic UI
│
├── frontend/                # [Milestone 1] Baseline ML App
│   ├── artifacts/           # Trained pipeline (model.pkl)
│   └── app.py               # Baseline ML UI
│
├── backend/                 # Model training logic
└── data/                    # Kaggle No-Show dataset
```

---

## ⚙️ Installation & Setup (Milestone 2)

### 1. Navigate to the Agentic Project
```bash
cd noshow_project
```

### 2. Set up Environment
Create a `.env` file:
```env
GROQ_API_KEY=your_key_here
HF_API_TOKEN=your_token_here
```

### 3. Install & Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 👨‍💻 Contributors

- Aditya Chopra
- Soham Goel
- Ankit Kumar
- Nilesh Nand Nal

---

## 🎯 Project Evolution
- **Milestone 1**: Predictive modeling (XGBoost/Random Forest) with 73% AUC.
- **Milestone 2**: Integration of LLMs, RAG, and Agentic logic for clinical decision support.

## 📄 License
Academic project for educational purposes.
