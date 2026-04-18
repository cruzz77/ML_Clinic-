# No-Show Predictor & Agentic Care Coordination

**AIML Course | Milestone 2 | End-Semester Project**

This project implements an Agentic AI system that predicts patient no-show risks and generates personalized care coordination reports using clinical guidelines (RAG) and LLMs (Groq/HuggingFace).

## 🚀 Features
- **LangGraph Orchestration**: 4-node pipeline (Ingestor -> RAG -> Reasoner -> Builder).
- **RAG Knowledge Base**: 10+ clinical guidelines indexed with FAISS & Sentence-Transformers.
- **Dual LLM Support**: Primary (Groq llama-3.1-8b) with fallback to HuggingFace (Mistral-7B).
- **Structured Output**: Pydantic-validated care coordination reports.
- **Rich Streamlit UI**: Interactive patient analyzer and population risk distribution.
- **PDF Export**: Generate professional care coordination reports for staff.

## 🛠️ Installation & Setup

1. **Clone the repository** (or navigate to `noshow_project/`)
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up Environment Variables**:
   Update `.env` with your API keys:
   ```env
   GROQ_API_KEY=your_key
   HF_API_TOKEN=your_token
   ```
4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
- `agent/`: LangGraph state, nodes, and graph wiring.
- `rag/`: FAISS index building and retrieval logic.
- `utils/`: LLM wrappers, preprocessing, and tiering logic.
- `schemas/`: Pydantic models for structured reports.

## 🚢 Deployment
This app is designed to run on **Hugging Face Spaces**. 
- Add `GROQ_API_KEY` and `HF_API_TOKEN` to the Spaces Secrets.
- The app automatically builds the FAISS index on startup if it's missing.

## 📊 Dataset
The project uses the [Kaggle No-show appointments](https://www.kaggle.com/datasets/joniarroba/noshowappointments) dataset. Ensure `KaggleV2-May-2016.csv` is present in `data/raw/`.

## 🔗 Links
- **Hosted App**: [APP_URL]
- **GitHub Repository**: [GITHUB_URL]

---
**Team Name**: [TEAM_NAME_PLACEHOLDER]
