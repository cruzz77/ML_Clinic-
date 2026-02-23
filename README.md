# 🏥 ML Clinic  
## Clinical Appointment No-Show Prediction System  

A Machine Learning system that predicts whether a patient will miss a medical appointment based on historical appointment data.

Built as part of an academic ML project with a production-style architecture and real-time inference using Streamlit.

---

## 📌 Problem Statement

Missed medical appointments (No-Shows) lead to:

- Wasted clinical resources  
- Increased operational costs  
- Reduced healthcare efficiency  

This project builds a predictive ML model to estimate the probability of a patient not showing up for an appointment.

---

## 🧠 Machine Learning Approach

### Dataset
- Kaggle: No-Show Appointments Dataset  
- Contains patient demographic and appointment information.

### Feature Engineering
- Lead Time (days between scheduling and appointment)
- Appointment day of week
- Age filtering
- Removal of ID-based leakage columns

### Model Pipeline

The trained model is a Scikit-learn Pipeline consisting of:

ColumnTransformer (Preprocessing)
        ↓
Classifier (XGBoost / Tree-based model)

The full pipeline is serialized using `joblib` and deployed in Streamlit for real-time inference.

---

## 🏗 Project Structure

```
MLCLINIC/
│
├── backend/                # Model training & ML pipeline
│   └── app/
│
├── frontend/               # Streamlit inference app
│   ├── artifacts/
│   │   └── model.pkl       # Trained ML pipeline
│   ├── components/
│   │   ├── feature_importance.py
│   │   ├── prediction_display.py
│   │   └── upload_section.py
│   ├── utils/
│   │   └── model_loader.py
│   ├── app.py
│   └── requirements.txt
│
├── notebooks/
│   └── ML_Clinic_Baseline.ipynb
│
├── runtime.txt
└── README.md
```

---

## ⚙️ Installation & Setup (Local)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/ML_Clinic.git
cd ML_Clinic/frontend
```

---

### 2️⃣ Create Virtual Environment (Python 3.12 Recommended)

```bash
python3.12 -m venv venv
source venv/bin/activate   # Mac/Linux
```

Windows:
```bash
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```

App will open at:

```
http://localhost:8501
```

---

## 🚀 Deployment (Streamlit Cloud)

1. Push repository to GitHub  
2. Go to Streamlit Cloud  
3. Select repository  
4. Set Main file path to: `frontend/app.py`  
5. Ensure `runtime.txt` contains:

```
python-3.12
```

The app will automatically install dependencies and deploy.

---

## 📊 How It Works

1. User uploads CSV file  
2. Feature engineering is applied  
3. Trained pipeline (`model.pkl`) is loaded  
4. Model performs:
   - `predict()`
   - `predict_proba()`  
5. Results displayed:
   - Prediction table  
   - Average No-Show Risk  
   - Feature Importance  

---

## 📈 Example Output

- Predicted_No_Show (0 or 1)
- No_Show_Probability (0–1)
- Average No-Show Risk %

---

## 🎯 Model Performance

- AUC: ~0.73  
- Tree-based classifier with feature importance support  

---

## 🔍 Key Features

- End-to-end ML pipeline  
- Modular architecture  
- Real-time CSV upload inference  
- Feature importance visualization  
- Production-style model serialization  
- Streamlit Cloud deployment ready  

---

## 🧪 Tech Stack

- Python 3.12  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Joblib  

---

## 👨‍💻 Contributors

- Aditya Chopra  
- Team Members  

---

## 📌 Future Improvements

- FastAPI backend integration  
- Model monitoring  
- Threshold optimization  
- SHAP explainability  
- Docker deployment  
- CI/CD integration  

---

## 📄 License

Academic project for educational purposes.

---

## 🧠 Technical Note

The deployed frontend directly loads a serialized Scikit-learn Pipeline (`model.pkl`) that contains both preprocessing and classifier logic.  

All predictions are generated using real model inference. No dummy outputs are used in the deployed version.