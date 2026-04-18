import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Path Configuration
MODEL_PATH = "model/noshow_model.pkl"
FEATURES_PATH = "model/feature_columns.pkl"

def get_tier(prob: float) -> str:
    if prob >= 0.75: return "Critical"
    if prob >= 0.55: return "High"
    if prob >= 0.35: return "Medium"
    return "Low"

def enrich_patient_row(row: dict) -> dict:
    """
    Calculate derived features and handle noisy data gracefully.
    """
    row = row.copy()
    health_issues = []
    
    # Pre-cleaning: Handle missing basic columns
    for col in ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']:
        if col not in row or pd.isna(row[col]):
            row[col] = 0
            health_issues.append(f"Missing {col} imputed as 0")

    # Age check
    age = row.get('Age', 0)
    if pd.isna(age) or age < 0:
        row['Age'] = 0
        health_issues.append("Negative or missing Age corrected to 0")
    elif age > 115:
        health_issues.append(f"Unusual Age detected: {age}")
    
    # Calculate Lead Time
    try:
        sched = pd.to_datetime(row['ScheduledDay'])
        appt = pd.to_datetime(row['AppointmentDay'])
        lead_time = (appt.date() - sched.date()).days
        
        if lead_time < 0:
            health_issues.append(f"Negative LeadTime detected ({lead_time}); corrected to 0")
            lead_time = 0
            
        row['LeadTime'] = lead_time
        row['DayOfWeek'] = appt.strftime('%A')
    except Exception as e:
        row['LeadTime'] = 0
        row['DayOfWeek'] = "Unknown"
        health_issues.append(f"Date parsing failed: {str(e)}")
        
    # Chronic Conditions sum
    row['ChronicCount'] = int(row.get('Hipertension', 0)) + \
                          int(row.get('Diabetes', 0)) + \
                          int(row.get('Alcoholism', 0)) + \
                          int(row.get('Handcap', 0))
    
    row['_health_check'] = health_issues
    return row

def load_ml_model():
    """
    Load pre-trained model and features.
    """
    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else f"noshow_project/{MODEL_PATH}"
    feat_path = FEATURES_PATH if os.path.exists(FEATURES_PATH) else f"noshow_project/{FEATURES_PATH}"
    
    if os.path.exists(model_path) and os.path.exists(feat_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(feat_path, 'rb') as f:
            features = pickle.load(f)
        return model, features
    return None, None

def run_prediction(patient_row: dict) -> dict:
    """
    Run inference and return probability + tier.
    """
    model, features = load_ml_model()
    
    if model is None:
        # Fallback dummy prediction if model is not found
        # (Useful for initial setup/testing)
        prob = 0.45 if patient_row.get('Scholarship', 0) == 1 else 0.20
        return {
            "probability": prob,
            "tier": get_tier(prob),
            "top_features": ["Scholarship", "LeadTime"]
        }
        
    # Prepare input for model
    # Note: Milestone 1 likely used specific encoding, 
    # here we assume a flat numeric representation for simplicity
    input_data = []
    for f in features:
        val = patient_row.get(f, 0)
        # Handle simple mapping for specific known columns
        if f == 'Gender': val = 1 if val == 'F' else 0
        input_data.append(val)
        
    prob = model.predict_proba([input_data])[0][1]
    
    return {
        "probability": float(prob),
        "tier": get_tier(prob),
        "top_features": features[:3] # Simplified for POC
    }

def format_risk_profile(row: dict, prediction: dict) -> str:
    """
    Create a clean text profile for the LLM.
    """
    profile = f"""
- No-show Probability: {prediction['probability']:.2%} ({prediction['tier']} Risk)
- Lead Time: {row.get('LeadTime', 0)} days
- Day of Week: {row.get('DayOfWeek')}
- Neighbourhood: {row.get('Neighbourhood')}
- SMS Received: {'Yes' if row.get('SMS_received') else 'No'}
- Scholarship (SES Proxy): {'Yes' if row.get('Scholarship') else 'No'}
- Chronic Conditions:
  - Hypertension: {'Yes' if row.get('Hipertension') else 'No'}
  - Diabetes: {'Yes' if row.get('Diabetes') else 'No'}
  - Alcoholism: {'Yes' if row.get('Alcoholism') else 'No'}
  - Handicap Score: {row.get('Handcap', 0)}
- Total Chronic Factors: {row.get('ChronicCount', 0)}
"""
    return profile.strip()
