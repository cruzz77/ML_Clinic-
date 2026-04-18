import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def train_basic_model():
    print("Training basic model for Milestone 2 runnability...")
    
    # Path to dataset
    data_path = "data/raw/KaggleV2-May-2016.csv"
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Simple preprocessing
    df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})
    
    # Features mentioned in prompt + basic ones
    features = ['Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']
    
    X = df[features]
    y = df['No-show']
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Create model dir
    os.makedirs("model", exist_ok=True)
    
    with open("model/noshow_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/feature_columns.pkl", "wb") as f:
        pickle.dump(features, f)
        
    print("Model and features saved to model/ directory.")

if __name__ == "__main__":
    train_basic_model()
