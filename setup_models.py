"""
Script to handle model setup for Streamlit Cloud deployment
This script can be used to download or recreate models if they're not available
"""

import os
import requests
import joblib
from pathlib import Path

def download_model_from_url(url: str, filename: str):
    """Download model from external URL if available"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Create model directory if it doesn't exist
        Path("model").mkdir(exist_ok=True)
        
        with open(f"model/{filename}", 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def check_model_files():
    """Check if model files exist"""
    model_files = [
        "model/imdb_model.pkl",
        "model/tfidf_vectorizer.pkl"
    ]
    
    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return missing_files

def create_dummy_models():
    """Create dummy models for demonstration purposes"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        # Create model directory
        Path("model").mkdir(exist_ok=True)
        
        # Create a simple dummy vectorizer
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        # Fit on some dummy data
        dummy_texts = [
            "This movie is great and amazing",
            "This movie is terrible and boring",
            "Good film with excellent acting",
            "Bad movie with poor story"
        ]
        vectorizer.fit(dummy_texts)
        
        # Create a simple dummy model
        model = LogisticRegression()
        X_dummy = vectorizer.transform(dummy_texts)
        y_dummy = [1, 0, 1, 0]  # positive, negative, positive, negative
        model.fit(X_dummy, y_dummy)
        
        # Save the models
        joblib.dump(model, "model/imdb_model.pkl")
        joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
        
        print("✅ Created dummy models for demonstration")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create dummy models: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Checking model files...")
    
    missing_files = check_model_files()
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        print("🔧 Attempting to create dummy models...")
        
        if create_dummy_models():
            print("✅ Setup complete! App should work with dummy models.")
            print("⚠️  Note: These are demonstration models with limited accuracy.")
        else:
            print("❌ Failed to setup models. Manual intervention required.")
    else:
        print("✅ All model files present!")