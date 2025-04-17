import os
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

# Load pre-trained models from Trained_Models folder
MODELS_DIR = "Trained_Models"

# 1. Load sentence transformer model
embedder = SentenceTransformer(os.path.join(MODELS_DIR, 'sentence_transformer_model'))

# 2. Load scaler using joblib
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

# 3. Load Gradient Boosting model using joblib
model = joblib.load(os.path.join(MODELS_DIR, 'GradientBoosting_relevancy_model.pkl'))

def calculate_relevancy(extracted_data, job_description):
    """
    Calculates relevancy between a resume (extracted_data) and a job description using saved models.
    """
    # Join resume sections
    resume_text = " ".join([
        " ".join(extracted_data.get("skills", [])),
        " ".join(extracted_data.get("education", [])),
        " ".join(extracted_data.get("experience", []))
    ])

    # Join job description sections
    job_desc_text = " ".join([
        " ".join(job_description.get("required_skills", [])),
        " ".join(job_description.get("required_education", [])),
        " ".join(job_description.get("required_experience", []))
    ])

    # Compute embeddings using saved sentence transformer
    resume_embedding = embedder.encode(resume_text, convert_to_numpy=True)
    job_embedding = embedder.encode(job_desc_text, convert_to_numpy=True)

    # Combine embeddings (assuming concatenation was used during training)
    combined_embedding = np.hstack([resume_embedding, job_embedding])

    # Predict scaled score and inverse transform
    scaled_score = model.predict(combined_embedding.reshape(1, -1))
    relevancy_score = scaler.inverse_transform(scaled_score.reshape(-1, 1))[0][0]

    # Ensure score stays within 0-100 bounds
    relevancy_score = max(0, min(100, round(relevancy_score, 2)))
    
    return relevancy_score

def interpret_score(score):
    """Interpret the numerical score into categorical labels"""
    if score > 85:
        return "Excellent Match"
    elif score > 70:
        return "Good Match"
    elif score > 50:
        return "Moderate Match"
    else:
        return "Low Match"
