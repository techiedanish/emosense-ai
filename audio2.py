import os
import librosa
import numpy as np
import pickle

# --- Config ---
CHUNK_DURATION     = 3.0      
CHUNK_OFFSET       = 0.5    
MIN_CHUNK_DURATION = 1.0    

def load_trained_assets(model_path="model.pkl", le_path="label_encoder.pkl", scaler_path="scaler.pkl"):
    if os.path.exists(model_path) and os.path.exists(le_path) and os.path.exists(scaler_path):
        model = pickle.load(open(model_path, "rb"))
        le = pickle.load(open(le_path, "rb"))
        scaler = pickle.load(open(scaler_path, "rb"))
        return model, le, scaler
    return None, None, None

def extract_features_from_array(y, sr):
    if len(y) < sr * MIN_CHUNK_DURATION:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_delta_mean = np.mean(librosa.feature.delta(mfcc).T, axis=0)
    chroma_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast_mean = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    return np.hstack([mfcc_mean, mfcc_delta_mean, chroma_mean, contrast_mean, zcr_mean])

def stream_audio_emotions(file_path, model, le, scaler):
    """Yields emotion probabilities for each 3-second chunk individually."""
    y, sr = librosa.load(file_path, sr=None)
    chunk_samples = int(CHUNK_DURATION * sr)
    offset_samples = int(CHUNK_OFFSET * sr)

    for start in range(0, len(y), chunk_samples):
        chunk = y[start : start + chunk_samples]
        if len(chunk) / sr < MIN_CHUNK_DURATION:
            break
            
        chunk_with_offset = chunk[offset_samples:] if len(chunk) > offset_samples else chunk
        features = extract_features_from_array(chunk_with_offset, sr)
        
        if features is not None:
            features_scaled = scaler.transform(features.reshape(1, -1))
            proba = model.predict_proba(features_scaled)[0]
            yield dict(zip(le.classes_, map(float, proba)))