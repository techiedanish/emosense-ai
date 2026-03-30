import streamlit as st
import os
import librosa
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# ---------------- UI ----------------
st.title("Speech Emotion Recognition")
st.write("Upload an audio file (.wav) to predict emotion")

# ---------------- Emotion Mapping ----------------
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# ---------------- Config ----------------
CHUNK_DURATION  = 3      # seconds per chunk
CHUNK_OFFSET    = 0.5    # skip first 0.5s of each chunk (removes silence/noise)
MIN_CHUNK_DURATION = 1.0 # ignore chunks shorter than this (too little audio to analyse)

# ---------------- Functions ----------------
def get_emotion(file_path):
    file_name = os.path.basename(file_path)
    emotion_code = file_name.split("-")[2]
    return emotion_dict[emotion_code]


def extract_features_from_array(y, sr):
    # Guard — if chunk is too short, librosa features will be empty/broken
    if len(y) < sr * MIN_CHUNK_DURATION:
        return None

    mfcc            = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean       = np.mean(mfcc.T, axis=0)
    mfcc_delta_mean = np.mean(librosa.feature.delta(mfcc).T, axis=0)
    chroma_mean     = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast_mean   = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr_mean        = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)

    return np.hstack([mfcc_mean, mfcc_delta_mean, chroma_mean, contrast_mean, zcr_mean])


def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=CHUNK_DURATION, offset=CHUNK_OFFSET)
    return extract_features_from_array(y, sr)


# ---------------- Chunking Logic ----------------
def split_audio_into_chunks(file_path):
    # Load full file — no duration limit
    y, sr = librosa.load(file_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)

    chunks = []
    chunk_samples = int(CHUNK_DURATION * sr)
    offset_samples = int(CHUNK_OFFSET * sr)

    start = 0
    chunk_index = 0

    while start < len(y):
        end = start + chunk_samples

        # Slice this chunk
        chunk = y[start:end]
        chunk_duration = len(chunk) / sr

        # Skip chunk if it's too short to analyse reliably
        if chunk_duration < MIN_CHUNK_DURATION:
            break

        # Apply offset — skip first CHUNK_OFFSET seconds of each chunk
        # This removes leading silence/noise from each window
        chunk_with_offset = chunk[offset_samples:] if len(chunk) > offset_samples else chunk

        start_time = round(start / sr, 2)
        end_time   = round(min(end, len(y)) / sr, 2)

        chunks.append((chunk_with_offset, sr, start_time, end_time))

        start += chunk_samples
        chunk_index += 1

    return chunks, total_duration


def predict_chunks(file_path, model, le, scaler):
    chunks, total_duration = split_audio_into_chunks(file_path)

    if len(chunks) == 0:
        raise ValueError("Audio file is too short or could not be loaded.")

    chunk_results   = []
    all_proba       = []

    for y_chunk, sr, start_time, end_time in chunks:
        features = extract_features_from_array(y_chunk, sr)

        # Skip chunk if feature extraction failed (too short, silent, etc.)
        if features is None:
            continue

        features_scaled = scaler.transform(features.reshape(1, -1))
        proba           = model.predict_proba(features_scaled)[0]
        prob_dict       = dict(zip(le.classes_, map(float, proba)))

        chunk_results.append((start_time, end_time, prob_dict))
        all_proba.append(proba)

    if len(all_proba) == 0:
        raise ValueError("No valid chunks found. Audio may be silent or corrupted.")

    avg_proba    = np.mean(all_proba, axis=0)
    final_probs  = dict(zip(le.classes_, map(float, avg_proba)))

    return final_probs, chunk_results, total_duration


# ---------------- Train & Save ----------------
def train_and_save(dataset_path, use_grid_search=False):
    file_paths = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_paths.append(os.path.join(root, file))

    if len(file_paths) == 0:
        st.error("No .wav files found at the given path. Please check the dataset path.")
        return None, None, None, None

    X, y = [], []

    st.write(f"Found {len(file_paths)} audio files. Extracting features...")
    progress = st.progress(0)

    for i, file in enumerate(file_paths):
        try:
            features = extract_features(file)
            emotion  = get_emotion(file)
            if features is not None:
                X.append(features)
                y.append(emotion)
        except Exception:
            pass
        progress.progress((i + 1) / len(file_paths))

    X = np.array(X)
    y = np.array(y)

    le        = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    if use_grid_search:
        st.write("Running Grid Search to find best C and gamma (this takes longer)...")
        param_grid = {
            "C":     [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01]
        }
        base_svm = SVC(kernel="rbf", probability=True,
                       class_weight="balanced", random_state=42)
        grid = GridSearchCV(base_svm, param_grid, cv=5,
                            scoring="accuracy", n_jobs=-1, verbose=0)
        grid.fit(X_train_scaled, y_train)
        model = grid.best_estimator_
        st.write(f"Best params — C={grid.best_params_['C']}, gamma={grid.best_params_['gamma']}")
    else:
        model = SVC(kernel="rbf", C=10, gamma="scale",
                    probability=True, class_weight="balanced", random_state=42)
        model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test)

    pickle.dump(model,  open("model.pkl",         "wb"))
    pickle.dump(le,     open("label_encoder.pkl",  "wb"))
    pickle.dump(scaler, open("scaler.pkl",          "wb"))

    return model, le, scaler, accuracy


# ---------------- Load Model ----------------
def load_model():
    if os.path.exists("model.pkl"):
        model  = pickle.load(open("model.pkl",        "rb"))
        le     = pickle.load(open("label_encoder.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl",         "rb"))
        return model, le, scaler
    return None, None, None


# ---------------- Fusion-Ready Predict Function ----------------
def predict_emotion(file_path, model, le, scaler):
    # Check actual audio duration first
    y_check, sr_check   = librosa.load(file_path, sr=None)
    actual_duration     = librosa.get_duration(y=y_check, sr=sr_check)

    if actual_duration <= CHUNK_DURATION:
        # Short file — original single-pass prediction
        features        = extract_features(file_path)
        features_scaled = scaler.transform(features.reshape(1, -1))
        proba           = model.predict_proba(features_scaled)[0]
        return dict(zip(le.classes_, map(float, proba)))
    else:
        # Long file — chunk and average
        final_probs, _, _ = predict_chunks(file_path, model, le, scaler)
        return final_probs


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
dataset_path = st.sidebar.text_input("Dataset Path", "C:/Kaggle/ravdess/")

use_grid_search = st.sidebar.checkbox(
    "Use Grid Search (slower but finds best C & gamma)",
    value=False
)

if st.sidebar.button("Train Model"):
    with st.spinner("Training SVM model..."):
        model, le, scaler, accuracy = train_and_save(dataset_path, use_grid_search)
        if model is not None:
            st.sidebar.success(f"SVM trained! Accuracy: {accuracy:.2%}")

# ---------------- Load Automatically ----------------
model, le, scaler = load_model()

if model is None:
    st.warning("No saved model found. Please train first using the sidebar.")
else:
    st.success("SVM model loaded successfully")

# ---------------- File Upload & Prediction ----------------
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None and model is not None:
    st.audio(uploaded_file)

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        # Check duration to decide display mode
        y_check, sr_check = librosa.load("temp.wav", sr=None)
        duration          = librosa.get_duration(y=y_check, sr=sr_check)

        st.caption(f"Audio duration: {duration:.1f}s")

        if duration <= CHUNK_DURATION:
            # ── Short audio — single prediction ──────────────────────
            emotion_probs = predict_emotion("temp.wav", model, le, scaler)
            emotion_probs = dict(sorted(emotion_probs.items(),
                                        key=lambda x: x[1], reverse=True))
            top_emotion   = list(emotion_probs.keys())[0]
            confidence    = emotion_probs[top_emotion]

            st.subheader("Prediction")
            st.metric(label="Detected Emotion", value=top_emotion.capitalize())
            st.metric(label="Confidence",        value=f"{confidence:.2%}")

            st.subheader("All Emotion Probabilities")
            for emo, prob in emotion_probs.items():
                st.progress(float(prob), text=f"{emo.capitalize()}: {prob:.2%}")

        else:
            # ── Long audio — chunked prediction ──────────────────────
            st.info(f"Audio is {duration:.1f}s — splitting into {CHUNK_DURATION}s chunks and averaging predictions.")

            with st.spinner("Analysing chunks..."):
                final_probs, chunk_results, total_duration = predict_chunks(
                    "temp.wav", model, le, scaler
                )

            # Overall result (averaged across all chunks)
            final_sorted  = dict(sorted(final_probs.items(),
                                        key=lambda x: x[1], reverse=True))
            top_emotion   = list(final_sorted.keys())[0]
            confidence    = final_sorted[top_emotion]

            st.subheader("Overall Prediction (averaged across all chunks)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Detected Emotion", top_emotion.capitalize())
            col2.metric("Confidence",        f"{confidence:.2%}")
            col3.metric("Chunks analysed",   str(len(chunk_results)))

            st.subheader("Overall Emotion Probabilities")
            for emo, prob in final_sorted.items():
                st.progress(float(prob), text=f"{emo.capitalize()}: {prob:.2%}")

            # Per-chunk timeline — shows how emotion changes over time
            st.subheader("Emotion Timeline (per chunk)")
            st.caption("Shows the dominant emotion detected in each time window.")

            for start_time, end_time, prob_dict in chunk_results:
                chunk_sorted  = sorted(prob_dict.items(),
                                       key=lambda x: x[1], reverse=True)
                chunk_top_emo = chunk_sorted[0][0]
                chunk_conf    = chunk_sorted[0][1]

                with st.expander(
                    f"{start_time}s → {end_time}s   |   "
                    f"{chunk_top_emo.capitalize()} ({chunk_conf:.0%})"
                ):
                    for emo, prob in chunk_sorted:
                        st.progress(float(prob),
                                    text=f"{emo.capitalize()}: {prob:.2%}")

            emotion_probs = final_probs  # use for fusion output below

        # Fusion-ready JSON — same format regardless of audio length
        st.subheader("Fusion Output (for multimodal pipeline)")
        st.json({k: round(float(v), 4)
                 for k, v in sorted(final_probs.items()
                                    if duration > CHUNK_DURATION
                                    else emotion_probs.items(),
                                    key=lambda x: x[1], reverse=True)})

    except Exception as e:
        st.error(f"Error during prediction: {e}")

    finally:
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")