import cv2
from deepface import DeepFace
import os
from collections import deque

# Hide TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_cascade.empty():
    print("ERROR: Could not load haarcascade_frontalface_default.xml. Check file path!")

# FIXED EMOTION ORDER FOR UNIFIED UI
EMOTION_ORDER = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Global history to smooth out flickering scores (last 15 frames)
score_history = {emo: deque(maxlen=15) for emo in EMOTION_ORDER}

def get_vision_prediction(frame, emotion_buffer):
    """
    Processes a single frame with Temporal Smoothing and Fixed Ordering.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize with 0s in the correct order
    emotions_dict = {emo: 0.0 for emo in EMOTION_ORDER}
    smooth_emotion = "No Face"

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        pad = 10
        face_roi = rgb_frame[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]

        try:
            result = DeepFace.analyze(
                face_roi,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='skip'
            )

            raw_emotions = result[0]['emotion']
            
            # --- SMOOTHING & ORDERING LOGIC ---
            smoothed_dict = {}
            for emo in EMOTION_ORDER:
                val = raw_emotions.get(emo, 0.0)
                score_history[emo].append(val)
                smoothed_dict[emo] = sum(score_history[emo]) / len(score_history[emo])

            emotions_dict = smoothed_dict
            dominant_emotion = max(smoothed_dict, key=smoothed_dict.get)
            confidence = smoothed_dict[dominant_emotion] / 100

            emotion_buffer.append(dominant_emotion)
            smooth_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

            # Annotate
            cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                rgb_frame,
                f"{smooth_emotion} ({confidence:.2f})",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
        except Exception:
            pass

    return rgb_frame, emotions_dict, smooth_emotion
