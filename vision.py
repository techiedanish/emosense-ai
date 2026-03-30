import streamlit as st
import cv2
from deepface import DeepFace
import tempfile
import time
import os
from collections import deque
import numpy as np

# Hide TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.title("🎭 Emotion Detection App")
st.write("Webcam + Video Upload")

mode = st.radio("Choose Input Mode:", ["Webcam", "Upload Video"])

FRAME_WINDOW = st.image([])

# 🔥 Emotion smoothing buffer
emotion_buffer = deque(maxlen=10)

# Face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

#WEBCAM MODE
if mode == "Webcam":
    run = st.checkbox("Start Webcam")
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # 🔥 Slight padding for better detection
            pad = 10
            face = rgb[max(0,y-pad):y+h+pad, max(0,x-pad):x+w+pad]

            try:
                result = DeepFace.analyze(
                    face,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='skip'
                )

                emotions = result[0]['emotion']
                emotion = result[0]['dominant_emotion']
                confidence = max(emotions.values()) / 100  # normalize

                # 🔥 Add to buffer
                emotion_buffer.append(emotion)

                # 🔥 Smooth output
                smooth_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

                cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

                cv2.putText(
                    rgb,
                    f"{smooth_emotion} ({confidence:.2f})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

            except:
                pass

        FRAME_WINDOW.image(rgb)
        time.sleep(0.03)

    cap.release()
    
    
# VIDEO UPLOAD MODE

elif mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        st.write("Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                pad = 10
                face = rgb[max(0,y-pad):y+h+pad, max(0,x-pad):x+w+pad]

                try:
                    result = DeepFace.analyze(
                        face,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='skip'
                    )

                    emotions = result[0]['emotion']
                    emotion = result[0]['dominant_emotion']
                    confidence = max(emotions.values()) / 100

                    emotion_buffer.append(emotion)
                    smooth_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

                    cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    cv2.putText(
                        rgb,
                        f"{smooth_emotion} ({confidence:.2f})",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

                except:
                    pass

            FRAME_WINDOW.image(rgb)
            time.sleep(0.03)

        cap.release()
        st.success("✅ Video processing complete")