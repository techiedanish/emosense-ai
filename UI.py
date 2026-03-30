import streamlit as st
import cv2
import tempfile
import os
import time
import pandas as pd
import altair as alt
from collections import deque

# Import your custom modules
from vision2 import get_vision_prediction, EMOTION_ORDER
from audio2 import load_trained_assets, stream_audio_emotions
from fusion_engine import calculate_fusion, get_dominant_emotion

# --- PAGE CONFIG ---
st.set_page_config(page_title="EmoSense Multimodal AI", layout="wide")

# --- INITIALIZE PERSISTENT MEMORY ---
if 'v_probs' not in st.session_state: st.session_state.v_probs = {e: 0.0 for e in EMOTION_ORDER}
if 'a_probs' not in st.session_state: st.session_state.a_probs = {e: 0.0 for e in EMOTION_ORDER}
if 'f_probs' not in st.session_state: st.session_state.f_probs = {e: 0.0 for e in EMOTION_ORDER}
if 'v_samples' not in st.session_state: st.session_state.v_samples = [] 
if 'audio_gen' not in st.session_state: st.session_state.audio_gen = None
if 'last_update' not in st.session_state: st.session_state.last_update = 0

def draw_chart(data_dict, title, color):
    df = pd.DataFrame(list(data_dict.items()), columns=['Emotion', 'Confidence'])
    df['Emotion'] = pd.Categorical(df['Emotion'], categories=EMOTION_ORDER, ordered=True)
    df = df.sort_values('Emotion')

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Emotion:N', sort=EMOTION_ORDER, title="Emotion Type", axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Confidence:Q', scale=alt.Scale(domain=[0, 100]), title="Confidence (%)"),
        color=alt.value(color)
    ).properties(title=title, height=250).configure_axisX(
        titleColor=color, titleFontWeight='bold'
    ).configure_axisY(
        titleColor=color, titleFontWeight='bold'
    )
    return st.altair_chart(chart, use_container_width=True)

st.title("EmoSense AI")
model_a, le_a, scaler_a = load_trained_assets()

# --- INPUTS ---
col1, col2 = st.columns(2)
with col1:
    v_source = st.radio("Visual Source:", ("Webcam", "Upload Video"))
    v_file = st.file_uploader("Video", type=["mp4"]) if v_source == "Upload Video" else None
with col2:
    a_file = st.file_uploader("Audio (.wav)", type=["wav"])

start_btn = st.button("Start Multimodal Analysis")
video_slot = st.empty()

# --- OUTPUT SLOTS ---
st.header("Individual Model Confidence")
out_v, out_a = st.columns(2)
v_chart = out_v.empty()
a_chart = out_a.empty()

st.markdown("---")
st.header("Multimodal Fusion")
f_chart = st.empty()
verdict_slot = st.empty() # NEW: Slot for the largest value text

# --- MASTER LOOP ---
if start_btn:
    v_path = None
    if v_source == "Upload Video" and v_file:
        t = tempfile.NamedTemporaryFile(delete=False); t.write(v_file.read()); v_path = t.name
    
    if a_file and model_a:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as t:
            t.write(a_file.getvalue()); a_path = t.name
        st.session_state.audio_gen = stream_audio_emotions(a_path, model_a, le_a, scaler_a)
        st.session_state.last_update = time.time()

    cap = cv2.VideoCapture(0 if v_source == "Webcam" else v_path)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Vision (Every Frame)
            img, v_dict, _ = get_vision_prediction(frame, deque(maxlen=10))
            st.session_state.v_probs = v_dict
            st.session_state.v_samples.append(v_dict) 
            
            video_slot.image(img, channels="RGB")
            with v_chart.container(): draw_chart(v_dict, "Vision (Live)", "#FF4B4B")

            # 2. Audio & Fusion (Every 3 Seconds)
            if st.session_state.audio_gen and (time.time() - st.session_state.last_update >= 3.0):
                try:
                    raw_a = next(st.session_state.audio_gen)
                    a_norm = {e: 0.0 for e in EMOTION_ORDER}
                    c_val = raw_a.get('calm', 0.0); n_val = raw_a.get('neutral', 0.0)
                    for k, v in raw_a.items():
                        if k in a_norm and k != 'neutral': a_norm[k] = v * 100
                    a_norm['neutral'] = (c_val + n_val) * 100
                    st.session_state.a_probs = a_norm
                    
                    # Compute Fusion
                    st.session_state.f_probs = calculate_fusion(
                        st.session_state.v_samples, 
                        st.session_state.a_probs, 
                        EMOTION_ORDER
                    )
                    st.session_state.v_samples = [] 
                    st.session_state.last_update = time.time()
                except StopIteration: st.session_state.audio_gen = None

            # 3. UI Updates
            with a_chart.container(): draw_chart(st.session_state.a_probs, "Audio (3s Window)", "#1F77B4")
            with f_chart.container(): draw_chart(st.session_state.f_probs, "Fusion Result", "#7D3C98")
            
            # --- SHOW THE LARGEST VALUE ---
            dom_emo, dom_val = get_dominant_emotion(st.session_state.f_probs)
            if dom_val > 0:
                verdict_slot.markdown(f"### Final Verdict: **{dom_emo.upper()}** ({dom_val:.2f}%)")

    finally:
        cap.release()
        if v_path: os.unlink(v_path)