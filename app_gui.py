# streamlit_app.py
import streamlit as st
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime, time
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    from pytz import timezone as ZoneInfo  # fallback if pytz installed

import tensorflow as tf
from tensorflow.keras.models import load_model
import json

# -------------------------
# CONFIG - change if needed
# -------------------------
MODEL_PATH = "asl_cnn_model.h5"    # path to your saved model (change if necessary)
LABELS_PATH = "labels.json"        # optional file that maps class indices to label names
IMG_SIZE = 64                      # must match training target_size
TIMEZONE = "Asia/Kolkata"
START_TIME = time(18, 0)  # 6:00 PM
END_TIME = time(22, 0)    # 10:00 PM

# -------------------------
# Helpers
# -------------------------
@st.cache_resource
def load_asl_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at '{path}'. Please place your model there or change MODEL_PATH.")
        return None
    model = load_model(path)
    return model

def load_labels(labels_path):
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            labels = json.load(f)
        # If labels saved as dict {label: index} convert to ordered list
        if isinstance(labels, dict):
            # If stored as {label: index}, invert and sort by index
            try:
                inv = {v:k for k,v in labels.items()}
                labels_list = [inv[i] for i in range(len(inv))]
                return labels_list
            except Exception:
                # fallback
                return list(labels.keys())
        elif isinstance(labels, list):
            return labels
    # fallback default set (ensure this order matches your training)
    default = [
        'A','B','C','D','E','F','G','H','I','J','K','L','M',
        'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
        'nothing','space','delete'
    ]
    return default

def preprocess_pil_image(pil_img, img_size=IMG_SIZE):
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((img_size, img_size))
    arr = np.array(pil_img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dim
    return arr

def is_within_time_window(tz_name=TIMEZONE):
    try:
        now = datetime.now(ZoneInfo(tz_name))
    except Exception:
        # if zoneinfo not available, use naive now
        now = datetime.now()
    now_t = now.time()
    # supports ranges that do not cross midnight (our case 18:00-22:00)
    return START_TIME <= now_t <= END_TIME, now

# -------------------------
# App UI
# -------------------------
st.set_page_config(page_title="ASL Recognizer (6pm-10pm)", layout="centered")
st.title("ASL Recognizer — Image & Camera (Active 6:00 PM → 10:00 PM Asia/Kolkata)")
st.caption("Upload an image or use the camera to predict a sign. Predictions allowed only between 6 PM and 10 PM (Asia/Kolkata).")

# load model & labels
with st.spinner("Loading model..."):
    model = load_asl_model(MODEL_PATH)
labels = load_labels(LABELS_PATH)

if model is None:
    st.stop()

st.sidebar.header("Settings")
st.sidebar.markdown(f"**Model path:** `{MODEL_PATH}`")
st.sidebar.markdown(f"**Label file (optional):** `{LABELS_PATH}`")
st.sidebar.markdown(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
known_words_input = st.sidebar.text_input("Known words (comma-separated)", value="HELLO,THANKYOU,YES,NO")
known_words = [w.strip().upper() for w in known_words_input.split(",") if w.strip()]

# Session state for word-building
if "current_word" not in st.session_state:
    st.session_state.current_word = ""
if "last_pred" not in st.session_state:
    st.session_state.last_pred = ""

# Show current time & whether predictions are allowed
allowed, now_dt = is_within_time_window()
st.write(f"Current time ({TIMEZONE}): **{now_dt.strftime('%Y-%m-%d %H:%M:%S')}**")
if allowed:
    st.success("Predictions are **ENABLED** now (within 6:00 PM — 10:00 PM).")
else:
    st.warning("Predictions are **DISABLED** now. They work only between 6:00 PM and 10:00 PM (Asia/Kolkata).")

# Main input tabs
tab1, tab2 = st.tabs(["Image Upload", "Camera Snapshot"])

def predict_from_pil(pil_image):
    x = preprocess_pil_image(pil_image)
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=1)[0])
    prob = float(np.max(preds))
    label = labels[idx] if idx < len(labels) else str(idx)
    return label, prob, preds[0]

with tab1:
    st.subheader("Upload an image")
    uploaded_file = st.file_uploader("Choose an image (jpg/png)", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        if not allowed:
            st.info("Cannot run prediction: outside allowed time window (6 PM - 10 PM).")
        else:
            if st.button("Predict from uploaded image"):
                with st.spinner("Running prediction..."):
                    label, prob, raw = predict_from_pil(img)
                    st.session_state.last_pred = label
                    st.success(f"Predicted: **{label}**  (confidence: {prob*100:.2f}%)")
                    # show top-5
                    top5_idx = np.argsort(raw)[-5:][::-1]
                    top5 = [(labels[i] if i < len(labels) else str(i), float(raw[i])) for i in top5_idx]
                    st.write("Top predictions:")
                    for lab, p in top5:
                        st.write(f"- {lab}: {p*100:.2f}%")
                    # buttons to append or clear
                    col1, col2 = st.columns([1,1])
                    with col1:
                        if st.button("Append predicted letter to current word"):
                            # treat 'space' as space char, 'nothing' and 'delete' specially
                            if label.lower()=="space":
                                st.session_state.current_word += " "
                            elif label.lower()=="delete":
                                st.session_state.current_word = st.session_state.current_word[:-1]
                            elif label.lower()=="nothing":
                                # ignore
                                pass
                            else:
                                st.session_state.current_word += label.upper()
                    with col2:
                        if st.button("Clear current word"):
                            st.session_state.current_word = ""

with tab2:
    st.subheader("Camera snapshot (take a picture with your webcam)")
    cam_image = st.camera_input("Click to take snapshot")
    if cam_image is not None:
        img = Image.open(cam_image)
        st.image(img, caption="Camera Snapshot", use_column_width=True)
        if not allowed:
            st.info("Cannot run prediction: outside allowed time window (6 PM - 10 PM).")
        else:
            if st.button("Predict from camera snapshot"):
                with st.spinner("Running prediction..."):
                    label, prob, raw = predict_from_pil(img)
                    st.session_state.last_pred = label
                    st.success(f"Predicted: **{label}**  (confidence: {prob*100:.2f}%)")
                    top5_idx = np.argsort(raw)[-5:][::-1]
                    top5 = [(labels[i] if i < len(labels) else str(i), float(raw[i])) for i in top5_idx]
                    st.write("Top predictions:")
                    for lab, p in top5:
                        st.write(f"- {lab}: {p*100:.2f}%")
                    col1, col2 = st.columns([1,1])
                    with col1:
                        if st.button("Append predicted letter to current word (camera)"):
                            if label.lower()=="space":
                                st.session_state.current_word += " "
                            elif label.lower()=="delete":
                                st.session_state.current_word = st.session_state.current_word[:-1]
                            elif label.lower()=="nothing":
                                pass
                            else:
                                st.session_state.current_word += label.upper()
                    with col2:
                        if st.button("Clear current word (camera)"):
                            st.session_state.current_word = ""

# Word display / known words checking
st.markdown("---")
st.subheader("Assemble & check words")
st.write("Current built word (you can append predictions using the buttons):")
st.code(st.session_state.current_word or "(empty)")

col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    if st.button("Check if current word is a known word"):
        cw = st.session_state.current_word.strip().upper()
        if cw == "":
            st.info("Current word is empty.")
        elif cw in known_words:
            st.success(f"✅ Recognized known word: **{cw}**")
        else:
            st.error(f"Not a known word: **{cw}**")
with col_b:
    if st.button("Save current word to known words"):
        new = st.session_state.current_word.strip().upper()
        if new:
            if new not in known_words:
                known_words.append(new)
                st.sidebar.text_input("Known words (comma-separated)", value=",".join(known_words))
                st.success(f"Added `{new}` to known words (session only).")
            else:
                st.info(f"`{new}` already in known words.")
        else:
            st.info("Current word empty — nothing to add.")
with col_c:
    if st.button("Reset current word"):
        st.session_state.current_word = ""

st.markdown("**Last single prediction:**")
st.write(st.session_state.last_pred or "No prediction yet")

st.markdown("---")
st.caption("Notes: \n- Make sure `IMG_SIZE` equals the size you used during training. \n- For best results ensure `labels.json` (mapping index→label) matches the order used when the model was trained. \n- This app uses camera snapshots (one-shot). For continuous real-time prediction you'd typically run an OpenCV loop — which is outside Streamlit's normal flow; snapshots work well for demo & manual real-time capture.")
