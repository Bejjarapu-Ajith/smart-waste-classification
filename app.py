import os
import time
from pathlib import Path

import cv2
import torch
import streamlit as st
from ultralytics import YOLO

DEFAULT_MODEL_PATH = r"D:\DryWetProject\roboflow_dataset\runs\detect\DryWetWaste_v8n\weights\best.pt"

st.set_page_config(page_title="Dry vs Wet Waste – YOLO", page_icon="♻️", layout="wide")

@st.cache_resource
def load_model(weights_path: str):
    m = YOLO(weights_path)
    # choose GPU if available
    if torch.cuda.is_available():
        m.to("cuda:0")
    else:
        m.to("cpu")
    return m

def list_cameras(max_index=5):
    """Return indices of available cameras (0..max_index)."""
    indices = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            indices.append(i)
            cap.release()
    return indices or [0]

# sidebar controls 
st.sidebar.header("⚙️ Controls")
model_path = st.sidebar.text_input("Model .pt path", DEFAULT_MODEL_PATH)
conf = st.sidebar.slider("Confidence", 0.1, 0.9, 0.35, 0.05)
iou = st.sidebar.slider("IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
imgsz = st.sidebar.select_slider("Image size", options=[320, 480, 640, 800], value=640)
max_fps = st.sidebar.slider("Max FPS (limit CPU/GPU heat)", 5, 60, 20, 1)
draw_labels = st.sidebar.checkbox("Show labels", True)
draw_conf = st.sidebar.checkbox("Show confidence", True)

# camera selector
available_cams = list_cameras(3)
cam_index = st.sidebar.selectbox("Webcam", available_cams, index=0)

col_l, col_r = st.columns([2, 1])
col_l.title("♻️ Live Dry vs Wet Waste Detection (YOLOv8)")
col_l.caption("Your camera feed starts when you press **Start**. The model draws boxes for *dry* and *wet*.")

# load model
if not Path(model_path).exists():
    st.error(f"Model not found: {model_path}")
    st.stop()

model = load_model(model_path)

# session state for start/stop
if "running" not in st.session_state:
    st.session_state.running = False

start = col_r.button("▶️ Start", type="primary", use_container_width=True)
stop = col_r.button("⏹ Stop", use_container_width=True)

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# video output container
frame_placeholder = col_l.empty()
stats_placeholder = col_r.empty()

def annotate(frame_bgr, results):
    """Draw YOLO results on the frame using Ultralytics' built-in plot()."""
    plot_args = {"labels": draw_labels, "conf": draw_conf, "line_width": 2}
    return results[0].plot(**plot_args)  # returns BGR image

# main loop
if st.session_state.running:
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    # Camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = 0.0
    frame_count = 0
    t0 = time.time()

    try:
        while st.session_state.running:
            ok, frame = cap.read()
            if not ok:
                st.warning("Couldn't read frame from camera. Try another webcam index.")
                break

            # FPS
            now = time.time()
            if now - prev_time < 1.0 / max_fps:
                time.sleep(max(0, 1.0 / max_fps - (now - prev_time)))
            prev_time = time.time()

            # predict (BGR frame)
            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False,
                device=0 if torch.cuda.is_available() else "cpu",
            )

            out_bgr = annotate(frame, results)

            # Streamlit expects RGB
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(out_rgb, channels="RGB", use_container_width=True)

            frame_count += 1
            elapsed = now - t0 if now - t0 > 0 else 1e-6
            stats_placeholder.metric("FPS", f"{frame_count/elapsed:0.1f}")

    finally:
        cap.release()
        frame_placeholder.empty()
        stats_placeholder.empty()
        st.session_state.running = False
else:
    st.info("Press **Start** to open the webcam and begin detection.")
