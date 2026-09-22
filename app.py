import os

import av
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

st.title("👁️ Real-Time Object Detection")

# ------------------------------------------------------------
# Load YOLO once and cache it across reruns.
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# ------------------------------------------------------------
# Runs on every incoming video frame: detect objects, draw
# boxes + labels, return the annotated frame.
# ------------------------------------------------------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    results = model(img, conf=0.40, verbose=False)
    annotated = results[0].plot()  # draws boxes + class names + confidence
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ------------------------------------------------------------
# TURN server: REQUIRED on Streamlit Community Cloud.
# Without one, the "Connection is taking longer than expected"
# error you saw will keep happening -- the platform blocks the
# direct browser-to-server path WebRTC normally uses.
#
# Add these two values in your app's Settings -> Secrets:
#   CLOUDFLARE_TURN_KEY_ID = "..."
#   CLOUDFLARE_TURN_KEY_API_TOKEN = "..."
# Get them free at the Cloudflare dashboard -> Realtime -> TURN.
# ------------------------------------------------------------
if "CLOUDFLARE_TURN_KEY_ID" in st.secrets:
    os.environ["CLOUDFLARE_TURN_KEY_ID"] = st.secrets["CLOUDFLARE_TURN_KEY_ID"]
    os.environ["CLOUDFLARE_TURN_KEY_API_TOKEN"] = st.secrets["CLOUDFLARE_TURN_KEY_API_TOKEN"]
else:
    st.warning(
        "No TURN credentials in Secrets -- the camera connection "
        "will likely time out on Streamlit Community Cloud."
    )

webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)