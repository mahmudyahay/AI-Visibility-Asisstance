import av
import cv2
import numpy as np
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Accessibility Assistant",
    page_icon="👁️",
    layout="wide"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("👁️ AI Accessibility Assistant")

st.write(
    "Aspect 1 — Real-Time Object Awareness"
)

st.info(
    "Start the camera and point it at objects around you."
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Loads the YOLO model once and caches it across reruns.
    Wrapped in try/except so a failed download or corrupt
    weights file shows a clear error instead of crashing
    the whole app.
    """
    try:
        return YOLO("yolo11n.pt")
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        st.stop()


model = load_model()


# ============================================================
# VIDEO PROCESSOR
# ============================================================
# NOTE: Using a class-based VideoProcessor instead of a bare
# video_frame_callback function. This matters on Streamlit
# Community Cloud because a plain function can be re-created
# on every rerun and lose its closure over `model`; the class
# form is what streamlit-webrtc's own docs recommend for
# anything beyond a trivial demo, and it lets us guard against
# per-frame exceptions (a single bad frame or a transient CUDA/
# CPU hiccup would otherwise kill the whole WebRTC stream).

class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        try:
            results = self.model(img, conf=0.40, verbose=False)
            annotated_frame = results[0].plot()
        except Exception as e:
            # If inference fails on a frame, show the raw frame
            # with an error banner instead of crashing the stream.
            annotated_frame = img.copy()
            cv2.putText(
                annotated_frame,
                f"Detection error: {str(e)[:60]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# ============================================================
# WEBRTC CAMERA
# ============================================================
# NOTE: A STUN-only ICE configuration frequently fails to
# establish a connection when the viewer is behind certain
# corporate/mobile NATs or when the deployment host's own
# egress is restrictive (this is a common cause of "camera
# never connects" on Streamlit Community Cloud). Adding a
# public TURN server as a fallback fixes that for most users.
# Openrelay's free TURN servers are used here as a drop-in
# fallback; swap in your own TURN credentials for production.

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
)

webrtc_streamer(
    key="accessibility-camera",
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    rtc_configuration=RTC_CONFIGURATION,
    async_processing=True,
)


# ============================================================
# CURRENT CAPABILITY
# ============================================================

st.divider()

st.subheader("Current Capability")

st.write(
    """
    The system can currently:

    • Access the user's camera
    • Process the video continuously
    • Detect objects using YOLO
    • Draw bounding boxes around detected objects
    • Display object names and confidence scores
    """
)


# ============================================================
# PROJECT PIPELINE
# ============================================================

st.subheader("Accessibility Assistant Pipeline")

st.write(
    """
    Camera
        ↓
    Real-Time Video
        ↓
    YOLO Object Detection
        ↓
    Object Awareness
        ↓
    [Next: Spatial Awareness]
        ↓
    [Next: Environmental Understanding]
        ↓
    [Final: AI Accessibility Assistant]
    """
)