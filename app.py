import av
import cv2
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer


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
    return YOLO("yolo11n.pt")


model = load_model()


# ============================================================
# OBJECT DETECTION
# ============================================================

def detect_objects(frame):
    """
    Receives one video frame,
    runs YOLO,
    and returns the annotated frame.
    """

    results = model(
        frame,
        conf=0.40,
        verbose=False
    )

    annotated_frame = results[0].plot()

    return annotated_frame


# ============================================================
# VIDEO CALLBACK
# ============================================================

def video_frame_callback(frame: av.VideoFrame):

    # Convert WebRTC frame to OpenCV/Numpy format
    img = frame.to_ndarray(format="bgr24")

    # Run YOLO
    annotated_frame = detect_objects(img)

    # Return processed frame
    return av.VideoFrame.from_ndarray(
        annotated_frame,
        format="bgr24"
    )


# ============================================================
# WEBRTC CAMERA
# ============================================================

webrtc_streamer(
    key="accessibility-camera",

    video_frame_callback=video_frame_callback,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    },

    async_processing=True
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