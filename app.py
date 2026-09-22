import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Accessibility Assistant",
    page_icon="👁️",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("👁️ AI Accessibility Assistant")

st.write(
    "Aspect 1: Real-time object detection using YOLO."
)

st.info(
    "Start the camera and point it at objects around you."
)


# --------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = YOLO("yolo11n.pt")
    return model


model = load_model()


# --------------------------------------------------
# VIDEO PROCESSOR
# --------------------------------------------------

class ObjectDetectionProcessor(VideoProcessorBase):

    def recv(self, frame):

        # Convert video frame to OpenCV format
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO detection
        results = model(img)

        # Draw bounding boxes
        annotated_frame = results[0].plot()

        # Return processed frame
        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )


# --------------------------------------------------
# START CAMERA
# --------------------------------------------------

webrtc_streamer(
    key="object-detection",
    video_processor_factory=ObjectDetectionProcessor
)


# --------------------------------------------------
# INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("Current Stage")

st.write(
    """
    The system currently performs real-time object detection.

    Camera
        ↓
    Video Frames
        ↓
    YOLO
        ↓
    Object Detection
        ↓
    Bounding Boxes
    """
)

st.subheader("Next Stages")

st.write(
    """
    • Spatial awareness
    • Distance estimation
    • OCR
    • Scene understanding
    • AI reasoning
    • Text-to-speech
    """
)