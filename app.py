import av
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(page_title="AI Accessibility Assistant", page_icon="👁️", layout="wide")

st.title("👁️ AI Accessibility Assistant")
st.caption("Aspect 1 — Real-Time Object Awareness")
st.write("Start the camera and point it at objects around you.")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    results = model(image, conf=0.40, verbose=False)
    annotated = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="accessibility-camera",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=video_frame_callback,
    async_processing=True,
)

st.divider()
st.subheader("Aspect 1 complete")
st.write("The app processes the live camera feed with YOLO and displays detected objects with bounding boxes, names, and confidence scores.")
st.caption("Later aspects will add spatial awareness, OCR, scene understanding, reasoning, and speech.")
