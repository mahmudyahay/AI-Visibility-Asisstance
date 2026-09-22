import logging
import os

import av
import requests
import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

logger = logging.getLogger(__name__)

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
# TURN server: REQUIRED on Streamlit Community Cloud, or you
# get exactly the "Connection is taking longer than expected"
# error. Cloudflare does NOT auto-inject credentials -- we have
# to call their API ourselves to mint short-lived TURN creds.
#
# Setup (one time):
#   1. Cloudflare dashboard -> Realtime -> TURN -> Create TURN Key
#      (NOT the general "API Tokens" page -- this is a separate,
#      TURN-specific key/token pair).
#   2. In Streamlit Cloud: Settings -> Secrets, add:
#        CLOUDFLARE_TURN_KEY_ID = "your-turn-key-id"
#        CLOUDFLARE_TURN_KEY_API_TOKEN = "your-turn-api-token"
#
# Credentials are cached for 1 hour (well under the 24h TTL we
# request) so we're not hitting Cloudflare's API on every rerun.
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_ice_servers():
    key_id = st.secrets.get("CLOUDFLARE_TURN_KEY_ID")
    api_token = st.secrets.get("CLOUDFLARE_TURN_KEY_API_TOKEN")

    if not key_id or not api_token:
        logger.warning("Cloudflare TURN credentials not set; falling back to STUN only.")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    try:
        resp = requests.post(
            f"https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers",
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            },
            json={"ttl": 86400},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["iceServers"]
    except Exception as e:
        logger.warning(f"Failed to fetch Cloudflare TURN credentials: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]


ice_servers = get_ice_servers()

if len(ice_servers) == 1 and "turn" not in str(ice_servers).lower():
    st.warning(
        "No TURN credentials available -- the camera connection "
        "will likely time out on Streamlit Community Cloud."
    )

webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": ice_servers},
    async_processing=True,
)