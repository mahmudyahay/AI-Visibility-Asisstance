import logging
import threading

import av
import cv2
import requests
import streamlit as st
import streamlit.components.v1 as components

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

try:
    import pytesseract
except ImportError:
    pytesseract = None

logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Accessibility Assistant", page_icon="👁️", layout="wide")

st.title("👁️ AI Accessibility Assistant")
st.write("Full pipeline: Object Awareness → Spatial Awareness → Scene/Text Understanding → Voice Assistant")

# ============================================================
# PHASE 1+2: YOLO object detection + spatial zones (continuous)
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()


def get_zone(center_x: float, frame_width: int) -> str:
    if center_x < frame_width / 3:
        return "LEFT"
    elif center_x < 2 * frame_width / 3:
        return "CENTER"
    return "RIGHT"


def get_distance_label(box_area: float, frame_area: float) -> str:
    ratio = box_area / frame_area
    if ratio > 0.15:
        return "NEAR"
    elif ratio > 0.04:
        return "MID"
    return "FAR"


class FrameStore:
    """
    Thread-safe hand-off between the WebRTC video thread (which calls
    video_frame_callback continuously) and the main Streamlit thread
    (which runs when a button is clicked). Phase 3/4 buttons read the
    most recent raw frame + detections from here rather than trying
    to re-run detection on demand.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.detections = []

    def update(self, frame, detections):
        with self.lock:
            self.frame = frame.copy()
            self.detections = detections

    def get(self):
        with self.lock:
            if self.frame is None:
                return None, []
            return self.frame.copy(), list(self.detections)


@st.cache_resource
def get_frame_store():
    return FrameStore()


frame_store = get_frame_store()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    raw_copy = img.copy()
    h, w = img.shape[:2]
    frame_area = w * h

    results = model(img, conf=0.40, verbose=False)
    boxes = results[0].boxes
    detections = []

    cv2.line(img, (w // 3, 0), (w // 3, h), (255, 255, 255), 1)
    cv2.line(img, (2 * w // 3, 0), (2 * w // 3, h), (255, 255, 255), 1)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label_name = model.names[cls_id]

            center_x = (x1 + x2) / 2
            box_area = (x2 - x1) * (y2 - y1)
            zone = get_zone(center_x, w)
            distance = get_distance_label(box_area, frame_area)

            detections.append(
                {"label": label_name, "conf": conf, "zone": zone, "distance": distance}
            )

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            text = f"{label_name} {conf:.2f} | {zone} {distance}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (int(x1), int(y1) - th - 8), (int(x1) + tw + 4, int(y1)), (0, 255, 0), -1)
            cv2.putText(img, text, (int(x1) + 2, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    frame_store.update(raw_copy, detections)
    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================
# TURN server (Cloudflare Realtime) -- required on Streamlit
# Community Cloud. Settings -> Secrets:
#   CLOUDFLARE_TURN_KEY_ID
#   CLOUDFLARE_TURN_KEY_API_TOKEN
# ============================================================

@st.cache_data(ttl=3600)
def get_ice_servers():
    key_id = st.secrets.get("CLOUDFLARE_TURN_KEY_ID")
    api_token = st.secrets.get("CLOUDFLARE_TURN_KEY_API_TOKEN")
    if not key_id or not api_token:
        return [{"urls": ["stun:stun.l.google.com:19302"]}]
    try:
        resp = requests.post(
            f"https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers",
            headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
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
    st.warning("No TURN credentials available -- the camera connection may time out on Streamlit Community Cloud.")

webrtc_streamer(
    key="accessibility-camera",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": ice_servers},
    async_processing=True,
)

st.divider()

# ============================================================
# PHASE 3: Text & Scene Understanding (on-demand)
# ============================================================

def read_text(img_bgr):
    if pytesseract is None:
        return None, "pytesseract package failed to import -- check requirements.txt was installed."
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return None, (
            "Tesseract OCR engine not found on this server. Make sure "
            "'tesseract-ocr' is in packages.txt and reboot the app."
        )
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()
        return text, None
    except Exception as e:
        return None, str(e)


def zone_phrase(zone: str) -> str:
    return {"LEFT": "on your left", "CENTER": "in front of you", "RIGHT": "on your right"}[zone]


def distance_phrase(distance: str) -> str:
    return {"NEAR": "close by", "MID": "", "FAR": "further away"}.get(distance, "")


def generate_description(detections):
    """
    Builds a plain-language spoken description directly from the
    YOLO detections -- no external API needed, so this works with
    zero cost and no API key. Keeps at most one mention per
    (object, zone) pair, picking the highest-confidence detection.
    """
    if not detections:
        return "I don't see anything clearly right now."

    best = {}
    for d in detections:
        key = (d["label"], d["zone"])
        if key not in best or d["conf"] > best[key]["conf"]:
            best[key] = d

    phrases = []
    for d in best.values():
        phrase = f"a {d['label']} {zone_phrase(d['zone'])}"
        dist = distance_phrase(d["distance"])
        if dist:
            phrase += f", {dist}"
        phrases.append(phrase)

    if len(phrases) == 1:
        return f"There is {phrases[0]}."
    return "There is " + ", ".join(phrases[:-1]) + f", and {phrases[-1]}."


def describe_scene(img_bgr, detections):
    """
    Free, local, rule-based description built from detections.
    (A previous version of this function called the paid Claude
    API for a more natural, richer description -- that's an
    optional upgrade path, not a requirement; see the comment
    below if you want to re-enable it later.)
    """
    return generate_description(detections), None

    # ---- Optional upgrade: AI-generated description ----
    # If you later get Anthropic API credit (or want to use a
    # provider with a real free tier, like Google's Gemini API),
    # you can swap the "return" above for a call that sends
    # img_bgr + detections to that provider instead, for a more
    # natural-sounding, context-aware description.


# ============================================================
# PHASE 4: Voice output (browser-side text-to-speech)
# ============================================================

def speak(text: str):
    safe_text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    components.html(
        f"""
        <script>
        window.speechSynthesis.cancel();
        const msg = new SpeechSynthesisUtterance("{safe_text}");
        window.speechSynthesis.speak(msg);
        </script>
        """,
        height=0,
    )


st.subheader("Ask the Assistant")
col1, col2 = st.columns(2)

with col1:
    if st.button("📖 Read Text Around Me", use_container_width=True):
        frame, _ = frame_store.get()
        if frame is None:
            st.warning("No camera frame yet -- start the camera above first.")
        else:
            with st.spinner("Reading text..."):
                text, err = read_text(frame)
            if err:
                st.error(err)
            elif text:
                st.success(text)
                speak(text)
            else:
                st.info("No readable text detected.")

with col2:
    if st.button("🗣️ Describe My Surroundings", use_container_width=True):
        frame, detections = frame_store.get()
        if frame is None:
            st.warning("No camera frame yet -- start the camera above first.")
        else:
            with st.spinner("Analyzing scene..."):
                description, err = describe_scene(frame, detections)
            if err:
                st.error(f"Could not generate description: {err}")
            else:
                st.success(description)
                speak(description)

st.divider()
st.subheader("Pipeline")
st.write(
    """
    **Continuous (every frame):** Camera → WebRTC → YOLO → Bounding boxes + LEFT/CENTER/RIGHT, NEAR/MID/FAR

    **On demand (button press):**
    - *Read Text*: current frame → OCR (Tesseract) → spoken aloud
    - *Describe Surroundings*: current frame + detections → Claude (vision) → spoken aloud
    """
)