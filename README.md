# AI Accessibility Assistant — Aspect 1

Real-time webcam object detection using YOLO, Streamlit, and streamlit-webrtc.

## Files
- `app.py` — application
- `requirements.txt` — Python dependencies
- `packages.txt` — Linux dependencies for Streamlit Community Cloud

## Streamlit Cloud
Use **Python 3.12** in Advanced settings. Streamlit Community Cloud currently defaults to Python 3.12.

The app uses the Ultralytics headless package intended for server/container environments, avoiding the normal OpenCV GUI package.

## Important
WebRTC camera access on a deployed app depends on browser/network connectivity. The app includes a public STUN server. Some networks may still require TURN; this is separate from the YOLO/OpenCV installation.
