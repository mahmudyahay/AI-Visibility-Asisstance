import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from camera_input_live import camera_input_live


st.set_page_config(
    page_title="AI Accessibility Assistant",
    page_icon="👁️",
    layout="wide",
)

st.title("👁️ AI Accessibility Assistant")
st.subheader("Aspect 1 — Real-Time Object Awareness")

st.write(
    "Point your camera at objects around you. "
    "YOLO will continuously detect what is visible."
)


@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()


# Live camera
image = camera_input_live()


if image is not None:

    # Convert captured image to NumPy
    frame = np.array(
        Image.open(image)
    )

    # RGB → BGR for OpenCV / YOLO
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    # YOLO detection
    results = model(
        frame,
        conf=0.40,
        verbose=False
    )

    result = results[0]

    # Draw bounding boxes
    annotated_frame = result.plot()

    # BGR → RGB for Streamlit
    annotated_frame = cv2.cvtColor(
        annotated_frame,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        annotated_frame,
        channels="RGB",
        use_container_width=True
    )

    # Object list
    st.subheader("Objects Detected")

    if result.boxes is not None and len(result.boxes) > 0:

        for index, box in enumerate(
            result.boxes,
            start=1
        ):

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            object_name = model.names[class_id]

            st.write(
                f"**{index}. "
                f"{object_name.capitalize()}** "
                f"— {confidence:.1%}"
            )

    else:
        st.info("No objects detected.")


st.divider()

st.caption(
    "Aspect 1: Camera → OpenCV → YOLO → "
    "Object Detection → Bounding Boxes"
)