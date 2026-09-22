import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image


st.set_page_config(
    page_title="AI Accessibility Assistant",
    page_icon="👁️",
    layout="wide"
)


st.title("👁️ AI Accessibility Assistant")
st.subheader("Aspect 1 — Object Awareness")

st.write(
    "Capture an image with your camera. "
    "YOLO will detect and label the objects in the image."
)


@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()


camera_image = st.camera_input(
    "📷 Capture your surroundings"
)


if camera_image is not None:

    image = Image.open(camera_image)

    frame = np.array(image)

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    results = model(
        frame,
        conf=0.40,
        verbose=False
    )

    result = results[0]

    annotated_frame = result.plot()

    annotated_frame = cv2.cvtColor(
        annotated_frame,
        cv2.COLOR_BGR2RGB
    )

    st.subheader("Detected Scene")

    st.image(
        annotated_frame,
        channels="RGB",
        use_container_width=True
    )

    st.subheader("Objects Detected")

    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:

        detected_objects = []

        for box in boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            object_name = model.names[class_id]

            detected_objects.append(
                (
                    object_name,
                    confidence
                )
            )

        for index, (name, confidence) in enumerate(
            detected_objects,
            start=1
        ):

            st.write(
                f"**{index}. {name.capitalize()}** "
                f"— {confidence:.1%}"
            )

    else:

        st.warning(
            "No objects were detected."
        )


st.divider()

st.subheader("Current Capability")

st.write(
    """
    Camera → Image Capture → OpenCV → YOLO
    → Object Detection → Bounding Boxes
    """
)