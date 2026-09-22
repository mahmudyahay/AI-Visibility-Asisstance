import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Accessibility Assistant",
    page_icon="👁️",
    layout="wide"
)


# --------------------------------------------------
# 2. APPLICATION TITLE
# --------------------------------------------------

st.title("👁️ AI Accessibility Assistant")

st.write(
    "Aspect 1: Real-time object detection using YOLO."
)

st.info(
    "Point your camera at objects around you. "
    "The system will detect and label them."
)


# --------------------------------------------------
# 3. LOAD YOLO MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = YOLO("yolo11n.pt")
    return model


model = load_model()


# --------------------------------------------------
# 4. CAMERA INPUT
# --------------------------------------------------

camera_image = st.camera_input(
    "Take a picture with your camera"
)


# --------------------------------------------------
# 5. PROCESS IMAGE
# --------------------------------------------------

if camera_image is not None:

    # Read the uploaded camera image
    image = Image.open(camera_image)

    # Convert PIL image to OpenCV format
    frame = cv2.cvtColor(
        __import__("numpy").array(image),
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------
    # 6. RUN YOLO DETECTION
    # --------------------------------------------------

    results = model(frame)


    # --------------------------------------------------
    # 7. DRAW DETECTIONS
    # --------------------------------------------------

    annotated_frame = results[0].plot()


    # --------------------------------------------------
    # 8. DISPLAY DETECTED IMAGE
    # --------------------------------------------------

    st.subheader("Detected Objects")

    st.image(
        cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        ),
        channels="RGB"
    )


    # --------------------------------------------------
    # 9. EXTRACT DETECTION INFORMATION
    # --------------------------------------------------

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:

        st.subheader("Objects Detected")

        detected_objects = []

        for box in boxes:

            # Class ID
            class_id = int(
                box.cls[0].item()
            )

            # Confidence
            confidence = float(
                box.conf[0].item()
            )

            # Object name
            object_name = model.names[class_id]

            detected_objects.append(
                (object_name, confidence)
            )


        # --------------------------------------------------
        # 10. DISPLAY OBJECTS ONE BY ONE
        # --------------------------------------------------

        for index, (object_name, confidence) in enumerate(
            detected_objects,
            start=1
        ):

            st.write(
                f"**{index}. {object_name.capitalize()}** "
                f"— {confidence:.2%}"
            )

    else:

        st.warning(
            "No objects were detected."
        )


# --------------------------------------------------
# 11. INFORMATION SECTION
# --------------------------------------------------

st.divider()

st.subheader("About Aspect 1")

st.write(
    """
    This is the first stage of the AI Accessibility Assistant.

    The system currently uses YOLO to identify objects
    visible in the camera image.

    Future versions will add:

    • Spatial awareness and distance estimation
    • OCR for reading text
    • Vision-language understanding
    • AI reasoning
    • Text-to-speech
    """
)
