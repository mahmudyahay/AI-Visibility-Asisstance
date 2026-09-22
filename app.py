import base64
import time

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Accessibility Assistant",
    page_icon="👁️",
    layout="wide",
)


# ============================================================
# PAGE
# ============================================================

st.title("👁️ AI Accessibility Assistant")
st.subheader("Aspect 1 — Real-Time Object Awareness")

st.write(
    "Point your camera at an object. "
    "YOLO will detect it automatically."
)


# ============================================================
# LOAD YOLO ONCE
# ============================================================

@st.cache_resource
def load_yolo():

    model = YOLO("yolo11n.pt")

    return model


model = load_yolo()


# ============================================================
# CAMERA COMPONENT
# ============================================================

camera = st.components.v2.component(

    name="ai_accessibility_live_camera",

    html="""
    <div class="camera-wrapper">

        <video
            id="camera"
            autoplay
            playsinline
            muted
        ></video>

        <canvas id="canvas"></canvas>

        <div id="camera-status">
            Starting camera...
        </div>

    </div>
    """,

    css="""
    .camera-wrapper {

        width: 100%;
        max-width: 900px;

        margin: 0 auto;

        position: relative;

        background: black;

        border-radius: 14px;

        overflow: hidden;
    }


    #camera {

        width: 100%;

        display: block;
    }


    #canvas {

        display: none;
    }


    #camera-status {

        position: absolute;

        left: 12px;
        bottom: 12px;

        padding: 8px 12px;

        background: rgba(0,0,0,0.75);

        color: white;

        border-radius: 8px;

        font-size: 14px;

        font-family: Arial, sans-serif;
    }
    """,

    js="""
    export default function(component) {

        const {
            parentElement,
            setStateValue
        } = component;


        const video =
            parentElement.querySelector("#camera");

        const canvas =
            parentElement.querySelector("#canvas");

        const status =
            parentElement.querySelector("#camera-status");


        const context =
            canvas.getContext("2d");


        let stream = null;

        let timer = null;

        let running = true;


        // ====================================================
        // START CAMERA
        // ====================================================

        async function startCamera() {

            try {

                status.textContent =
                    "Starting rear camera...";


                // ------------------------------------------------
                // Rear/environment camera
                // ------------------------------------------------

                try {

                    stream =
                        await navigator.mediaDevices.getUserMedia({

                            audio: false,

                            video: {

                                facingMode: {
                                    ideal: "environment"
                                },

                                width: {
                                    ideal: 640
                                },

                                height: {
                                    ideal: 480
                                }
                            }
                        });

                }

                catch (rearError) {

                    console.log(
                        "Environment camera unavailable.",
                        rearError
                    );


                    // --------------------------------------------
                    // Desktop/front-camera fallback
                    // --------------------------------------------

                    stream =
                        await navigator.mediaDevices.getUserMedia({

                            audio: false,

                            video: {

                                width: {
                                    ideal: 640
                                },

                                height: {
                                    ideal: 480
                                }
                            }
                        });
                }


                // ------------------------------------------------
                // Attach camera
                // ------------------------------------------------

                video.srcObject = stream;

                await video.play();


                status.textContent =
                    "✓ Camera active";


                // ------------------------------------------------
                // Start frame transmission
                // ------------------------------------------------

                sendFrame();

            }

            catch (error) {

                console.error(
                    "Camera error:",
                    error
                );


                status.textContent =
                    "❌ Unable to access camera";
            }
        }


        // ====================================================
        // SEND FRAME TO PYTHON
        // ====================================================

        function sendFrame() {

            if (!running) {
                return;
            }


            if (
                video.readyState >=
                HTMLMediaElement.HAVE_CURRENT_DATA
            ) {

                // --------------------------------------------
                // Use actual video dimensions when available
                // --------------------------------------------

                const width =
                    video.videoWidth || 640;

                const height =
                    video.videoHeight || 480;


                // --------------------------------------------
                // Keep transmission manageable
                // --------------------------------------------

                const targetWidth = 640;

                const targetHeight =
                    Math.round(
                        height *
                        (targetWidth / width)
                    );


                canvas.width =
                    targetWidth;

                canvas.height =
                    targetHeight;


                // --------------------------------------------
                // Draw current camera frame
                // --------------------------------------------

                context.drawImage(

                    video,

                    0,
                    0,

                    targetWidth,
                    targetHeight
                );


                // --------------------------------------------
                // JPEG
                // --------------------------------------------

                const image =
                    canvas.toDataURL(
                        "image/jpeg",
                        0.75
                    );


                // --------------------------------------------
                // Send current frame to Python
                // --------------------------------------------

                setStateValue(
                    "frame",
                    image
                );
            }


            // ------------------------------------------------
            // 4 frames/second
            // ------------------------------------------------

            timer =
                setTimeout(
                    sendFrame,
                    250
                );
        }


        // ====================================================
        // START
        // ====================================================

        startCamera();


        // ====================================================
        // CLEANUP
        // ====================================================

        return () => {

            running = false;


            if (timer) {

                clearTimeout(timer);
            }


            if (stream) {

                stream
                    .getTracks()
                    .forEach(
                        track => track.stop()
                    );
            }
        };
    }
    """,
)


# ============================================================
# MOUNT CAMERA
# ============================================================

camera_result = camera(

    key="accessibility_camera",

    on_frame_change=lambda: None,
)


# ============================================================
# GET FRAME
# ============================================================

frame_data = camera_result.frame


if not frame_data:

    st.info(
        "Starting camera and waiting for first frame..."
    )

    st.stop()


# ============================================================
# DECODE FRAME
# ============================================================

try:

    if "," in frame_data:

        image_data = frame_data.split(
            ",",
            1
        )[1]

    else:

        image_data = frame_data


    image_bytes = base64.b64decode(
        image_data
    )


    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )


    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


except Exception as error:

    st.error(
        "Could not decode camera frame."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# VERIFY IMAGE
# ============================================================

if frame is None:

    st.error(
        "Camera frame arrived, but OpenCV "
        "could not decode the image."
    )

    st.stop()


# ============================================================
# YOLO INFERENCE
# ============================================================

try:

    results = model.predict(

        source=frame,

        imgsz=640,

        conf=0.15,

        iou=0.45,

        verbose=False,

        device="cpu",
    )


except Exception as error:

    st.error(
        "YOLO inference failed."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# GET YOLO RESULT
# ============================================================

result = results[0]


# ============================================================
# CREATE OUTPUT IMAGE
# ============================================================

output = frame.copy()


# ============================================================
# DETECTIONS
# ============================================================

detections = []


if result.boxes is not None:

    boxes = result.boxes


    for i in range(
        len(boxes)
    ):

        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        coordinates = (
            boxes.xyxy[i]
            .cpu()
            .numpy()
        )


        x1, y1, x2, y2 = (
            coordinates
            .astype(int)
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = float(
            boxes.conf[i]
            .cpu()
            .item()
        )


        # ----------------------------------------------------
        # Class
        # ----------------------------------------------------

        class_id = int(
            boxes.cls[i]
            .cpu()
            .item()
        )


        # ----------------------------------------------------
        # Name
        # ----------------------------------------------------

        name = str(
            model.names[class_id]
        )


        # ----------------------------------------------------
        # Save detection
        # ----------------------------------------------------

        detections.append({

            "name": name,

            "confidence": confidence,

            "x1": x1,

            "y1": y1,

            "x2": x2,

            "y2": y2,
        })


        # ====================================================
        # DRAW BOX
        # ====================================================

        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            (0, 255, 0),

            3,
        )


        # ====================================================
        # LABEL
        # ====================================================

        label = (
            f"{name} "
            f"{confidence:.0%}"
        )


        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(

            label,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            2,
        )


        label_top = max(
            0,
            y1 - text_height - 12
        )


        label_bottom = y1


        # ----------------------------------------------------
        # Label background
        # ----------------------------------------------------

        cv2.rectangle(

            output,

            (x1, label_top),

            (
                x1 + text_width + 12,
                label_bottom
            ),

            (0, 255, 0),

            -1,
        )


        # ----------------------------------------------------
        # Label text
        # ----------------------------------------------------

        cv2.putText(

            output,

            label,

            (
                x1 + 6,
                y1 - 6
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 0, 0),

            2,

            cv2.LINE_AA,
        )


# ============================================================
# DETECTION DISPLAY
# ============================================================

st.subheader("🔎 Live YOLO Detection")


output_rgb = cv2.cvtColor(
    output,
    cv2.COLOR_BGR2RGB
)


st.image(

    output_rgb,

    channels="RGB",

    use_container_width=True,
)


# ============================================================
# OBJECT LIST
# ============================================================

st.subheader("Objects Detected")


if len(detections) == 0:

    st.info(
        "No objects detected in the current frame."
    )

else:

    for index, detection in enumerate(
        detections,
        start=1
    ):

        st.write(
            f"**{index}. "
            f"{detection['name'].capitalize()}** "
            f"— "
            f"{detection['confidence']:.1%}"
        )


# ============================================================
# STATUS
# ============================================================

st.caption(
    f"Camera ✓  |  "
    f"Python ✓  |  "
    f"YOLO ✓  |  "
    f"Detections: {len(detections)}"
)