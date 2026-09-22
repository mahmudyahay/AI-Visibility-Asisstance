import base64
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
# TITLE
# ============================================================

st.title("👁️ AI Accessibility Assistant")
st.subheader("Aspect 1 — Real-Time Object Awareness")

st.write(
    "Point your camera at your surroundings. "
    "YOLO will detect objects in view."
)


# ============================================================
# LOAD YOLO
# ============================================================

@st.cache_resource
def load_model():
    model = YOLO("yolo11n.pt")
    return model


model = load_model()


# ============================================================
# CAMERA COMPONENT
# ============================================================

camera_component = st.components.v2.component(
    name="live_rear_camera_v3",

    html="""
    <div id="camera-container">

        <video
            id="camera"
            autoplay
            playsinline
            muted
        ></video>

        <canvas
            id="canvas"
        ></canvas>

        <div id="status">
            Starting camera...
        </div>

    </div>
    """,

    css="""
    #camera-container {
        width: 100%;
        max-width: 900px;
        margin: 0 auto;
        position: relative;
        background: #000;
        border-radius: 12px;
        overflow: hidden;
    }

    #camera {
        width: 100%;
        height: auto;
        display: block;
    }

    #canvas {
        display: none;
    }

    #status {
        position: absolute;
        left: 12px;
        bottom: 12px;

        background: rgba(0, 0, 0, 0.75);
        color: white;

        padding: 7px 12px;
        border-radius: 7px;

        font-family: Arial, sans-serif;
        font-size: 14px;
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
            parentElement.querySelector("#status");

        const ctx =
            canvas.getContext("2d");


        let stream = null;

        let running = true;

        let captureTimer = null;


        // ====================================================
        // START CAMERA
        // ====================================================

        async function startCamera() {

            try {

                status.textContent =
                    "Requesting rear camera...";


                // --------------------------------------------
                // Try rear/environment camera first
                // --------------------------------------------

                stream =
                    await navigator.mediaDevices.getUserMedia({

                        audio: false,

                        video: {

                            facingMode: {
                                exact: "environment"
                            },

                            width: {
                                ideal: 640
                            },

                            height: {
                                ideal: 480
                            },

                            frameRate: {
                                ideal: 10,
                                max: 15
                            }
                        }
                    });


                video.srcObject = stream;

                await video.play();


                status.textContent =
                    "✓ Rear camera active";


                startCapture();

            }

            catch (error) {

                console.log(
                    "Rear camera unavailable:",
                    error
                );


                // --------------------------------------------
                // Fallback camera
                // --------------------------------------------

                try {

                    status.textContent =
                        "Trying available camera...";


                    stream =
                        await navigator.mediaDevices.getUserMedia({

                            audio: false,

                            video: {

                                width: {
                                    ideal: 640
                                },

                                height: {
                                    ideal: 480
                                },

                                frameRate: {
                                    ideal: 10,
                                    max: 15
                                }
                            }
                        });


                    video.srcObject = stream;

                    await video.play();


                    status.textContent =
                        "✓ Camera active";


                    startCapture();

                }

                catch (fallbackError) {

                    console.error(
                        fallbackError
                    );

                    status.textContent =
                        "❌ Camera permission denied";
                }
            }
        }


        // ====================================================
        // CAPTURE CAMERA FRAMES
        // ====================================================

        function startCapture() {

            function capture() {

                if (!running) {
                    return;
                }


                if (
                    video.readyState >=
                    HTMLMediaElement.HAVE_CURRENT_DATA
                ) {

                    // ----------------------------------------
                    // Fixed frame size
                    // ----------------------------------------

                    canvas.width = 640;
                    canvas.height = 480;


                    // ----------------------------------------
                    // Copy video frame to canvas
                    // ----------------------------------------

                    ctx.drawImage(
                        video,
                        0,
                        0,
                        640,
                        480
                    );


                    // ----------------------------------------
                    // Convert to JPEG
                    // ----------------------------------------

                    const image =
                        canvas.toDataURL(
                            "image/jpeg",
                            0.70
                        );


                    // ----------------------------------------
                    // IMPORTANT
                    //
                    // This is continuous camera STATE,
                    // not a one-time button/event.
                    // ----------------------------------------

                    setStateValue(
                        "frame",
                        image
                    );
                }


                // --------------------------------------------
                // Approximately 4 FPS
                //
                // We intentionally start slower so that
                // Python has enough time to run YOLO.
                // --------------------------------------------

                captureTimer =
                    setTimeout(
                        capture,
                        250
                    );
            }


            capture();
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


            if (captureTimer) {

                clearTimeout(
                    captureTimer
                );
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
# MOUNT CAMERA COMPONENT
# ============================================================

camera_result = camera_component(

    key="accessibility_camera",

    on_frame_change=lambda: None,
)


# ============================================================
# GET CURRENT FRAME
# ============================================================

frame_data = camera_result.frame


# ============================================================
# DEBUG: DID PYTHON RECEIVE THE FRAME?
# ============================================================

if not frame_data:

    st.info(
        "Waiting for the first camera frame..."
    )

    st.stop()


st.success(
    "🟢 Camera frame received by Python"
)


# ============================================================
# DECODE BASE64 IMAGE
# ============================================================

try:

    # Remove:
    # data:image/jpeg;base64,...

    if "," in frame_data:

        encoded_image = frame_data.split(
            ",",
            1
        )[1]

    else:

        encoded_image = frame_data


    # Base64 → bytes

    image_bytes = base64.b64decode(
        encoded_image
    )


    # Bytes → NumPy array

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )


    # JPEG → OpenCV image

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if frame is None:

        st.error(
            "Python received the camera frame, "
            "but OpenCV could not decode it."
        )

        st.stop()


except Exception as error:

    st.error(
        "Camera frame decoding failed."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# SHOW RAW FRAME
# ============================================================

st.caption(
    f"Frame received successfully: "
    f"{frame.shape[1]} × {frame.shape[0]}"
)


# ============================================================
# YOLO DETECTION
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
# GET RESULT
# ============================================================

result = results[0]


# ============================================================
# DETECTION INFORMATION
# ============================================================

if result.boxes is None:

    detection_count = 0

else:

    detection_count = len(
        result.boxes
    )


st.caption(
    f"YOLO detections: {detection_count}"
)


# ============================================================
# DRAW DETECTIONS MANUALLY
# ============================================================

annotated_frame = frame.copy()

detected_objects = []


if (
    result.boxes is not None
    and len(result.boxes) > 0
):

    for box in result.boxes:

        # -----------------------------------------------
        # Coordinates
        # -----------------------------------------------

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )


        # -----------------------------------------------
        # Confidence
        # -----------------------------------------------

        confidence = float(
            box.conf[0]
            .cpu()
            .item()
        )


        # -----------------------------------------------
        # Class ID
        # -----------------------------------------------

        class_id = int(
            box.cls[0]
            .cpu()
            .item()
        )


        # -----------------------------------------------
        # Class name
        # -----------------------------------------------

        object_name = model.names[
            class_id
        ]


        # -----------------------------------------------
        # Store detection
        # -----------------------------------------------

        detected_objects.append(
            {
                "name": object_name,
                "confidence": confidence,
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            }
        )


        # -----------------------------------------------
        # Draw bounding box
        # -----------------------------------------------

        cv2.rectangle(

            annotated_frame,

            (x1, y1),

            (x2, y2),

            (0, 255, 0),

            3,
        )


        # -----------------------------------------------
        # Label
        # -----------------------------------------------

        label = (
            f"{object_name} "
            f"{confidence:.0%}"
        )


        # -----------------------------------------------
        # Label size
        # -----------------------------------------------

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(

            label,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            2,
        )


        # -----------------------------------------------
        # Label background
        # -----------------------------------------------

        label_y1 = max(
            0,
            y1 - text_height - baseline - 8
        )


        label_y2 = y1


        cv2.rectangle(

            annotated_frame,

            (x1, label_y1),

            (
                x1 + text_width + 10,
                label_y2
            ),

            (0, 255, 0),

            -1,
        )


        # -----------------------------------------------
        # Label text
        # -----------------------------------------------

        cv2.putText(

            annotated_frame,

            label,

            (
                x1 + 5,
                y1 - 6
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 0, 0),

            2,

            cv2.LINE_AA,
        )


# ============================================================
# DISPLAY DETECTED FRAME
# ============================================================

st.subheader("🔎 YOLO Detection")

annotated_rgb = cv2.cvtColor(
    annotated_frame,
    cv2.COLOR_BGR2RGB
)

st.image(
    annotated_rgb,
    channels="RGB",
    use_container_width=True,
)


# ============================================================
# OBJECT LIST
# ============================================================

st.subheader("Objects Detected")


if detected_objects:

    for index, detection in enumerate(
        detected_objects,
        start=1
    ):

        name = detection["name"]

        confidence = detection[
            "confidence"
        ]


        st.write(
            f"**{index}. "
            f"{name.capitalize()}** "
            f"— {confidence:.1%}"
        )

else:

    st.info(
        "No objects detected in this frame."
    )


# ============================================================
# DEBUG INFORMATION
# ============================================================

with st.expander("🔧 Detection Debug Information"):

    st.write(
        "Frame received:",
        True
    )

    st.write(
        "Frame shape:",
        frame.shape
    )

    st.write(
        "YOLO model:",
        "yolo11n.pt"
    )

    st.write(
        "Confidence threshold:",
        0.15
    )

    st.write(
        "Detection count:",
        detection_count
    )

    if detected_objects:

        st.write(
            "Detected classes:"
        )

        for detection in detected_objects:

            st.write(
                f"- {detection['name']} "
                f"({detection['confidence']:.1%})"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Rear Camera → Python Frame → "
    "YOLO → Bounding Boxes → Object List"
)