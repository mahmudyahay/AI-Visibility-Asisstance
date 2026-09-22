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
# PAGE
# ============================================================

st.title("👁️ AI Accessibility Assistant")
st.subheader("Aspect 1 — Real-Time Object Awareness")

st.write(
    "Point your camera at your surroundings. "
    "The AI will detect visible objects."
)


# ============================================================
# YOLO
# ============================================================

@st.cache_resource
def load_model():

    return YOLO("yolo11n.pt")


model = load_model()


# ============================================================
# CAMERA COMPONENT
# ============================================================

camera_component = st.components.v2.component(

    name="stable_accessibility_camera",

    html="""
    <div id="camera-root">

        <video
            id="camera-video"
            autoplay
            playsinline
            muted
        ></video>

        <canvas
            id="capture-canvas"
        ></canvas>

        <div id="camera-status">
            Starting camera...
        </div>

    </div>
    """,

    css="""
    #camera-root {

        width: 100%;
        max-width: 900px;

        margin: 0 auto;

        position: relative;

        background: #000;

        border-radius: 14px;

        overflow: hidden;
    }


    #camera-video {

        width: 100%;

        height: auto;

        display: block;

        background: #000;
    }


    #capture-canvas {

        display: none;
    }


    #camera-status {

        position: absolute;

        left: 12px;

        bottom: 12px;

        padding: 8px 12px;

        border-radius: 8px;

        background: rgba(0, 0, 0, 0.75);

        color: white;

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


        // ====================================================
        // IMPORTANT:
        //
        // Streamlit can rerun Python when a frame is sent.
        //
        // We therefore store the camera controller directly
        // on the DOM element.
        //
        // If the component runs again, we DO NOT create
        // another camera or another capture loop.
        // ====================================================

        if (parentElement.__cameraController) {

            return;
        }


        const video =
            parentElement.querySelector(
                "#camera-video"
            );


        const canvas =
            parentElement.querySelector(
                "#capture-canvas"
            );


        const status =
            parentElement.querySelector(
                "#camera-status"
            );


        const context =
            canvas.getContext("2d");


        // ====================================================
        // CAMERA CONTROLLER
        // ====================================================

        const controller = {

            stream: null,

            timer: null,

            running: true,

            busy: false,

            lastFrameTime: 0,

        };


        parentElement.__cameraController =
            controller;


        // ====================================================
        // CAMERA START
        // ====================================================

        async function startCamera() {

            if (!controller.running) {
                return;
            }


            status.textContent =
                "Starting rear camera...";


            try {

                // ------------------------------------------------
                // Prefer environment/rear camera
                // ------------------------------------------------

                controller.stream =
                    await navigator.mediaDevices
                        .getUserMedia({

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
                                },

                                frameRate: {
                                    ideal: 15,

                                    max: 20
                                }
                            }
                        });


            }

            catch (error) {

                console.log(
                    "Rear camera request failed:",
                    error
                );


                try {

                    // --------------------------------------------
                    // Fallback
                    // --------------------------------------------

                    controller.stream =
                        await navigator.mediaDevices
                            .getUserMedia({

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

                catch (fallbackError) {

                    console.error(
                        "Camera unavailable:",
                        fallbackError
                    );


                    status.textContent =
                        "❌ Camera unavailable";

                    return;
                }
            }


            // ====================================================
            // CONNECT STREAM
            // ====================================================

            video.srcObject =
                controller.stream;


            try {

                await video.play();

            }

            catch (playError) {

                console.error(
                    "Video play error:",
                    playError
                );

            }


            status.textContent =
                "✓ Camera active";


            // ====================================================
            // WAIT UNTIL CAMERA HAS REAL DIMENSIONS
            // ====================================================

            waitForVideo();
        }


        // ====================================================
        // WAIT FOR VIDEO
        // ====================================================

        function waitForVideo() {

            if (!controller.running) {
                return;
            }


            if (
                video.videoWidth > 0 &&
                video.videoHeight > 0
            ) {

                captureFrame();

                return;
            }


            setTimeout(
                waitForVideo,
                300
            );
        }


        // ====================================================
        // CAPTURE ONE FRAME
        // ====================================================

        function captureFrame() {

            if (!controller.running) {
                return;
            }


            // ------------------------------------------------
            // Don't send another frame while one is already
            // being processed by Streamlit.
            // ------------------------------------------------

            if (controller.busy) {

                controller.timer =
                    setTimeout(
                        captureFrame,
                        500
                    );

                return;
            }


            if (
                video.readyState <
                HTMLMediaElement.HAVE_CURRENT_DATA
            ) {

                controller.timer =
                    setTimeout(
                        captureFrame,
                        500
                    );

                return;
            }


            // =================================================
            // FRAME SIZE
            // =================================================

            const sourceWidth =
                video.videoWidth || 640;


            const sourceHeight =
                video.videoHeight || 480;


            const targetWidth = 640;


            const targetHeight =
                Math.round(

                    sourceHeight *
                    (
                        targetWidth /
                        sourceWidth
                    )
                );


            canvas.width =
                targetWidth;


            canvas.height =
                targetHeight;


            // =================================================
            // DRAW CAMERA FRAME
            // =================================================

            context.drawImage(

                video,

                0,
                0,

                targetWidth,
                targetHeight
            );


            // =================================================
            // JPEG
            // =================================================

            const image =
                canvas.toDataURL(
                    "image/jpeg",
                    0.72
                );


            // =================================================
            // SEND TO PYTHON
            // =================================================

            controller.busy = true;


            setStateValue(
                "frame",
                image
            );


            controller.lastFrameTime =
                Date.now();


            // =================================================
            // NEXT FRAME
            //
            // 1 frame every ~1.5 seconds.
            //
            // That's ~6–7 frames / 10 seconds.
            //
            // More than your minimum of 3 / 10 seconds,
            // while keeping Streamlit/YOLO workload reasonable.
            // =================================================

            controller.timer =
                setTimeout(

                    () => {

                        controller.busy =
                            false;

                        captureFrame();

                    },

                    1500
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

            controller.running =
                false;


            if (controller.timer) {

                clearTimeout(
                    controller.timer
                );
            }


            if (controller.stream) {

                controller.stream
                    .getTracks()
                    .forEach(
                        track => track.stop()
                    );
            }


            delete parentElement.__cameraController;
        };
    }
    """,
)


# ============================================================
# MOUNT COMPONENT
# ============================================================

camera_result = camera_component(

    key="accessibility_camera",

    default={
        "frame": None
    },

    on_frame_change=lambda: None,
)


# ============================================================
# GET FRAME
# ============================================================

frame_data = camera_result.frame


if not frame_data:

    st.info(
        "📷 Starting camera and waiting for first frame..."
    )

    st.stop()


# ============================================================
# CAMERA FRAME RECEIVED
# ============================================================

st.success(
    "🟢 Camera frame received — running YOLO..."
)


# ============================================================
# DECODE JPEG
# ============================================================

try:

    if "," in frame_data:

        encoded =
            frame_data.split(
                ",",
                1
            )[1]

    else:

        encoded = frame_data


    image_bytes =
        base64.b64decode(
            encoded
        )


    image_array =
        np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )


    frame =
        cv2.imdecode(
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
# VALIDATE FRAME
# ============================================================

if frame is None:

    st.error(
        "The camera frame arrived, "
        "but OpenCV could not decode it."
    )

    st.stop()


# ============================================================
# YOLO
# ============================================================

try:

    results = model.predict(

        source=frame,

        imgsz=640,

        conf=0.20,

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
# RESULT
# ============================================================

result = results[0]


# ============================================================
# OUTPUT IMAGE
# ============================================================

annotated =
    frame.copy()


detections = []


# ============================================================
# PROCESS BOXES
# ============================================================

if (
    result.boxes is not None
    and len(result.boxes) > 0
):

    for box in result.boxes:

        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        x1, y1, x2, y2 = (

            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = float(

            box.conf[0]
            .cpu()
            .item()
        )


        # ----------------------------------------------------
        # Class
        # ----------------------------------------------------

        class_id = int(

            box.cls[0]
            .cpu()
            .item()
        )


        # ----------------------------------------------------
        # Name
        # ----------------------------------------------------

        name =
            model.names[
                class_id
            ]


        detections.append({

            "name": name,

            "confidence":
                confidence,

            "box": (
                x1,
                y1,
                x2,
                y2
            )
        })


        # ====================================================
        # DRAW BOX
        # ====================================================

        cv2.rectangle(

            annotated,

            (x1, y1),

            (x2, y2),

            (0, 255, 0),

            3
        )


        # ====================================================
        # LABEL
        # ====================================================

        label =
            f"{name} {confidence:.0%}"


        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(

            label,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            2
        )


        label_top =
            max(
                0,
                y1 -
                text_height -
                12
            )


        cv2.rectangle(

            annotated,

            (x1, label_top),

            (
                x1 +
                text_width +
                12,

                y1
            ),

            (0, 255, 0),

            -1
        )


        cv2.putText(

            annotated,

            label,

            (
                x1 + 6,

                y1 - 6
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 0, 0),

            2,

            cv2.LINE_AA
        )


# ============================================================
# DISPLAY RESULT
# ============================================================

st.subheader(
    "🔎 YOLO Detection"
)


annotated_rgb =
    cv2.cvtColor(

        annotated,

        cv2.COLOR_BGR2RGB
    )


st.image(

    annotated_rgb,

    channels="RGB",

    use_container_width=True
)


# ============================================================
# OBJECTS
# ============================================================

st.subheader(
    "Objects Detected"
)


if detections:

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

else:

    st.info(
        "No recognizable objects "
        "were detected in this frame."
    )


# ============================================================
# STATUS
# ============================================================

st.caption(

    f"Camera ✓  |  "
    f"Python ✓  |  "
    f"YOLO ✓  |  "
    f"{len(detections)} object(s) detected"
)