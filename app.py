```python
import base64
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Accessibility Assistant",
    page_icon="👁️",
    layout="wide",
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("👁️ AI Accessibility Assistant")

st.subheader("Aspect 1 — Real-Time Object Awareness")

st.write(
    "Point your camera at your surroundings. "
    "The system continuously detects visible objects."
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()


# ============================================================
# CAMERA COMPONENT
# ============================================================

camera_component = st.components.v2.component(
    name="live_rear_camera",

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    html="""
    <div id="camera-wrapper">

        <video
            id="camera"
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

    # --------------------------------------------------------
    # CSS
    # --------------------------------------------------------

    css="""
    #camera-wrapper {
        width: 100%;
        max-width: 900px;
        margin: 0 auto;
        position: relative;
        overflow: hidden;
        border-radius: 14px;
        background: #000;
    }

    #camera {
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

        padding: 7px 12px;

        background: rgba(0, 0, 0, 0.70);
        color: white;

        border-radius: 8px;

        font-family: sans-serif;
        font-size: 14px;
    }
    """,

    # --------------------------------------------------------
    # JAVASCRIPT
    # --------------------------------------------------------

    js="""
    export default function(component) {

        const {
            parentElement,
            setStateValue
        } = component;


        const video =
            parentElement.querySelector("#camera");

        const canvas =
            parentElement.querySelector("#capture-canvas");

        const status =
            parentElement.querySelector("#camera-status");

        const context =
            canvas.getContext("2d");


        let stream = null;

        let running = true;


        // ====================================================
        // START CAMERA
        // ====================================================

        async function startCamera() {

            try {

                status.textContent =
                    "Requesting rear camera...";


                /*
                 * First attempt:
                 *
                 * explicitly request the
                 * environment/rear camera.
                 */

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
                                ideal: 15,
                                max: 20
                            }
                        }
                    });


                video.srcObject = stream;

                await video.play();


                status.textContent =
                    "Rear camera active";


                startFrameCapture();

            }

            catch (error) {

                console.warn(
                    "Rear camera unavailable:",
                    error
                );


                /*
                 * If the device does not have
                 * a rear camera, fall back to
                 * the available camera.
                 */

                try {

                    status.textContent =
                        "Rear camera unavailable. "
                        + "Trying available camera...";


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
                                    ideal: 15,
                                    max: 20
                                }
                            }
                        });


                    video.srcObject = stream;

                    await video.play();


                    status.textContent =
                        "Camera active";


                    startFrameCapture();

                }

                catch (fallbackError) {

                    console.error(
                        fallbackError
                    );

                    status.textContent =
                        "Camera access failed. "
                        + "Please allow camera permission.";
                }
            }
        }


        // ====================================================
        // CAPTURE FRAMES
        // ====================================================

        function startFrameCapture() {

            /*
             * We don't need to send every camera frame.
             *
             * Camera:
             * approximately 15 FPS
             *
             * YOLO:
             * approximately 5 FPS
             *
             * This reduces the amount of data
             * sent from browser to Python.
             */

            const capture = () => {

                if (!running) {
                    return;
                }


                if (
                    video.readyState >=
                    HTMLMediaElement.HAVE_CURRENT_DATA
                ) {

                    const width = 640;
                    const height = 480;


                    canvas.width = width;
                    canvas.height = height;


                    context.drawImage(
                        video,
                        0,
                        0,
                        width,
                        height
                    );


                    /*
                     * JPEG compression keeps
                     * frame size manageable.
                     */

                    const imageData =
                        canvas.toDataURL(
                            "image/jpeg",
                            0.65
                        );


                    /*
                     * Send the current frame
                     * to Python.
                     *
                     * IMPORTANT:
                     * "frame" is registered on
                     * the Python side below.
                     */

                    setStateValue(
                        "frame",
                        imageData
                    );
                }


                /*
                 * 200 ms = approximately 5 FPS.
                 */

                setTimeout(
                    capture,
                    200
                );
            };


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


            if (stream) {

                stream
                    .getTracks()
                    .forEach(
                        track => track.stop()
                    );
            }
        };
    }
    """
)


# ============================================================
# MOUNT CAMERA COMPONENT
# ============================================================

camera_result = camera_component(

    key="accessibility_camera",

    /*
     * Register the state value.
     *
     * This is required so Streamlit knows that
     * "frame" exists on camera_result.
     */

    default={
        "frame": None
    },

    on_frame_change=lambda: None,
)


# ============================================================
# GET CURRENT FRAME
# ============================================================

frame_data = camera_result.frame


# ============================================================
# PROCESS FRAME
# ============================================================

if frame_data:

    try:

        # ----------------------------------------------------
        # Base64 → bytes
        # ----------------------------------------------------

        encoded_image = frame_data.split(
            ",",
            1
        )[1]


        image_bytes = base64.b64decode(
            encoded_image
        )


        # ----------------------------------------------------
        # Bytes → NumPy array
        # ----------------------------------------------------

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )


        # ----------------------------------------------------
        # JPEG → OpenCV image
        # ----------------------------------------------------

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if frame is not None:

            # =================================================
            # YOLO OBJECT DETECTION
            # =================================================

            results = model(
                frame,

                # Keep 640 for better accuracy
                imgsz=640,

                # Detection confidence
                conf=0.35,

                verbose=False,
            )


            result = results[0]


            # =================================================
            # DRAW BOUNDING BOXES
            # =================================================

            annotated_frame = result.plot()


            # =================================================
            # BGR → RGB
            # =================================================

            annotated_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )


            # =================================================
            # DISPLAY DETECTION
            # =================================================

            st.subheader(
                "🔎 Live Detection"
            )


            st.image(
                annotated_frame,
                channels="RGB",
                use_container_width=True,
            )


            # =================================================
            # OBJECT LIST
            # =================================================

            st.subheader(
                "Objects Detected"
            )


            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                detected_objects = []


                for box in result.boxes:

                    # ----------------------------------------
                    # Class ID
                    # ----------------------------------------

                    class_id = int(
                        box.cls[0].item()
                    )


                    # ----------------------------------------
                    # Confidence
                    # ----------------------------------------

                    confidence = float(
                        box.conf[0].item()
                    )


                    # ----------------------------------------
                    # Object name
                    # ----------------------------------------

                    object_name = model.names[
                        class_id
                    ]


                    detected_objects.append(
                        (
                            object_name,
                            confidence
                        )
                    )


                # =================================================
                # PRINT OBJECTS
                # =================================================

                for index, (
                    name,
                    confidence
                ) in enumerate(
                    detected_objects,
                    start=1
                ):

                    st.write(
                        f"**{index}. "
                        f"{name.capitalize()}** "
                        f"— {confidence:.1%}"
                    )


            else:

                st.info(
                    "No objects detected."
                )


    except Exception as error:

        st.error(
            "Frame processing error:"
        )

        st.code(
            str(error)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Rear Camera → Live Frames → "
    "OpenCV → YOLO → Bounding Boxes → "
    "Object List"
)
```
