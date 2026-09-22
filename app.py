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
    "The system detects objects in real time."
)


# ============================================================
# LOAD YOLO
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()


# ============================================================
# CAMERA COMPONENT
# ============================================================

camera_component = st.components.v2.component(
    name="live_camera",

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

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
            Starting rear camera...
        </div>

    </div>
    """,

    # --------------------------------------------------------
    # CSS
    # --------------------------------------------------------

    css="""
    #camera-container {
        width: 100%;
        max-width: 900px;
        margin: auto;
        position: relative;
        background: black;
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
        bottom: 10px;
        left: 10px;

        background: rgba(0, 0, 0, 0.7);
        color: white;

        padding: 6px 10px;
        border-radius: 6px;

        font-family: Arial, sans-serif;
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
            parentElement.querySelector("#canvas");

        const status =
            parentElement.querySelector("#status");

        const ctx =
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
                 * Request the rear/environment camera.
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
                    "✓ Rear camera active";


                startCapture();

            }

            catch (error) {

                console.log(
                    "Rear camera unavailable:",
                    error
                );


                /*
                 * Fallback for laptops or devices
                 * without an environment camera.
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
        // CAPTURE FRAMES
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

                    /*
                     * Keep frames at 640 × 480.
                     */

                    canvas.width = 640;

                    canvas.height = 480;


                    ctx.drawImage(
                        video,
                        0,
                        0,
                        640,
                        480
                    );


                    /*
                     * Compress frame to JPEG.
                     */

                    const image =
                        canvas.toDataURL(
                            "image/jpeg",
                            0.65
                        );


                    /*
                     * Send frame to Python.
                     */

                    setStateValue(
                        "frame",
                        image
                    );
                }


                /*
                 * Approximately 5 frames per second.
                 */

                setTimeout(
                    capture,
                    200
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
# MOUNT CAMERA
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


# ============================================================
# YOLO PROCESSING
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
        # Bytes → NumPy
        # ----------------------------------------------------

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )


        # ----------------------------------------------------
        # JPEG → OpenCV
        # ----------------------------------------------------

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if frame is not None:

            # =================================================
            # YOLO
            # =================================================

            results = model(
                frame,

                imgsz=640,

                conf=0.35,

                verbose=False,
            )


            result = results[0]


            # =================================================
            # DRAW BOXES
            # =================================================

            annotated =
                result.plot()


            # BGR → RGB

            annotated = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )


            # =================================================
            # DISPLAY
            # =================================================

            st.subheader(
                "🔎 Live Detection"
            )

            st.image(
                annotated,
                channels="RGB",
                use_container_width=True,
            )


            # =================================================
            # OBJECTS
            # =================================================

            st.subheader(
                "Objects Detected"
            )


            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

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


                    object_name = model.names[
                        class_id
                    ]


                    st.write(
                        f"**{index}. "
                        f"{object_name.capitalize()}** "
                        f"— {confidence:.1%}"
                    )


            else:

                st.info(
                    "No objects detected."
                )


    except Exception as error:

        st.error(
            "Frame processing error"
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
