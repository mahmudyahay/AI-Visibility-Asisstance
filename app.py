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
    "Point the camera at your surroundings. "
    "YOLO will detect the objects in view."
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
    name="live_rear_camera",

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
            Starting camera...
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

    # --------------------------------------------------------
    # JAVASCRIPT
    # --------------------------------------------------------

    js="""
    export default function(component) {

        const {
            parentElement,
            setTriggerValue
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
                 * Request the BACK camera.
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
                 * Fallback for computers without
                 * an environment/rear camera.
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
        // CAPTURE FRAME
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
                     * Capture at 640 × 480.
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
                     * Convert the frame to JPEG.
                     */

                    const image =
                        canvas.toDataURL(
                            "image/jpeg",
                            0.65
                        );


                    /*
                     * IMPORTANT:
                     *
                     * A camera frame is an EVENT.
                     * Therefore we use a trigger,
                     * not persistent state.
                     */

                    setTriggerValue(
                        "frame",
                        image
                    );
                }


                /*
                 * Send approximately 5 frames
                 * per second.
                 */

                setTimeout(
                    capture,
                    200
                );
            }


            capture();
        }


        // ====================================================
        // START CAMERA
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
    """,
)


# ============================================================
# MOUNT COMPONENT
# ============================================================

camera_result = camera_component(

    key="accessibility_camera",

    on_frame_change=lambda: None,
)


# ============================================================
# GET FRAME FROM COMPONENT
# ============================================================

frame_data = camera_result.frame


# ============================================================
# SHOW COMMUNICATION STATUS
# ============================================================

if frame_data:

    st.caption("🟢 Camera frame received by Python")


    try:

        # ----------------------------------------------------
        # Remove Base64 header
        # ----------------------------------------------------

        if "," in frame_data:

            encoded_image = frame_data.split(
                ",",
                1
            )[1]

        else:

            encoded_image = frame_data


        # ----------------------------------------------------
        # Base64 → bytes
        # ----------------------------------------------------

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


        if frame is None:

            st.error(
                "Python received the frame, "
                "but OpenCV could not decode it."
            )

        else:

            # =================================================
            # YOLO
            # =================================================

            results = model(
                frame,

                # Keep 640 for accuracy.
                imgsz=640,

                # Confidence threshold.
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
            # DISPLAY
            # =================================================

            st.subheader("🔎 Live Detection")


            st.image(
                annotated_frame,
                channels="RGB",
                use_container_width=True,
            )


            # =================================================
            # OBJECT LIST
            # =================================================

            st.subheader("Objects Detected")


            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                detected_objects = []


                for box in result.boxes:

                    class_id = int(
                        box.cls[0].item()
                    )


                    confidence = float(
                        box.conf[0].item()
                    )


                    object_name = model.names[
                        class_id
                    ]


                    detected_objects.append(
                        (
                            object_name,
                            confidence
                        )
                    )


                # ------------------------------------------------
                # PRINT EACH OBJECT
                # ------------------------------------------------

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
                    "No objects detected in this frame."
                )


    except Exception as error:

        st.error(
            "YOLO processing error"
        )

        st.code(
            str(error)
        )

else:

    st.info(
        "Waiting for the first camera frame..."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Rear Camera → Frame → OpenCV → "
    "YOLO → Bounding Boxes → Object List"
)
