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
    "The system detects objects and displays their bounding boxes."
)


# ============================================================
# LOAD YOLO
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()


# ============================================================
# LIVE CAMERA COMPONENT
# ============================================================

camera = st.components.v2.component(
    name="rear_camera",
    
    html="""
    <div class="camera-container">

        <video
            id="camera"
            autoplay
            playsinline
            muted
        ></video>

        <canvas id="canvas"></canvas>

        <div id="status">
            Starting rear camera...
        </div>

    </div>
    """,

    css="""
    .camera-container {
        width: 100%;
        max-width: 800px;
        margin: auto;
        position: relative;
        overflow: hidden;
        border-radius: 12px;
        background: #000;
    }

    #camera {
        width: 100%;
        height: auto;
        display: block;
        transform: none;
    }

    #canvas {
        display: none;
    }

    #status {
        position: absolute;
        bottom: 10px;
        left: 10px;
        background: rgba(0,0,0,0.7);
        color: white;
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 13px;
    }
    """,

    js="""
    export default function(component) {

        const {
            parentElement,
            setStateValue
        } = component;


        const video = parentElement.querySelector("#camera");
        const canvas = parentElement.querySelector("#canvas");
        const status = parentElement.querySelector("#status");

        const context = canvas.getContext("2d");


        let stream = null;
        let running = true;


        async function startCamera() {

            try {

                /*
                 * Request the BACK / ENVIRONMENT camera.
                 *
                 * exact: "environment" tells the browser
                 * that we specifically want the rear camera.
                 */

                stream = await navigator.mediaDevices.getUserMedia({

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

                status.textContent = "Rear camera active";


                /*
                 * Capture frames periodically.
                 *
                 * The camera itself can run smoothly,
                 * but we only send about 5 frames/second
                 * to Python for YOLO processing.
                 */

                const captureFrame = () => {

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
                         * JPEG keeps the amount of data sent
                         * from browser → Python much smaller.
                         */

                        const imageData =
                            canvas.toDataURL(
                                "image/jpeg",
                                0.65
                            );


                        /*
                         * Send the frame to Python.
                         */

                        setStateValue(
                            "frame",
                            imageData
                        );
                    }


                    /*
                     * Approximately 5 FPS.
                     *
                     * This prevents us from sending
                     * 30+ frames every second to Python.
                     */

                    setTimeout(
                        captureFrame,
                        200
                    );
                };


                captureFrame();

            } catch (error) {

                console.error(error);

                status.textContent =
                    "Could not access rear camera";

                /*
                 * Some laptops only have a front webcam.
                 *
                 * In that case, try the normal camera
                 * instead of completely failing.
                 */

                try {

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
                        "Camera active";


                    const captureFallback = () => {

                        if (!running) {
                            return;
                        }

                        if (
                            video.readyState >=
                            HTMLMediaElement.HAVE_CURRENT_DATA
                        ) {

                            canvas.width = 640;
                            canvas.height = 480;

                            context.drawImage(
                                video,
                                0,
                                0,
                                640,
                                480
                            );


                            const imageData =
                                canvas.toDataURL(
                                    "image/jpeg",
                                    0.65
                                );


                            setStateValue(
                                "frame",
                                imageData
                            );
                        }


                        setTimeout(
                            captureFallback,
                            200
                        );
                    };


                    captureFallback();

                } catch (fallbackError) {

                    status.textContent =
                        "Camera permission denied";

                    console.error(
                        fallbackError
                    );
                }
            }
        }


        startCamera();


        /*
         * Cleanup when component is removed.
         */

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

camera_result = camera(
    key="accessibility_camera"
)


# ============================================================
# PROCESS FRAME WITH YOLO
# ============================================================

if camera_result is not None:

    frame_data = camera_result.state.get("frame")


    if frame_data:

        try:

            # ------------------------------------------------
            # Remove base64 header
            # ------------------------------------------------

            encoded = frame_data.split(",", 1)[1]


            # ------------------------------------------------
            # Base64 → bytes
            # ------------------------------------------------

            image_bytes = base64.b64decode(
                encoded
            )


            # ------------------------------------------------
            # Bytes → NumPy
            # ------------------------------------------------

            image_array = np.frombuffer(
                image_bytes,
                dtype=np.uint8
            )


            # ------------------------------------------------
            # JPEG → OpenCV image
            # ------------------------------------------------

            frame = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )


            if frame is not None:

                # ------------------------------------------------
                # YOLO
                # ------------------------------------------------

                results = model(
                    frame,

                    # 640 keeps better accuracy
                    imgsz=640,

                    # Adjust later after testing
                    conf=0.35,

                    verbose=False
                )


                result = results[0]


                # ------------------------------------------------
                # Draw bounding boxes
                # ------------------------------------------------

                annotated_frame = result.plot()


                # ------------------------------------------------
                # Convert BGR → RGB
                # ------------------------------------------------

                annotated_frame = cv2.cvtColor(
                    annotated_frame,
                    cv2.COLOR_BGR2RGB
                )


                # ------------------------------------------------
                # Display detection result
                # ------------------------------------------------

                st.subheader(
                    "🔎 Live Detection"
                )

                st.image(
                    annotated_frame,
                    channels="RGB",
                    use_container_width=True
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
                    # Print objects one by one
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
                        "No objects detected."
                    )


        except Exception as error:

            st.error(
                f"Frame processing error: {error}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Rear Camera → Live Frames → OpenCV → "
    "YOLO → Bounding Boxes → Object List"
)

