
import os
import pickle

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="ISL Recognition",
    page_icon="🤞",
    layout="wide"
)


# =========================================================
# CUSTOM STYLING — ORIGINAL GREEN THEME
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

header[data-testid="stHeader"] {
    display: none !important;
}

.stApp {
    background-color: #f7f9f8;
    color: #1b2622;
    font-family: 'Inter', sans-serif;
}

.top-banner {
    background: #ffffff;
    padding: 20px 24px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
}

.card-box {
    background-color: #ffffff;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.04);
    margin-bottom: 10px;
}

.big-letter {
    font-size: 85px;
    font-weight: 800;
    color: #114b3e;
    text-align: center;
    line-height: 1;
    margin: 10px 0;
}

button[data-baseweb="tab"] {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #5c6f66 !important;
    background-color: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 20px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #114b3e !important;
    background-color: #e1f0ec !important;
    border-bottom: 3px solid #114b3e !important;
}

div[data-baseweb="tab-highlight"] {
    background-color: #114b3e !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================
MODEL_PATH = "isl_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Error: '{MODEL_PATH}' not found! "
        "Please ensure the trained model is available."
    )
    st.stop()

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# =========================================================
# MEDIAPIPE INITIALIZATION
# =========================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# =========================================================
# VIDEO PROCESSOR
# Runs on frames received from the user's browser.
# =========================================================
class SignSpeakVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.predicted_char = "-"
        self.confidence_text = "—"
        self.status_text = "Show a hand sign"

    def recv(self, frame):
        # Convert the incoming browser frame into an OpenCV image.
        image = frame.to_ndarray(format="bgr24")

        # Mirror the image like a selfie camera.
        image = cv2.flip(image, 1)

        # MediaPipe expects RGB input.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        predicted_char = "-"
        confidence_text = "—"
        status_text = "No hand detected"

        if results.multi_hand_landmarks:

            # Draw landmarks for all detected hands.
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(
                        color=(17, 75, 62),
                        thickness=2,
                        circle_radius=3
                    ),
                    mp_draw.DrawingSpec(
                        color=(82, 183, 136),
                        thickness=2,
                        circle_radius=2
                    )
                )

            # Use the first detected hand for prediction,
            # matching the original model input format.
            first_hand = results.multi_hand_landmarks[0]

            coords = np.array(
                [[landmark.x, landmark.y]
                 for landmark in first_hand.landmark],
                dtype=np.float32
            )

            # Normalize coordinates relative to the wrist.
            wrist = coords[0]
            normalized_coords = coords - wrist

            max_distance = np.max(
                np.sqrt(
                    np.sum(normalized_coords ** 2, axis=1)
                )
            )

            if max_distance > 0:
                normalized_coords = normalized_coords / max_distance

                # 21 landmarks × (x, y) = 42 features.
                features = normalized_coords.flatten().reshape(1, -1)

                try:
                    prediction = model.predict(features)
                    predicted_char = str(prediction[0])
                    status_text = "Sign detected"

                    # Show model confidence if supported.
                    if hasattr(model, "predict_proba"):
                        try:
                            probabilities = model.predict_proba(features)[0]
                            confidence_text = (
                                f"{np.max(probabilities) * 100:.1f}%"
                            )
                        except Exception:
                            confidence_text = "Unavailable"
                    else:
                        confidence_text = "Unavailable"

                except Exception:
                    predicted_char = "-"
                    confidence_text = "Unavailable"
                    status_text = "Prediction failed"

        # Save the latest result for the Streamlit UI.
        self.predicted_char = predicted_char
        self.confidence_text = confidence_text
        self.status_text = status_text

        # Return the processed frame with the skeleton overlay.
        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# =========================================================
# TABS
# =========================================================
tab1, tab2 = st.tabs([
    "📷 Live Recognition",
    "📖 Learn ISL Alphabet"
])


# =========================================================
# TAB 1 — LIVE RECOGNITION
# =========================================================
with tab1:

    st.markdown("""
    <div class="top-banner">
        <h2 style="margin:0; color:#114b3e; font-size:24px;
                   font-weight:700;">
            Turn your signs into letters.
        </h2>
        <p style="margin:6px 0 0 0; color:#5c6f66; font-size:15px;">
            Show your ISL alphabet signs to your camera.
            Skeleton tracking supports up to two hands.
        </p>
    </div>
    """, unsafe_allow_html=True)

    run_cam = st.toggle(
        "🟢 Activate Camera Stream",
        value=False,
        key="camera_toggle"
    )

    col_cam, col_res = st.columns([1.2, 1], gap="large")

    with col_cam:
        st.markdown("### 📹 Live camera")

        if run_cam:
            webrtc_ctx = webrtc_streamer(
                key="signspeak-camera",
                video_processor_factory=SignSpeakVideoProcessor,
                media_stream_constraints={
                    "video": True,
                    "audio": False
                },
                async_processing=True
            )
        else:
            st.markdown("""
            <div class="card-box"
                 style="text-align:center; padding:100px 20px;
                        color:#5c6f66;">
                <h3>Camera feed appears here</h3>
                <p>Toggle the switch above to activate your webcam.</p>
            </div>
            """, unsafe_allow_html=True)

        st.caption("Position your hands clearly inside the frame.")

    with col_res:
        st.markdown("### 🎯 Recognition result")

        result_placeholder = st.empty()

        result_placeholder.markdown("""
        <div class="card-box"
             style="text-align:center; min-height:240px;">
            <p class="big-letter">-</p>
            <p style="color:#114b3e; font-size:16px; font-weight:600;">
                Predicted letter
            </p>
            <hr style="border-color:#e2e8f0; margin:18px 0;">
            <p style="color:#5c6f66; font-size:14px;">
                Confidence: — | Waiting for a sign
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Refresh only the prediction panel periodically.
    # The WebRTC component continues receiving browser frames
    # independently of these UI refreshes.
    if run_cam:

        @st.fragment(run_every=0.3)
        def update_prediction():
            processor = webrtc_ctx.video_processor

            if not webrtc_ctx.state.playing:
                result_placeholder.markdown("""
                <div class="card-box" style="text-align:center;
                                             min-height:240px;">
                    <p class="big-letter">-</p>
                    <p style="color:#114b3e; font-weight:600;">
                        Predicted letter
                    </p>
                    <p style="color:#5c6f66;">
                        Start the camera to recognize a sign.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                return

            if processor is None:
                return

            letter = processor.predicted_char
            confidence = processor.confidence_text
            status = processor.status_text

            result_placeholder.markdown(f"""
            <div class="card-box"
                 style="text-align:center; min-height:240px;">
                <p class="big-letter">{letter}</p>
                <p style="color:#114b3e; font-size:16px;
                          font-weight:600; margin:0;">
                    Predicted letter
                </p>
                <hr style="border-color:#e2e8f0; margin:18px 0;">
                <p style="color:#114b3e; font-size:14px;
                          font-weight:600;">
                    Model confidence: {confidence}
                </p>
                <p style="color:#5c6f66; font-size:13px;">
                    {status}
                </p>
            </div>
            """, unsafe_allow_html=True)

        update_prediction()


# =========================================================
# TAB 2 — LEARN ISL ALPHABET
# =========================================================
with tab2:

    st.markdown("""
    <div class="top-banner">
        <h2 style="margin:0; color:#114b3e; font-size:24px;
                   font-weight:700;">
            Explore Indian Sign Language Alphabet (A–Z)
        </h2>
        <p style="margin:6px 0 0 0; color:#5c6f66; font-size:15px;">
            Reference the alphabet chart and gesture descriptions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🖼️ Official ISL Alphabet Reference Chart")

    st.image(
        "https://enablemart.in/wp-content/uploads/2025/04/ISL-Basic.png",
        caption="Indian Sign Language Manual Alphabet Reference",
        width="stretch"
    )

    st.markdown("### 📋 Quick Hand Gesture Reference Guide")

    isl_guides = {
        "A": "Make a fist with thumb resting straight up against the side of the index finger.",
        "B": "Hold hand straight up with fingers together and thumb folded across palm.",
        "C": "Curve your hand and fingers to form the shape of the letter C.",
        "D": "Point index finger straight up while thumb and other fingers form a circle.",
        "E": "Curl all fingers down tightly across your palm with thumb resting underneath.",
        "F": "Touch tip of thumb and index finger to make a circle; keep other fingers upright.",
        "G": "Point index finger and thumb forward parallel to each other.",
        "H": "Extend index and middle fingers horizontally forward together.",
        "I": "Make a fist with only the pinky finger extended straight up.",
        "J": "Extend pinky finger and trace a J curve in the air.",
        "K": "Point index and middle fingers up in a V, placing thumb between them.",
        "L": "Extend thumb and index finger outward at a 90-degree angle to form an L.",
        "M": "Fold thumb under index, middle, and ring fingers draped over it.",
        "N": "Fold thumb under index and middle fingers.",
        "O": "Curve fingers and thumb to touch tips, forming an O shape.",
        "P": "Point index finger forward and middle finger down with thumb across.",
        "Q": "Point index finger and thumb down toward the floor.",
        "R": "Cross your index and middle fingers over each other.",
        "S": "Make a fist with your thumb wrapped tightly across the front of your fingers.",
        "T": "Place thumb between your index and middle finger in a closed fist.",
        "U": "Hold index and middle fingers straight up close together.",
        "V": "Hold index and middle fingers up in a separated V shape.",
        "W": "Hold index, middle, and ring fingers up spread apart.",
        "X": "Hook your index finger like a question mark.",
        "Y": "Extend thumb and pinky finger outward on opposite sides.",
        "Z": "Use index finger to trace the letter Z shape in the air."
    }

    cols = st.columns(3)

    for index, (letter, description) in enumerate(isl_guides.items()):
        with cols[index % 3]:
            st.markdown(f"""
            <div class="card-box"
                 style="margin-bottom:15px; padding:15px 20px;">
                <h4 style="margin:0; color:#114b3e;">
                    Letter {letter}
                </h4>
                <p style="color:#5c6f66; font-size:13px;
                          margin:5px 0 0 0;">
                    {description}
                </p>
            </div>
            """, unsafe_allow_html=True)
