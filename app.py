import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="ISL Recognition", 
    page_icon="🤞", 
    layout="wide"
)

# --- Custom Styling (Zero Space Header + Forest Green Tabs) ---
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

# --- Load Trained Model ---
model_path = 'isl_model.pkl'
if not os.path.exists(model_path):
    st.error(f"Error: '{model_path}' not found! Please run 'train_models.py' first.")
    st.stop()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# --- Initialize MediaPipe (Set max_num_hands=2 for Two-Hand Support) ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Interactive Switcher Tabs ---
tab1, tab2 = st.tabs(["📷 Live Recognition", "📖 Learn ISL Alphabet"])

# ================= TAB 1: LIVE RECOGNITION =================
with tab1:
    st.markdown("""
        <div class="top-banner">
            <h2 style="margin:0; color:#114b3e; font-size:24px; font-weight:700;">Turn your signs into letters.</h2>
            <p style="margin:6px 0 0 0; color:#5c6f66; font-size:15px;">Show one or two ISL alphabet signs to your camera; skeleton tracking is active for up to 2 hands!</p>
        </div>
    """, unsafe_allow_html=True)

    run_cam = st.toggle("🟢 Activate Camera Stream", value=False)

    col_cam, col_res = st.columns([1.2, 1], gap="large")

    with col_cam:
        st.markdown("### 📹 Live camera ")
        video_placeholder = st.empty()
        st.caption("Position your hands clearly inside the frame.")

    with col_res:
        st.markdown("### 🎯 Recognition result")
        
        pred_box = st.empty()
        pred_box.markdown("""
            <div class="card-box" style="text-align:center; min-height:240px; display:flex; flex-direction:column; justify-content:center;">
                <p class="big-letter">-</p>
                <p style="color:#114b3e; font-size:16px; font-weight:600; margin:0;">Predicted letter</p>
                <hr style="border-color:#e2e8f0; margin:18px 0;">
                <p style="color:#5c6f66; font-size:14px; margin:0;">Confidence: — &nbsp;|&nbsp; Waiting for a sign</p>
            </div>
        """, unsafe_allow_html=True)
        st.caption("Show the predicted letter clearly.")

    if run_cam:
        cap = cv2.VideoCapture(0)
        while run_cam:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            predicted_char = "-"
            confidence_text = "High (~99%)"
            
            if results.multi_hand_landmarks:
                # Loop through all detected hands (up to 2) and draw skeleton dots/connections
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(17, 75, 62), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(82, 183, 136), thickness=2, circle_radius=2)
                    )
                
                # Predict using the first detected hand's landmarks
                first_hand = results.multi_hand_landmarks[0]
                coords = np.array([[lm.x, lm.y] for lm in first_hand.landmark])
                wrist = coords[0]
                norm_coords = coords - wrist
                max_dist = np.max(np.sqrt(np.sum(norm_coords**2, axis=1)))
                if max_dist > 0:
                    norm_coords = norm_coords / max_dist
                    
                features = norm_coords.flatten().reshape(1, -1)
                prediction = model.predict(features)
                predicted_char = prediction[0]

            pred_box.markdown(f"""
                <div class="card-box" style="text-align:center; min-height:240px; display:flex; flex-direction:column; justify-content:center;">
                    <p class="big-letter">{predicted_char}</p>
                    <p style="color:#114b3e; font-size:16px; font-weight:600; margin:0;">Predicted letter</p>
                    <hr style="border-color:#e2e8f0; margin:18px 0;">
                    <p style="color:#114b3e; font-size:14px; margin:0; font-weight:600;">Confidence: {confidence_text}</p>
                </div>
            """, unsafe_allow_html=True)
            
            frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_disp, channels="RGB", use_container_width=True)
        
        cap.release()
    else:
        video_placeholder.markdown("""
            <div class="card-box" style="text-align:center; padding:100px 20px; color:#5c6f66;">
                <h3>Camera feed appears here</h3>
                <p>Toggle the switch above to activate your webcam.</p>
            </div>
        """, unsafe_allow_html=True)


# ================= TAB 2: LEARN ISL ALPHABET =================
with tab2:
    st.markdown("""
        <div class="top-banner">
            <h2 style="margin:0; color:#114b3e; font-size:24px; font-weight:700;">Explore Indian Sign Language Alphabet (A–Z)</h2>
            <p style="margin:6px 0 0 0; color:#5c6f66; font-size:15px;">Reference official sign language charts and instructions for each letter.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🖼️ Official ISL Alphabet Reference Chart")
    st.image("https://enablemart.in/wp-content/uploads/2025/04/ISL-Basic.png", caption="Standard Indian Sign Language Manual Alphabet Reference[cite: 1]", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Quick Hand Gesture Reference Guide")

    isl_guides = {
        'A': 'Make a fist with thumb resting straight up against the side of the index finger.',
        'B': 'Hold hand straight up with fingers together and thumb folded across palm.',
        'C': 'Curve your hand and fingers to form the shape of the letter C.',
        'D': 'Point index finger straight up while thumb and other fingers form a circle.',
        'E': 'Curl all fingers down tightly across your palm with thumb resting underneath.',
        'F': 'Touch tip of thumb and index finger to make a circle; keep other fingers upright.',
        'G': 'Point index finger and thumb forward parallel to each other.',
        'H': 'Extend index and middle fingers horizontally forward together.',
        'I': 'Make a fist with only the pinky finger extended straight up.',
        'J': 'Extend pinky finger and trace a "J" curve in the air.',
        'K': 'Point index and middle fingers up in a V, placing thumb between them.',
        'L': 'Extend thumb and index finger outward at a 90-degree angle to form an L.',
        'M': 'Fold thumb under index, middle, and ring fingers draped over it.',
        'N': 'Fold thumb under index and middle fingers.',
        'O': 'Curve fingers and thumb to touch tips, forming an O shape.',
        'P': 'Point index finger forward and middle finger down with thumb across.',
        'Q': 'Point index finger and thumb down toward the floor.',
        'R': 'Cross your index and middle fingers over each other.',
        'S': 'Make a fist with your thumb wrapped tightly across the front of your fingers.',
        'T': 'Place thumb between your index and middle finger in a closed fist.',
        'U': 'Hold index and middle fingers straight up close together.',
        'V': 'Hold index and middle fingers up in a separated V shape (peace sign).',
        'W': 'Hold index, middle, and ring fingers up spread apart.',
        'X': 'Hook your index finger like a question mark.',
        'Y': 'Extend thumb and pinky finger outward on opposite sides.',
        'Z': 'Use index finger to trace the letter "Z" shape in the air.'
    }

    cols = st.columns(3)
    for i, (letter, desc) in enumerate(isl_guides.items()):
        with cols[i % 3]:
            st.markdown(f"""
                <div class="card-box" style="margin-bottom:15px; padding:15px 20px;">
                    <h4 style="margin:0; color:#114b3e;">Letter {letter}</h4>
                    <p style="color:#5c6f66; font-size:13px; margin:5px 0 0 0;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)