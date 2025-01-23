import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser
from PIL import Image

# Load emotion detection model and labels
model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic # type: ignore
hands = mp.solutions.hands # type: ignore
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils # type: ignore

# Set up the Streamlit app header
st.header("Emotion Based Music Recommender")

# Initialize the session state for the first run
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Attempt to load previously detected emotion
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

# If no emotion is detected, set "run" to true to capture emotion
if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Define a class for processing video frames to detect emotions
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        # Flip the frame horizontally for better user experience
        frm = cv2.flip(frm, 1)

        # Process the frame to detect face and hand landmarks
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        # Extract landmarks and calculate relative coordinates
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1,-1)

            # Predict emotion label using the model
            pred = label[np.argmax(model.predict(lst))] # type: ignore

            # Display the predicted emotion label on the frame
            cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

            # Save the detected emotion
            np.save("emotion.npy", np.array([pred]))

      # No need of face and hand Draw landmarks on the frame
      #  drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
       #                         landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
        #                        connection_drawing_spec=drawing.DrawingSpec(thickness=1))
       # drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
       # drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Language and singer input fields
lang = st.text_input("Language")
singer = st.text_input("Singer")

# Image upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Convert the file to an OpenCV image
    image = Image.open(uploaded_image)
    cv_image = np.array(image)

    # Process the image to detect emotions
    res = holis.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # Further processing and emotion detection as before...

# If language, singer, and emotion detection are enabled, start webcam streaming
if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True,
                    video_processor_factory=EmotionProcessor) # type: ignore

# Button to recommend songs
btn = st.button("Recommend me songs")

if btn:
    # If emotion is not detected, display a warning message
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        # Open YouTube search results based on language, emotion, and singer
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
        # Reset detected emotion and update session state
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"
