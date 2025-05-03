import streamlit as st
import joblib
import os

# Get current directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load files safely using full paths
model = joblib.load(os.path.join(BASE_DIR, "emotion_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

# App UI
st.set_page_config(page_title="Emotion Detector", page_icon="🧠")
st.title("🧠 Emotion Detection from Text")
st.markdown("Type a sentence to detect the emotion.")

# Input from user
user_input = st.text_area("Enter your sentence:")

# Check for dangerous phrases
sensitive_words = [
    "suicide", "dead", "die", "kill", "self-harm", "depressed",
    "sad", "alone", "worthless", "hopeless", "hate myself", "empty", "no one cares", "end it"
]

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    elif any(word in user_input.lower() for word in sensitive_words):
        st.error("⚠️ This may contain harmful or sensitive content. Please talk to someone you trust.")
    else:
        # Vectorize and predict
        input_vec = vectorizer.transform([user_input]).toarray()
        prediction = model.predict(input_vec)
        emotion = label_encoder.inverse_transform([prediction[0]])[0]

        st.success(f"Predicted Emotion: **{emotion}**")



import os
import joblib
import streamlit as st

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct full paths to the model files
model_path = os.path.join(BASE_DIR, 'emotion_model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
label_encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')

# Load the models
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
label_encoder = joblib.load(label_encoder_path)
