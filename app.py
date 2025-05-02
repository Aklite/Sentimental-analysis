import streamlit as st
import joblib

# ============================
# Load your model and files
# ============================

# Load trained model, vectorizer, and label encoder
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# ============================
# Streamlit App UI
# ============================

st.set_page_config(page_title="Emotion Detection", page_icon="🧠")
st.title("🧠 Emotion Detection from Text")
st.subheader("Type a sentence and let the model detect the emotion")

# ============================
# User Input
# ============================

user_input = st.text_area("Enter your sentence here:")

# ============================
# Sensitive Keyword Filter
# ============================

dangerous_words = [
    "suicide", "dead", "die", "kill", "self-harm", "depressed", "sad", "alone",
    "worthless", "hopeless", "i hate myself", "i feel empty", "no one cares", "end it"
]

# ============================
# Predict Button
# ============================

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a sentence.")
    elif any(word in user_input.lower() for word in dangerous_words):
        st.error("⚠️ This sentence may contain harmful or sensitive content. Please talk to someone you trust.")
        st.markdown("💬 [Need support? Find help](https://www.who.int/campaigns/world-mental-health-day)")
    else:
        # Vectorize input and predict
        input_vector = vectorizer.transform([user_input]).toarray()
        prediction = model.predict(input_vector)
        predicted_emotion = le.inverse_transform([prediction[0]])[0]

        # Display prediction
        st.success(f"Predicted Emotion: **{predicted_emotion}**")
