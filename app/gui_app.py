import streamlit as st
import pickle
import re
import string

# Clean incoming text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|\d+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

# Load model
@st.cache_resource
def load_model():
    with open("models/spam_filter_model.pkl", "rb") as f:
        return pickle.load(f)

model, vectorizer = load_model()

# Streamlit app
st.title(" SMS Spam Filter")
st.write("Classify messages with adjustable spam-risk levels")

# Message input
message = st.text_area("Enter your message")

# Risk threshold slider
risk_level = st.slider("Spam Risk Threshold", 0.1, 0.99, 0.5, step=0.05)

if st.button("Check Message"):
    if message:
        cleaned = clean_text(message)
        vect = vectorizer.transform([cleaned])
        prob = model.predict_proba(vect)[0][1]
        label = "SPAM" if prob >= risk_level else "HAM"

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {prob:.2f}")
        if label == "SPAM" and risk_level > 0.8:
            st.warning("This message was flagged under strict rules.")
        elif label == "HAM" and prob > 0.4:
            st.info("Message is borderline — consider flagging for manual review.")
    else:
        st.error("Please enter a message first.")
