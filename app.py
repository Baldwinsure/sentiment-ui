import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import requests

def download_model_from_drive(drive_id, destination):
    if not os.path.exists(destination):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={drive_id}"
        response = requests.get(url, stream=True)
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")

# Replace with your actual Drive file ID
drive_file_id = "16F80tsiZb2rAhOL1aTzs4yso7NWPQKFu"
model_file_path = "sentiment_model.h5"
download_model_from_drive(drive_file_id, model_file_path)
# Load model with custom object
model = tf.keras.models.load_model(
    "sentiment_model.h5",
    custom_objects={"SpatialDropout1D": SpatialDropout1D}
)

# Load tokenizer
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Streamlit UI
st.title("Sentiment Classifier (LSTM Model)")

text_input = st.text_area("Enter a sentence to classify")

if st.button("Predict"):
    # Load model only when needed
    with st.spinner("Loading model..."):
        model = tf.keras.models.load_model("sentiment_model.h5")
    
    # Preprocess
    sequences = tokenizer.texts_to_sequences([text_input])
    padded = pad_sequences(sequences, maxlen=30)

    # Predict
    prediction = model.predict(padded)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    st.write("### Predicted Sentiment:")
    st.success(predicted_label)
