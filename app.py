import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import gdown

def download_model_from_drive(drive_id, destination):
    if not os.path.exists(destination):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={drive_id}&export=download"
        gdown.download(url, destination, quiet=False)
        print("Download complete.")

# Replace with your actual Drive file ID
drive_file_id = "16F80tsiZb2rAhOL1aTzs4yso7NWPQKFu"
model_file_path = "sentiment_model.h5"
download_model_from_drive(drive_file_id, model_file_path)

# Load model with custom object
model = tf.keras.models.load_model(
    model_file_path,
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
    sequences = tokenizer.texts_to_sequences([text_input])
    padded = pad_sequences(sequences, maxlen=30)

    prediction = model.predict(padded)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    st.write("### Predicted Sentiment:")
    st.success(predicted_label)
