from tensorflow.keras.models import load_model
import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import pickle
# Load the pre-trained model
model = load_model("gun_sound_classification_model.h5")
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)


# Function to extract features from an audio file
def extract_features(file_path, sample_rate=3000, n_mfcc=13, n_fft=512):
    try:
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=sample_rate)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)

        # Compute the mean of MFCCs across the time axis
        mfccs_scaled = np.mean(mfccs.T, axis=0)

        return mfccs_scaled
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Function to classify a new sound
def classify_sound(file_path):
    features = extract_features(file_path)
    if features is not None:
        features = np.expand_dims(features, axis=0)  # Reshape for the model input
        prediction = model.predict(features)
        predicted_label = le.inverse_transform([np.argmax(prediction)])
        return predicted_label
    else:
        return "Error in processing the sound file."


# Example usage
new_sound_path = "D:/gun_sound/dataset/Zastava M92/9 (4).wav"
predicted_gun = classify_sound(new_sound_path)
print(f"Predicted Gun: {predicted_gun[0]}")
