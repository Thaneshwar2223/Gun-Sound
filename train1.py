import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# Set paths and parameters
dataset_path = "D:/gun_sound/dataset"   # Replace with the path to your dataset
sample_rate = 3000
n_mfcc = 13  # Number of MFCCs to extract
test_size = 0.2  # 20% data for testing
num_classes = 9  # Assuming 9 different gun classes
n_fft = 512  # Smaller window size


def extract_features(file_path):
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


# Initialize lists to hold features and labels
features = []
labels = []

# Walk through the dataset directory
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav'):
            label = os.path.basename(root)
            file_path = os.path.join(root, file)
            mfccs = extract_features(file_path)
            if mfccs is not None:
                features.append(mfccs)
                labels.append(label)

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Encode labels to integers
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
# Print unique encoded labels to verify
print("Unique encoded labels:", np.unique(labels_encoded))
print("Number of unique labels:", len(np.unique(labels_encoded)))

# Update num_classes based on the actual unique labels
num_classes = len(np.unique(labels_encoded))

# Convert to categorical
labels_categorical = to_categorical(labels_encoded, num_classes=num_classes)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_categorical, test_size=test_size, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(256, input_shape=(n_mfcc,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Save the model
model.save("gun_sound_classification_model.h5")
print("saved")
