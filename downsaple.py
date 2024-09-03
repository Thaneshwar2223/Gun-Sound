import os
import librosa
import soundfile as sf

# Define the directories
input_directory = "D:/gun_sound/Zastava M92"  # Replace with your dataset path
output_directory = "D:/gun_sound/Zastava M92_3KHZ"  # Replace with the output directory path

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


# Function to downsample audio
def downsample_audio(input_file, output_file, target_sr=3000):
    # Load the audio file
    audio, sr = librosa.load(input_file, sr=None)

    # Downsample to target sample rate
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Save the resampled audio
    sf.write(output_file, audio_resampled, target_sr)


# Walk through the directory structure
for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.endswith('.wav'):
            # Construct full file path
            input_file_path = os.path.join(root, file)

            # Create corresponding output path
            relative_path = os.path.relpath(root, input_directory)
            output_folder_path = os.path.join(output_directory, relative_path)
            output_file_path = os.path.join(output_folder_path, file)

            # Create output folder if it doesn't exist
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            # Downsample and save the file
            downsample_audio(input_file_path, output_file_path)

print("Downsampling complete!")
