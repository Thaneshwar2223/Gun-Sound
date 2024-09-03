# Gun-Sound
This project implements a real-time gun sound classification system using a microphone. It processes live audio input and classifies it as either a specific gun type or "No Threat." The system is built using TensorFlow, Keras, and other essential libraries, and it provides an interactive UI for real-time classification.

Features

Real-Time Classification: Continuously monitors audio input through a microphone and classifies sounds instantly.

Threat Detection: Identifies gunshot sounds and categorizes them by type. Non-gun sounds are labeled as "No Threat."

Interactive UI: Displays a microphone icon that vibrates in response to sound input, providing a visual cue for audio activity.

Custom Model: Uses a pre-trained model that has been trained on gunshot sound data.

Prerequisites are in the requirement file

**HOW IT WORKS**

First we have a dataset of gun sounds of serval guns in dataset.tar

Then it is trained using train1.py

labeler.pkl is used to store the labels

The trained model is saved as  "gun_sound_classification_model.h5" 

The trained model is used to predict the gun the sound using val.py

labeler.pkl is used to store the labels

As of now it can detect gun names based on the dataset

Further updation under process
