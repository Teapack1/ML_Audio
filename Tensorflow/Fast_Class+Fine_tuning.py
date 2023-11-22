import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Function to preprocess audio file into a spectrogram
def preprocess_audio_file(file_path):
    raw_audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(raw_audio)
    spectrogram = tf.signal.stft(audio, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram

# Load and preprocess the dataset
data_dir = 'path/to/5_class_dataset'
categories = os.listdir(data_dir)
files = []
labels = []

for label, category in enumerate(categories):
    category_files = os.listdir(os.path.join(data_dir, category))
    for file in category_files:
        file_path = os.path.join(data_dir, category, file)
        spectrogram = preprocess_audio_file(file_path)
        files.append(spectrogram)
        labels.append(label)

files = tf.stack(files)
labels = to_categorical(labels)

# Split the dataset
train_features, val_features, train_labels, val_labels = train_test_split(files.numpy(), labels, test_size=0.2)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(train_features.shape[1], train_features.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_features, train_labels, epochs=10, validation_data=(val_features, val_labels))

# Save the model
model.save('base_audio_model.h5')


# ---------------------------- Fine tuning ----------------------------

"""
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input

# Load the base model
base_model = load_model('base_audio_model.h5')

# Assume the base model's last dense layer's name is 'dense'
base_model.pop()  # Remove the last layer
base_model.trainable = False  # Freeze the layers of the base model

# Add new output layer for 8 classes
new_output = Dense(8, activation='softmax')(base_model.layers[-1].output)
fine_tuned_model = Model(inputs=base_model.input, outputs=new_output)

# Load and preprocess the new dataset with additional 3 classes
new_data_dir = 'path/to/3_class_dataset'
new_categories = os.listdir(new_data_dir)
new_files = []
new_labels = []

for label, category in enumerate(new_categories, start=5):  # Start from label 5
    category_files = os.listdir(os.path.join(new_data_dir, category))
    for file in category_files:
        file_path = os.path.join(new_data_dir, category, file)
        spectrogram = preprocess_audio_file(file_path)
        new_files.append(spectrogram)
        new_labels.append(label)

new_files = tf.stack(new_files)
new_labels = to_categorical(new_labels, num_classes=8)

# Split the new dataset
train_new_features, val_new_features, train_new_labels, val_new_labels = train_test_split(new_files.numpy(), new_labels, test_size=0.2)

# Compile the fine-tuned model
fine_tuned_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
fine_tuned_model.fit(train_new_features, train_new_labels, epochs=10, validation_data=(val_new_features, val_new_labels))

# Save the fine-tuned model
fine_tuned_model.save('fine_tuned_audio_model
