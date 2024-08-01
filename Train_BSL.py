import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Constants
IMAGE_SIZE = 300
DATASET_DIR = '/Users/yash/Desktop/DataCollection_BSL5'
ALPHABETS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

def load_images_from_directory(directory):
    images = []
    labels = []
    for label in ALPHABETS:
        path = os.path.join(directory, label)
        for img_file in os.listdir(path):
            img_path = os.path.join(path, img_file)
            print(f'Processing {img_path}')  # Log file path
            img = cv2.imread(img_path)
            if img is None:
                print(f'Warning: {img_path} is empty or invalid')  # Log invalid image
                continue
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

images, labels = load_images_from_directory(DATASET_DIR)

# Convert labels to one-hot encoding
label_map = {label: idx for idx, label in enumerate(ALPHABETS)}
labels = np.array([label_map[label] for label in labels])
labels = tf.keras.utils.to_categorical(labels, num_classes=len(ALPHABETS))

# Split dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(ALPHABETS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the Model
batch_size = 32
epochs = 25

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    validation_data=(x_val, y_val),
    epochs=epochs
)

# Save the Model
model.save('bsl_fingerspelling_model7.h5')

# Evaluation Code
from tensorflow.keras.models import load_model

# # Load the trained model
# model = load_model('bsl_fingerspelling_model.h5')
#
# def predict_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#     img = np.expand_dims(img, axis=0) / 255.0
#     prediction = model.predict(img)
#     return ALPHABETS[np.argmax(prediction)]
#
# # Test on a new image
# image_path = 'path_to_test_image.jpg'
# predicted_alphabet = predict_image(image_path)
# print(f'The predicted alphabet is: {predicted_alphabet}')
