import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Constants
IMAGE_SIZE = 300
ALPHABETS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Load the trained model
model = load_model('/Users/yash/PycharmProjects/Final_ANN_BSL_Fingerspelling/bsl_fingerspelling_model7.h5')

# Initialize the HandDetector
detector = HandDetector(detectionCon=0.5, maxHands=2)

def preprocess_image(img):
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = np.expand_dims(img, axis=0) / 255.0
    return img

def predict_image(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    return ALPHABETS[np.argmax(prediction)]

def draw_prediction(frame, prediction):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_thickness = 3
    text_size, _ = cv2.getTextSize(prediction, font, font_scale, font_thickness)
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = frame.shape[0] - 50
    cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
    cv2.putText(frame, prediction, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

# Capture images from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hands, img = detector.findHands(frame)

    if hands:
        x_min, y_min = frame.shape[1], frame.shape[0]
        x_max, y_max = 0, 0

        for hand in hands:
            x_min = min(x_min, hand['bbox'][0])
            y_min = min(y_min, hand['bbox'][1])
            x_max = max(x_max, hand['bbox'][0] + hand['bbox'][2])
            y_max = max(y_max, hand['bbox'][1] + hand['bbox'][3])

        CROP_OFFSET = 20
        x_min -= CROP_OFFSET
        y_min -= CROP_OFFSET
        x_max += CROP_OFFSET
        y_max += CROP_OFFSET

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)

        hand_img = frame[y_min:y_max, x_min:x_max]
        prediction = predict_image(hand_img)
        draw_prediction(frame, prediction)

    cv2.imshow('BSL Fingerspelling Detection', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
