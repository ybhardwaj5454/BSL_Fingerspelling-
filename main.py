import cv2
import numpy as np
import os
import math
from cvzone.HandTrackingModule import HandDetector

# Constants
CROP_OFFSET = 20
IMAGE_SIZE = 300
ALPHABETS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # List of alphabets A-Z

# Initialize the HandDetector
detector = HandDetector(detectionCon=0.5, maxHands=2)

# Hardcoded directory for storing dataset
directory = '/Users/yash/Desktop/DataCollection_BSL5'
os.makedirs(directory, exist_ok=True)

for subDirectory in ALPHABETS:
    path = os.path.join(directory, subDirectory)
    os.makedirs(path, exist_ok=True)


def remove_text_from_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use the threshold to get a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small or large boxes assuming text will be within a certain size range
        if 20 < w < 100 and 10 < h < 40:
            # Draw a white rectangle over the text
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)

    return image


def preprocess_and_save(hand_img, count, sub_dir):
    # Remove text from the image
    hand_img = remove_text_from_image(hand_img)

    # Create a white background image
    white_bg = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8) * 255
    hand_height, hand_width, _ = hand_img.shape

    # Calculate aspect ratio
    aspect_ratio = hand_height / hand_width

    # Resize and place the cropped hand image on the white background
    if aspect_ratio > 1:
        k = IMAGE_SIZE / hand_height
        width_cal = math.ceil(k * hand_width)
        image_resized = cv2.resize(hand_img, (width_cal, IMAGE_SIZE))
        width_gap = math.ceil((IMAGE_SIZE - width_cal) / 2)
        white_bg[:, width_gap:width_cal + width_gap] = image_resized
    else:
        k = IMAGE_SIZE / hand_width
        height_cal = math.ceil(k * hand_height)
        image_resized = cv2.resize(hand_img, (IMAGE_SIZE, height_cal))
        height_gap = math.ceil((IMAGE_SIZE - height_cal) / 2)
        white_bg[height_gap:height_cal + height_gap, :] = image_resized

    # Display the preprocessed hand image
    cv2.imshow("Preprocessed Hand", white_bg)

    # Save the preprocessed hand image
    cv2.imwrite(f'{sub_dir}/hand_{count}.jpg', white_bg)


# Capture images from webcam
cap = cv2.VideoCapture(0)
count = 0
CAPTURE_FLAG = False


def get_alphabet():
    while True:
        current_alphabet = input(
            "Enter the alphabet for which you want to capture images (or type 'exit' to quit): ").upper()
        if current_alphabet == 'EXIT':
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif current_alphabet in ALPHABETS:
            return current_alphabet
        else:
            print("Please enter a valid alphabet.")


current_alphabet = get_alphabet()
path = os.path.join(directory, current_alphabet)

print(
    'Now camera window will be open, then \n1) Place your hand gesture in ROI and press s key to start capturing images.\n2) Press esc key to stop capturing images.\n3) Press n key to switch to a different alphabet.\n4) Press esc key to exit.')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect hands
    hands, img = detector.findHands(frame)

    if hands:
        # Initialize min and max values
        x_min, y_min = frame.shape[1], frame.shape[0]
        x_max, y_max = 0, 0

        # Get the combined bounding box of all detected hands
        for hand in hands:
            x_min = min(x_min, hand['bbox'][0])
            y_min = min(y_min, hand['bbox'][1])
            x_max = max(x_max, hand['bbox'][0] + hand['bbox'][2])
            y_max = max(y_max, hand['bbox'][1] + hand['bbox'][3])

        # Apply offset
        x_min -= CROP_OFFSET
        y_min -= CROP_OFFSET
        x_max += CROP_OFFSET
        y_max += CROP_OFFSET

        # Ensure the bounding box is within frame boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)

        # Crop the hand region
        hand_img = frame[y_min:y_max, x_min:x_max]

        if CAPTURE_FLAG:
            preprocess_and_save(hand_img, count, path)
            count += 1

    # Display the frame
    cv2.imshow('Hand Detection', img)

    pressedKey = cv2.waitKey(1)
    if pressedKey == 27:  # ESC key to exit
        break
    elif pressedKey == ord('s'):  # 's' key to start/stop capturing
        CAPTURE_FLAG = not CAPTURE_FLAG
    elif pressedKey == ord('n'):  # 'n' key to switch to a different alphabet
        current_alphabet = get_alphabet()
        path = os.path.join(directory, current_alphabet)
        count = 0  # Reset count for new alphabet

cap.release()
cv2.destroyAllWindows()
