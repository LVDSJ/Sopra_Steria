import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from hand_sign_utils import count_fingers, detect_hand_sign

mp_hands = mp.solutions.hands

GESTURE_CLASSES = ['peace_hand', 'surf_hand', 'circle_game', 'open_hand']
IMAGE_DIR = 'images'  # Root folder with gesture subfolders

def classify_gesture(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "none"

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7) as hands:
        results = hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return "none"

        handedness = results.multi_handedness[0].classification[0].label
        hand_landmarks = results.multi_hand_landmarks[0]
        fingers = count_fingers(hand_landmarks, handedness)
        sign = detect_hand_sign(fingers, hand_landmarks)
        return sign.lower().replace(" ", "_") if sign else "none"

def evaluate_model():
    y_true, y_pred = [], []

    for gesture_class in GESTURE_CLASSES:
        folder_path = os.path.join(IMAGE_DIR, gesture_class)
        if not os.path.exists(folder_path):
            continue

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            predicted = classify_gesture(img_path)
            y_true.append(gesture_class)
            y_pred.append(predicted)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=GESTURE_CLASSES)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=GESTURE_CLASSES))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=GESTURE_CLASSES, yticklabels=GESTURE_CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()