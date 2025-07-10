import os
import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from hand_sign_utils import count_fingers, detect_hand_sign

# Initialize MediaPipe Hands solution for landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7)

# List of gesture classses in test images
GESTURE_CLASSES = ["peace_sign", "surfing_sign", "good_luck_sign", "open_hand"]
IMAGE_DIR = 'images'

# Initialize the MediaPipe GestureRecognizer with a pre-trained model
base_options = mp.tasks.BaseOptions(model_asset_path='gesture_recognizer/gesture_recognizer.task')
options = mp.tasks.vision.GestureRecognizerOptions(base_options=base_options)
gesture_recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)


# Classify images
def classify_gesture(image_path):
    """Return one of GESTURE_CLASSES or 'none'."""
    image = cv2.imread(image_path)
    if image is None:
        return "none"
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run MediaPipe GestureRecognizer
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    gesture_results = gesture_recognizer.recognize(mp_image)
    mediapipe_gesture = None
    if gesture_results.gestures and gesture_results.gestures[0]:
        mediapipe_gesture = gesture_results.gestures[0][0]

    # Run MediaPipe Hands
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return "none"

    handedness = results.multi_handedness[0].classification[0].label
    landmarks = results.multi_hand_landmarks[0]

    # Finger count + unified detection
    fingers = count_fingers(landmarks, handedness)
    sign = detect_hand_sign(fingers, landmarks, handedness, mediapipe_gesture)

    # Map to correct label set
    if sign:
        key = sign.lower().replace(" ", "_")
        mapping = {
            "peace_sign":     "peace_sign",
            "open_hand":      "open_hand",
            "surfing_sign":   "surfing_sign",
            "good_luck_sign": "good_luck_sign",
        }
        return mapping.get(key, "none")

    return "none"


def evaluate_model():
    y_true, y_pred = [], []
    total, processed = 0, 0
    # List to collect misclassified examples
    misclassified = []

    # Loop over each true gesture class folder
    for gesture in GESTURE_CLASSES:
        folder = os.path.join(IMAGE_DIR, gesture)
        if not os.path.isdir(folder):
            print(f"Warning: missing folder for '{gesture}' – skipping")
            continue

        # Gather image filenames
        images = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg'))]
        total += len(images)

        # Classify each image and record prediction vs truth
        for fn in images:
            path = os.path.join(folder, fn)
            pred = classify_gesture(path)
            y_true.append(gesture)
            y_pred.append(pred)
            processed += 1

            # Track misclassifications for review
            if pred != gesture:
                misclassified.append((path, gesture, pred))

    if not y_true:
        print("No images found for evaluation.")
        return

    labels = GESTURE_CLASSES 

    # Compute metrics
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    # Overall accuracy
    accuracy = np.mean([t == p for t, p in zip(y_true, y_pred)]) * 100
    precision = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0) * 100
    recall = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0) * 100

    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Overall Precision: {precision:.2f}%")

    # Print misclassified examples
    if misclassified:
        print("Misclassified examples:")
        for path, true, pred in misclassified:
            print(f"  {path} → predicted: {pred} (true: {true})")
    else:
        print("All images classified correctly!")

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Acc: {accuracy:.2f}%, Prcs: {precision:.2f}%)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_model()
    # Clean up the MediaPipe model
    gesture_recognizer.close()