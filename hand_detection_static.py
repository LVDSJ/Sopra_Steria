import cv2
import mediapipe as mp
from hand_sign_utils import count_fingers, detect_hand_sign

mp_hands = mp.solutions.hands

def classify_gesture_on_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7) as hands:
        results = hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            print("No hands detected.")
            return

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            fingers = count_fingers(hand_landmarks, handedness)
            sign = detect_hand_sign(fingers, hand_landmarks)

            # Only print if valid sign
            if sign and sign.lower() not in ["none", "unknown"]:
                print(f"Hand {handedness}: {sign}")

# Example usage
classify_gesture_on_image("images/peace/360_F_300231495_K2LTWbLCdjkFZCiCNzg2IG6IwY51T4Ge.jpg")