from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from hand_sign_utils import count_fingers, detect_hand_sign
from mediapipe.tasks import python as mp_tasks

app = Flask(__name__)
mp_hands = mp.solutions.hands

# Initialize MediaPipe GestureRecognizer
base_options = mp_tasks.BaseOptions(model_asset_path='gesture_recognizer/gesture_recognizer.task')
gesture_options = mp.tasks.vision.GestureRecognizerOptions(base_options=base_options)
gesture_recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(gesture_options)

# Initialize MediaPipe Hands once
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

import atexit
atexit.register(lambda: gesture_recognizer.close())
atexit.register(lambda: hands.close())

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        image_data = data["image"].split(",")[1] 
        decoded = base64.b64decode(image_data)
        image = Image.open(BytesIO(decoded)).convert("RGB")
        image_rgb = np.array(image)

        # Built-in MediaPipe gesture recognition
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        gesture_results = gesture_recognizer.recognize(mp_image)

        results = hands.process(image_rgb)
        predictions = []
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                fingers = count_fingers(hand_landmarks, handedness)
                mediapipe_gesture = None
                if gesture_results.gestures and idx < len(gesture_results.gestures) and gesture_results.gestures[idx]:
                    mediapipe_gesture = gesture_results.gestures[idx][0]
                sign = detect_hand_sign(fingers, hand_landmarks, handedness, mediapipe_gesture)
                predictions.append({"hand": handedness, "sign": sign})
        else:
            predictions.append({"message": "No hands detected"})

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)