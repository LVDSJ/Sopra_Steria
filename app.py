from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from hand_sign_utils import count_fingers, detect_hand_sign

app = Flask(__name__)
mp_hands = mp.solutions.hands

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        image_data = data["image"].split(",")[1]  # remove 'data:image/...;base64,'
        decoded = base64.b64decode(image_data)
        image = Image.open(BytesIO(decoded)).convert("RGB")
        image_rgb = np.array(image)

        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7) as hands:
            results = hands.process(image_rgb)
            predictions = []
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0].label
                    fingers = count_fingers(hand_landmarks, handedness)
                    sign = detect_hand_sign(fingers, hand_landmarks)
                    predictions.append({"hand": handedness, "sign": sign})
            else:
                predictions.append({"message": "No hands detected"})

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)