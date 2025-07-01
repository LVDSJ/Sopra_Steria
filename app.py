from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from hand_sign_utils import count_fingers, detect_hand_sign
import mediapipe as mp

app = Flask(__name__)
mp_hands = mp.solutions.hands

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        header, encoded = data["image"].split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img_array = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Run gesture classification
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7) as hands:
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
                return jsonify({"gesture": "No hand detected"})

            gestures = []
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                fingers = count_fingers(hand_landmarks, handedness)
                sign = detect_hand_sign(fingers, hand_landmarks)
                gestures.append({handedness: sign if sign else "unknown"})

            return jsonify({"gesture": gestures})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)