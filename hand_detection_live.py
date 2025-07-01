import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from hand_sign_utils import count_fingers, detect_hand_sign, detect_wave
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

motion_buffers = {"Left": deque(maxlen=20), "Right": deque(maxlen=20)}
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        results = hands.process(image_rgb)
        inference_time = int((time.time() - start_time) * 1000)

        hand_signs = []

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Gesture detection
                fingers = count_fingers(hand_landmarks, handedness)
                sign = detect_hand_sign(fingers, hand_landmarks)

                # Wave detection
                wrist_x = hand_landmarks.landmark[0].x
                motion_buffers[handedness].append(wrist_x)
                if sign == "Open hand" and detect_wave(motion_buffers[handedness]):
                    sign = "Waving"

                if sign and sign.lower() not in ["unknown", "none"]:
                    hand_signs.append(f"{sign} ({handedness})")
        else:
            motion_buffers["Left"].clear()
            motion_buffers["Right"].clear()

        # ----- White box overlay -----
        if hand_signs:
            box_x, box_y = 20, 20
            box_width, box_height = 350, 40 + 30 * len(hand_signs)
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), -1)

            cv2.putText(frame, f"Inference time: {inference_time}ms", (box_x + 10, box_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            for i, label in enumerate(hand_signs):
                cv2.putText(frame, label, (box_x + 10, box_y + 55 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.imshow("Hand Sign Detection (Video)", frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()