import cv2
import numpy as np
from collections import deque
from hand_sign_utils import count_fingers, detect_hand_sign, detect_wave
import time
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create an GestureRecognizer object
base_options = mp.tasks.BaseOptions(model_asset_path='gesture_recognizer/gesture_recognizer.task')
options = mp.tasks.vision.GestureRecognizerOptions(base_options=base_options)
gesture_recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

# Set motion buffer (sweat spot was around 30)
motion_buffers = {"Left": deque(maxlen=30), "Right": deque(maxlen=30)}

# Initiate video capture 
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
with mp_hands.Hands(min_detection_confidence=0.7, 
                    min_tracking_confidence=0.5, 
                    max_num_hands=2) as hands:
    
    while cap.isOpened():  # Main video loop
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frame is captured

        frame = cv2.flip(frame, 1)  # Flip horizontally for mirror view (more natural)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe

        start_time = time.time()  # Start timing inference

        results = hands.process(image_rgb)  # Run hand landmark detection
        
        # ----- Gesture recognizer block -----
        gesture_results = None
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        gesture_results = gesture_recognizer.recognize(mp_image)  # Run gesture recognition

        inference_time = int((time.time() - start_time) * 1000)  # Measure time in ms
        hand_signs = []  # List to collect detected hand signs

        if results.multi_hand_landmarks:  # If hands are detected
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label  # Left or Right hand

                # Draw hand landmarks on frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Try to fetch MediaPipe's pre-trained gesture result
                mediapipe_gesture = None
                if gesture_results and gesture_results.gestures:
                    if idx < len(gesture_results.gestures) and gesture_results.gestures[idx]:
                        mediapipe_gesture = gesture_results.gestures[idx][0]

                # ----- Custom gesture detection -----
                fingers = count_fingers(hand_landmarks, handedness)  # Count extended fingers
                sign = detect_hand_sign(fingers, hand_landmarks, handedness, mediapipe_gesture)  # Classify gesture with correct naming

                # ----- Wave detection -----
                wrist_x = hand_landmarks.landmark[0].x  # X-position of the wrist
                motion_buffers[handedness].append(wrist_x)  # Store x-position in motion buffer

                # Detect wave if hand is open and motion pattern fits
                if sign == "Open Hand" and detect_wave(motion_buffers[handedness]):
                    sign = "Waving"

                # Collect valid gesture for display
                if sign and sign.lower() not in ["unknown", "none"]:
                    hand_signs.append(f"{sign} ({handedness})")
                    
        else:
            # If no hands are visible, clear motion buffers to reset wave detection
            motion_buffers["Left"].clear()
            motion_buffers["Right"].clear()

        # ----- UI Overlay -----
        if hand_signs:
            box_x, box_y = 20, 20  # Top-left corner of overlay box
            box_width, box_height = 350, 40 + 30 * len(hand_signs)  # Dynamic height based on number of signs

            # Draw white rectangle for info box
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), -1)

            # Display inference time
            cv2.putText(frame, f"Inference time: {inference_time}ms", (box_x + 10, box_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # Display detected hand signs
            for i, label in enumerate(hand_signs):
                cv2.putText(frame, label, (box_x + 10, box_y + 55 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # ----- Display final frame -----
        cv2.imshow("Hand Sign Detection (Video)", frame)
        
        # Exit on 'Esc' key press
        if cv2.waitKey(10) & 0xFF == 27:
            break

# Clean up resources
cap.release()
cv2.destroyAllWindows()