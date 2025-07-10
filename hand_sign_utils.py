# hand_sign_utils.py
import numpy as np
from scipy.signal import find_peaks

def count_fingers(hand_landmarks, handedness):
    fingers = []  # List to hold binary values: 1 = finger extended, 0 = folded

    # Landmark indices for finger tips:
    # Thumb (4), Index (8), Middle (12), Ring (16), Pinky (20)
    tip_ids = [4, 8, 12, 16, 20]

    # ----- Thumb detection (uses x-axis because thumb extends sideways) -----
    if handedness == "Right":
        # For the right hand, the thumb tip (4) should be to the *left* of its base joint (3)
        is_thumb_extended = hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 2].x
    else:
        # For the left hand, the thumb tip (4) should be to the *right* of its base joint (3)
        is_thumb_extended = hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 2].x
    
    fingers.append(1 if is_thumb_extended else 0)

    # ----- Other fingers (Index to Pinky) -----
    # If the tip is above the middle joint, the finger is extended
    for i in range(1, 5):
        # tip_ids[i]: the tip of the finger
        # tip_ids[i] - 2: the middle joint of the finger
        is_extended = hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y
        fingers.append(1 if is_extended else 0)

    return fingers

def detect_hand_sign(fingers, hand_landmarks, handedness, mediapipe_gesture=None):
    """
    Detect hand signs using both custom logic and MediaPipe gesture recognition
    
    Args:
        fingers: List of finger states from count_fingers
        hand_landmarks: MediaPipe hand landmarks
        handedness: "Left" or "Right"
        mediapipe_gesture: Gesture result from MediaPipe GestureRecognizer
    
    Returns:
        String describing the detected gesture or None
    """
    # Use MediaPipe's gesture recognition for Open hand and Peace sign
    if mediapipe_gesture:
        gesture_name = mediapipe_gesture.category_name
        confidence = mediapipe_gesture.score
        
        # Only use high-confidence MediaPipe results
        if confidence > 0.5:
            if gesture_name == "Open_Palm":
                return "Open Hand"
            elif gesture_name == "Victory":
                return "Peace Sign"
    
    # Landmarks
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]

    # Good Luck Gesture
    if handedness == "Right":
        crossed = index_tip.x > middle_tip.x
    else:
        crossed = index_tip.x < middle_tip.x

    # --- Custom Gesture Checks ---
    if fingers == [0, 1, 1, 0, 0] and crossed:
        return "Good Luck Sign"

    if fingers == [1, 0, 0, 0, 1]:
        return "Surfing Sign"

    return None
    

def detect_wave(buffer, min_peaks=2, min_amplitude=0.025):
    """    
    Args:
        buffer:         length of array of last x positions should be higher than 10
        min_peaks:      peak (hand moves far to the rigth and turns back)
        min_amplitude:  minimum horizontal movement (max(x)-min(x)) to 
                        count a wave (the lower the smaller movement will be detect as wave)
    """
    if len(buffer) < 10:
        return False
    x_positions = np.array(buffer)
    smoothed = x_positions  # no smoothing performed (for fast movements)

    # Detect peaks, 
    peaks, _ = find_peaks(smoothed, prominence=0.01) # right and back
    troughs, _ = find_peaks(-smoothed, prominence=0.01) # left and back
    total_peaks = len(peaks) + len(troughs)

    motion_range = np.max(x_positions) - np.min(x_positions)
    return total_peaks >= min_peaks and motion_range >= min_amplitude