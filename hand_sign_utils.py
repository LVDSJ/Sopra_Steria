# hand_sign_utils.py
import numpy as np

def count_fingers(hand_landmarks, handedness):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    if handedness == "Right":
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x else 0)

    for i in range(1, 5):
        fingers.append(1 if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y else 0)
    return fingers

def detect_hand_sign(fingers, hand_landmarks):
    total_fingers = sum(fingers)

    # Landmarks
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    wrist = hand_landmarks.landmark[0]
    index_mcp = hand_landmarks.landmark[5]

    # 2D distance between thumb and index
    dx = thumb_tip.x - index_tip.x
    dy = thumb_tip.y - index_tip.y
    dist_thumb_index = np.sqrt(dx**2 + dy**2)

    # Reference: length of index finger (tip to MCP)
    dx_ref = index_tip.x - index_mcp.x
    dy_ref = index_tip.y - index_mcp.y
    ref_index_length = np.sqrt(dx_ref**2 + dy_ref**2)

    # Ratio (normalized distance)
    ratio = dist_thumb_index / (ref_index_length + 1e-6)

    # Hand upside down
    is_upside_down = wrist.y < index_tip.y

    # --- Gesture Checks ---
    if ratio < 0.35 and is_upside_down:
        return "Circle Game"

    if fingers == [1, 0, 0, 0, 1]:
        return "Surf hand"

    if fingers[1] == 1 and fingers[2] == 1 and sum(fingers[0:1] + fingers[3:]) <= 1:
        return "Peace hand"

    if total_fingers == 5:
        return "Open hand"

    return None
    
def detect_wave(buffer, min_peaks=3, min_amplitude=0.05):
    if len(buffer) < 15:
        return False
    x_positions = np.array(buffer)
    smoothed = np.convolve(x_positions, np.ones(3)/3, mode='valid') if len(x_positions) > 3 else x_positions
    if len(smoothed) < 5:
        return False
    diffs = np.diff(smoothed)
    sign_changes = sum((diffs[i-1] > 0 and diffs[i] < 0) or (diffs[i-1] < 0 and diffs[i] > 0) for i in range(1, len(diffs)))
    motion_range = np.max(x_positions) - np.min(x_positions)
    return sign_changes >= min_peaks and motion_range >= min_amplitude