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

def are_fingers_clustered(hand_landmarks, threshold_ratio=0.6):
    ids = [8, 12, 16, 20]  # index, middle, ring, pinky
    x_coords = [hand_landmarks.landmark[i].x for i in ids]

    # Estimate hand width
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    hand_width = abs(middle_mcp.x - wrist.x) + 1e-6  # avoid div by zero

    std_x = np.std(x_coords)
    normalized_std = std_x / hand_width

    return normalized_std < threshold_ratio

def detect_hand_sign(fingers, hand_landmarks, handedness):
    total_fingers = sum(fingers)

    # Landmarks
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    wrist = hand_landmarks.landmark[0]
    index_mcp = hand_landmarks.landmark[5]

    # Circle Game Detection
    dx = thumb_tip.x - index_tip.x
    dy = thumb_tip.y - index_tip.y
    dist_thumb_index = np.sqrt(dx**2 + dy**2)
    dx_ref = index_tip.x - index_mcp.x
    dy_ref = index_tip.y - index_mcp.y
    ref_index_length = np.sqrt(dx_ref**2 + dy_ref**2)
    ratio = dist_thumb_index / (ref_index_length + 1e-6)
    is_upside_down = wrist.y < index_tip.y

    # Good Luck Gesture
    if handedness == "Right":
        crossed = index_tip.x > middle_tip.x
    else:
        crossed = index_tip.x < middle_tip.x

    # WYW Detection
    knuckle_ids = [5, 9, 13, 17]
    knuckle_zs = [hand_landmarks.landmark[i].z for i in knuckle_ids]
    avg_knuckle_z = np.mean(knuckle_zs)
    wyw = wrist.z > avg_knuckle_z + 0.2

    # --- Gesture Checks ---
    if ratio < 0.35 and is_upside_down:
        return "You Lost!"

    if total_fingers == 5:
        
        if are_fingers_clustered(hand_landmarks):
            return "What do you want?"
        else:
            return "Open hand"

    if fingers == [0, 1, 1, 0, 0] and crossed:
        return "Good Luck"

    if fingers == [1, 0, 0, 0, 1]:
        return "Surf hand"

    if fingers == [0, 1, 1, 0, 0] and sum(fingers[0:1] + fingers[3:]) <= 1:
        return "Peace"

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