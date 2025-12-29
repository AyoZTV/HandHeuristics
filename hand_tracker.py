# hand_tracker.py
# Realtime hand/finger tracker using MediaPipe.
# Outputs:
#  - OpenCV window with landmarks
#  - "AI view" window showing normalized landmarks, velocities, and simple heatmap

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# params
MAX_HISTORY = 30  # frames to compute velocity / small history
SCALE_WINDOW = 512  # visualization canvas size

def normalized_landmarks_to_array(landmarks, img_w, img_h):
    arr = []
    for lm in landmarks.landmark:
        arr.append([lm.x, lm.y, lm.z])  # normalized x,y,z (z is relative depth)
    return np.array(arr)  # shape (21,3)

def draw_ai_view(canvas, norm_pts, velocities, heatmap=None):
    # canvas: square RGB numpy
    h, w = canvas.shape[:2]
    # draw points and velocity vectors
    for i, (x,y,z) in enumerate(norm_pts):
        px = int(x * w)
        py = int(y * h)
        cv2.circle(canvas, (px, py), 4, (255,255,255), -1)
        vx, vy = velocities[i]
        cv2.arrowedLine(canvas, (px, py), (int(px+vx*50), int(py+vy*50)), (0,255,0), 1, tipLength=0.3)
    if heatmap is not None:
        # overlay heatmap (normalized)
        heatmap_img = cv2.resize(heatmap, (w,h))
        heatmap_img = cv2.applyColorMap((heatmap_img*255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.addWeighted(heatmap_img, 0.6, canvas, 0.4, 0, canvas)
    return canvas

def landmarks_heatmap(norm_pts, size=64):
    # simple gaussian splat of points into small grid -> upsample for visualization
    grid = np.zeros((size,size), dtype=np.float32)
    for (x,y,_) in norm_pts:
        gx = int(np.clip(x * (size-1), 0, size-1))
        gy = int(np.clip(y * (size-1), 0, size-1))
        grid[gy, gx] += 1.0
    # gaussian blur (approx)
    grid = cv2.GaussianBlur(grid, (7,7), 0)
    grid = grid / (grid.max() + 1e-8)
    return grid

def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    history = deque(maxlen=MAX_HISTORY)  # store previous landmark arrays
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        ai_canvas = np.zeros((SCALE_WINDOW, SCALE_WINDOW, 3), dtype=np.uint8)
        overlay = frame.copy()

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            pts = normalized_landmarks_to_array(hand, w, h)  # (21,3)
            history.append(pts)

            # compute velocities (simple diff between last and current in normalized coords)
            if len(history) >= 2:
                prev = history[-2]
                curr = history[-1]
                velocities = (curr[:, :2] - prev[:, :2]) / (1/30)  # approx per-second (assuming ~30fps)
            else:
                velocities = np.zeros((pts.shape[0], 2), dtype=np.float32)

            # draw camera overlay
            mp_drawing.draw_landmarks(overlay, hand, mp_hands.HAND_CONNECTIONS)

            # AI view: normalized coords, velocities, heatmap
            heat = landmarks_heatmap(pts)
            ai_canvas = draw_ai_view(ai_canvas, pts, velocities, heatmap=heat)

            # show numerical "AI" data as text (top-left)
            info = f"landmarks: {pts.shape}  avg_speed:{np.mean(np.linalg.norm(velocities,axis=1)):.2f}"
            cv2.putText(overlay, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # compose windows
        small_ai = cv2.resize(ai_canvas, (w//2, h//2))
        overlay[0:small_ai.shape[0], 0:small_ai.shape[1]] = small_ai

        cv2.imshow('camera + overlay (press q to quit)', overlay)
        cv2.imshow('AI view (normalized landmarks + heatmap + velocities)', ai_canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
