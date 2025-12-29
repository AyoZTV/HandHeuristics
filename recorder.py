# recorder.py
# Press R to start/stop recording sequences of landmarks.
# Files saved to ./saved/YYYYMMDD_HHMMSS.npy with shape: (frames, 21, 3)

# ---- quiet logs BEFORE imports that trigger TF/absl ----
import os
os.makedirs("saved", exist_ok=True)
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import cv2

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("Install deps first:\n  python3 -m pip install mediapipe opencv-python numpy\n" + str(e))

from hand_tracker import normalized_landmarks_to_array  # reuse helper

# cam prefs
MAX_CAM_INDEX = 5
DESIRED_W, DESIRED_H, DESIRED_FPS = 640, 480, 30

def open_camera():
    backend_flag = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else 0
    for idx in range(MAX_CAM_INDEX):
        cap = cv2.VideoCapture(idx, backend_flag)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_H)
            cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)
            return cap, idx
        cap.release()
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap, 0
    raise SystemExit("No camera. Give Terminal/VS Code camera permission in System Settings → Privacy & Security → Camera.")

def main():
    cv2.setUseOptimized(True)
    try: cv2.ocl.setUseOpenCL(True)
    except: pass

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap, idx = open_camera()
    print(f"[info] recording from cam index {idx}. Keys: r=start/stop, q=quit")

    recording = False
    buffer = []
    last_save_msg = ""

    while True:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            pts = normalized_landmarks_to_array(res.multi_hand_landmarks[0])
            cv2.putText(frame, "hand detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            pts = None
            cv2.putText(frame, "no hand", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

        if recording and pts is not None:
            buffer.append(pts)

        # UI text
        status = "REC ●" if recording else "IDLE"
        cv2.putText(frame, f"{status}   frames:{len(buffer)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if last_save_msg:
            cv2.putText(frame, last_save_msg, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,220,255), 2)

        cv2.imshow("recorder (r=start/stop, q=quit)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('r'):
            recording = not recording
            if not recording and buffer:
                fname = time.strftime("%Y%m%d_%H%M%S") + ".npy"
                path = os.path.join("saved", fname)
                arr = np.array(buffer, dtype=np.float32)
                np.save(path, arr)
                last_save_msg = f"saved: {path}  shape {arr.shape}"
                print(last_save_msg)
                buffer = []
            elif recording:
                last_save_msg = "recording…"
        elif k == ord('q') or k == 27:
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
