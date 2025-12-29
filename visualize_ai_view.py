# visualize_ai_view.py
# Usage: python3 visualize_ai_view.py saved/2025xxxx_xxxxxx.npy
# Shows landmark points, 30-frame trails, and simple finger angles.

import sys, math
import numpy as np
import cv2

def angle_between(a, b):
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0: return 0.0
    cos = max(-1.0, min(1.0, dot / (na * nb)))
    return math.degrees(math.acos(cos))

def finger_angles(frame21):
    # indices per MediaPipe: 0 wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring; 17-20 pinky
    mapping = {"thumb": (1,2,3), "index": (5,6,7), "middle": (9,10,11), "ring": (13,14,15), "pinky": (17,18,19)}
    wrist = frame21[0, :2]
    out = {}
    for name, (mcp, pip, dip) in mapping.items():
        v1 = frame21[mcp, :2] - wrist
        v2 = frame21[pip, :2] - frame21[mcp, :2]
        out[name] = angle_between(v1, v2)
    return out

def main(path):
    seq = np.load(path)  # (frames, 21, 3) in normalized coords
    if seq.ndim != 3 or seq.shape[1] != 21 or seq.shape[2] != 3:
        raise SystemExit(f"Bad shape {seq.shape}. Expected (frames, 21, 3).")

    H, W = 600, 600
    trail = np.zeros((H, W, 3), dtype=np.uint8)

    for i in range(len(seq)):
        frame = seq[i]
        pts = (frame[:, :2] * [W, H]).astype(int)

        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for p in pts:
            cv2.circle(canvas, tuple(p), 3, (255,255,255), -1)

        # draw short trails (last 30 frames)
        j0 = max(0, i - 30)
        for j in range(j0, i):
            a = (seq[j][:, :2] * [W, H]).astype(int)
            b = (seq[j+1][:, :2] * [W, H]).astype(int)
            for k in range(21):
                cv2.line(trail, tuple(a[k]), tuple(b[k]), (60,60,200), 1)

        # angles
        ang = finger_angles(frame)
        txt = " | ".join(f"{k}:{v:.0f}°" for k, v in ang.items())
        cv2.putText(canvas, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        display = np.hstack([canvas, trail])
        cv2.imshow("AI playback — landmarks | trails", display)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('q') or k == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 visualize_ai_view.py saved/XXXX.npy")
    else:
        main(sys.argv[1])
