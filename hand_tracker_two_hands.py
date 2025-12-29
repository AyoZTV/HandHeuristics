# Two-hand realtime tracker + per-hand "AI view" (Mac-friendly)

# quiet logs
import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2, time, numpy as np
from collections import deque

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("Install deps:\n  python3 -m pip install mediapipe opencv-python numpy\n" + str(e))

# ---- params
AI_CANVAS = 384
HIST = 30
DESIRED_W, DESIRED_H, DESIRED_FPS = 640, 480, 30
MAX_CAM_INDEX = 5

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
CONN     = mp_hands.HAND_CONNECTIONS

def open_camera():
    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else 0
    for idx in range(MAX_CAM_INDEX):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_H)
            cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)
            return cap, idx
        cap.release()
    cap = cv2.VideoCapture(0)
    if cap.isOpened(): return cap, 0
    raise SystemExit("No camera. Enable Terminal/VSCode in Settings → Privacy & Security → Camera.")

def lm_to_np(lms):
    return np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=np.float32)  # (21,3)

def heatmap(norm_pts, size=64):
    g = np.zeros((size,size), np.float32)
    for x,y,_ in norm_pts:
        gx = int(np.clip(x*(size-1),0,size-1)); gy = int(np.clip(y*(size-1),0,size-1))
        g[gy,gx]+=1
    g = cv2.GaussianBlur(g,(7,7),0)
    m=g.max()
    return g/m if m>0 else g

def draw_ai(canvas, pts, vel, title):
    h,w = canvas.shape[:2]
    hm = heatmap(pts)
    hm = cv2.applyColorMap((cv2.resize(hm,(w,h))*255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.addWeighted(hm,0.6,canvas,0.4,0,canvas)
    for i,(x,y,_) in enumerate(pts):
        px,py = int(x*w), int(y*h)
        cv2.circle(canvas,(px,py),4,(255,255,255),-1)
        vx,vy = vel[i] if i < len(vel) else (0,0)
        cv2.arrowedLine(canvas,(px,py),(int(px+vx*50),int(py+vy*50)),(0,255,0),1,tipLength=0.3)
    cv2.putText(canvas, title, (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return canvas

def main():
    cv2.setUseOptimized(True)
    try: cv2.ocl.setUseOpenCL(True)
    except: pass

    cap, idx = open_camera()
    print(f"[info] cam index {idx}")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,                 # ← key change
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    # keep prev landmarks per handedness label ("Left"/"Right")
    prev = {"Left": None, "Right": None}
    history = {"Left": deque(maxlen=HIST), "Right": deque(maxlen=HIST)}
    fps_smooth, t_prev = 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok: break
        dt = max(time.time()-t_prev, 1e-6); t_prev = time.time()
        fps = 1.0/dt; fps_smooth = fps_smooth*0.9 + fps*0.1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        overlay = frame.copy()

        # prep two AI canvases
        ai_L = np.zeros((AI_CANVAS, AI_CANVAS,3), np.uint8)
        ai_R = np.zeros((AI_CANVAS, AI_CANVAS,3), np.uint8)
        seen = {"Left": False, "Right": False}

        if res.multi_hand_landmarks:
            # pair landmarks with handedness labels
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = hd.classification[0].label  # "Left" or "Right"
                pts = lm_to_np(lm)
                # draw on camera
                mp_draw.draw_landmarks(overlay, lm, CONN)

                # velocities in normalized space
                if prev[label] is not None and prev[label].shape == pts.shape:
                    vel = (pts[:,:2] - prev[label][:,:2]) / dt
                else:
                    vel = np.zeros((21,2), np.float32)
                prev[label] = pts.copy()
                history[label].append(pts)
                avg_speed = float(np.mean(np.linalg.norm(vel, axis=1)))

                # pick canvas
                if label == "Left":
                    ai_L = draw_ai(ai_L, pts, vel, f"Left  avg_speed:{avg_speed:.2f}")
                else:
                    ai_R = draw_ai(ai_R, pts, vel, f"Right avg_speed:{avg_speed:.2f}")
                seen[label] = True

        # UI text
        cv2.putText(overlay, f"fps:{fps_smooth:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        if not any(seen.values()):
            cv2.putText(overlay, "no hands", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

        # place both AI views on top-left (stacked)
        h, w = overlay.shape[:2]
        side = w // 3
        sL = cv2.resize(ai_L, (side, side))
        sR = cv2.resize(ai_R, (side, side))

        # Left panel at top-left
        overlay[0:sL.shape[0], 0:sL.shape[1]] = sL

        # Right panel directly below Left (clipped if not enough space)
        y2 = sL.shape[0] + 8
        h_take = min(sR.shape[0], max(0, h - y2))
        if h_take > 0:
            overlay[y2:y2 + h_take, 0:sR.shape[1]] = sR[0:h_take, :]


        cv2.imshow("camera + overlay (q to quit)", overlay)
        cv2.imshow("AI Left", ai_L)
        cv2.imshow("AI Right", ai_R)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
