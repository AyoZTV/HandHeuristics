# Tracks: Face (468 pts) + Left/Right hands (21 pts each)
# Shows: camera overlay + 3 "AI views" (Left hand, Right hand, Face)
# Mac-friendly (AVFoundation), quiet logs, safe UI placement.

# --- hush noisy logs (must be before mediapipe import) ---
import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2, time, numpy as np
from collections import deque

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("Install deps first:\n  python3 -m pip install mediapipe opencv-python numpy\n" + str(e))

# ===== params =====
AI_CANVAS = 384        # square px per AI view
HIST = 30              # history size for potential features
DESIRED_W, DESIRED_H, DESIRED_FPS = 640, 480, 30
MAX_CAM_INDEX = 5

# Mediapipe solutions
mp_hands = mp.solutions.hands
mp_face  = mp.solutions.face_mesh
mp_draw  = mp.solutions.drawing_utils
mp_styles= mp.solutions.drawing_styles

# Drawing specs
HAND_CONN = mp_hands.HAND_CONNECTIONS
FACE_TESSELATION = mp_face.FACEMESH_TESSELATION
FACE_CONTOURS    = mp_face.FACEMESH_CONTOURS

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
    raise SystemExit("No camera. Enable Terminal/VS Code in Settings → Privacy & Security → Camera.")

# ---- helpers ----
def hand_lm_to_np(lms):
    return np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=np.float32)  # (21,3)

def face_lm_to_np(lms):
    return np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark], dtype=np.float32)  # (468,3)

def heatmap(norm_pts, size=64):
    g = np.zeros((size, size), np.float32)
    for x, y, _ in norm_pts:
        gx = int(np.clip(x * (size - 1), 0, size - 1))
        gy = int(np.clip(y * (size - 1), 0, size - 1))
        g[gy, gx] += 1
    g = cv2.GaussianBlur(g, (7, 7), 0)
    m = g.max()
    return g / m if m > 0 else g

def draw_ai(canvas, pts, vel, title, arrow_scale=50, point_step=1):
    """canvas: RGB, pts: (N,3) normalized, vel: (N,2), title: str"""
    h, w = canvas.shape[:2]
    hm = heatmap(pts)
    hm = cv2.applyColorMap((cv2.resize(hm, (w, h)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.addWeighted(hm, 0.6, canvas, 0.4, 0, canvas)

    # draw landmarks + velocity (subsample with point_step to keep it light for face)
    for i in range(0, len(pts), point_step):
        x, y, _ = pts[i]
        px, py = int(x * w), int(y * h)
        cv2.circle(canvas, (px, py), 2, (255, 255, 255), -1)
        if i < len(vel):
            vx, vy = vel[i]
            cv2.arrowedLine(canvas, (px, py), (int(px + vx * arrow_scale), int(py + vy * arrow_scale)),
                            (0, 255, 0), 1, tipLength=0.3)

    cv2.putText(canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas

def safe_paste(dst, src, top, left):
    """Paste src into dst at (top,left) with clipping."""
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    if top >= H or left >= W: return
    h_take = min(h, max(0, H - top))
    w_take = min(w, max(0, W - left))
    if h_take <= 0 or w_take <= 0: return
    dst[top:top + h_take, left:left + w_take] = src[:h_take, :w_take]

def main():
    # perf hints
    cv2.setUseOptimized(True)
    try: cv2.ocl.setUseOpenCL(True)
    except: pass

    cap, idx = open_camera()
    print(f"[info] cam index {idx}")

    # Build solutions
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )
    face = mp_face.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,       # better around eyes/lips
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # histories for velocity (prev frame per stream)
    prev = {"Left": None, "Right": None, "Face": None}
    fps_smooth, t_prev = 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[warn] empty frame"); break

        now = time.time()
        dt = max(now - t_prev, 1e-6); t_prev = now
        fps = 1.0 / dt; fps_smooth = (fps_smooth * 0.9) + (fps * 0.1)

        overlay = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # run detectors
        hand_res = hands.process(rgb)
        face_res = face.process(rgb)

        # hand AI canvases
        ai_L = np.zeros((AI_CANVAS, AI_CANVAS, 3), np.uint8)
        ai_R = np.zeros((AI_CANVAS, AI_CANVAS, 3), np.uint8)
        seen_hand = {"Left": False, "Right": False}

        # FACE AI canvas
        ai_F = np.zeros((AI_CANVAS, AI_CANVAS, 3), np.uint8)
        seen_face = False

        # ---------- HANDS ----------
        if hand_res.multi_hand_landmarks:
            for lm, hd in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
                label = hd.classification[0].label  # "Left" or "Right"
                pts = hand_lm_to_np(lm)

                # draw on camera
                mp_draw.draw_landmarks(
                    overlay, lm, HAND_CONN,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                # velocities in normalized space
                if prev[label] is not None and prev[label].shape == pts.shape:
                    vel = (pts[:, :2] - prev[label][:, :2]) / dt
                else:
                    vel = np.zeros((pts.shape[0], 2), np.float32)
                prev[label] = pts.copy()

                avg_speed = float(np.mean(np.linalg.norm(vel, axis=1)))
                title = f"{label} hand  avg_speed:{avg_speed:.2f}"

                if label == "Left":
                    ai_L = draw_ai(ai_L, pts, vel, title, arrow_scale=60, point_step=1)
                else:
                    ai_R = draw_ai(ai_R, pts, vel, title, arrow_scale=60, point_step=1)
                seen_hand[label] = True
        else:
            prev["Left"] = None; prev["Right"] = None

        # ---------- FACE ----------
        if face_res.multi_face_landmarks:
            flm = face_res.multi_face_landmarks[0]
            fpts = face_lm_to_np(flm)

            # draw on camera (tessellation + contours)
            mp_draw.draw_landmarks(
                overlay, flm, FACE_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )
            mp_draw.draw_landmarks(
                overlay, flm, FACE_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
            )

            if prev["Face"] is not None and prev["Face"].shape == fpts.shape:
                fvel = (fpts[:, :2] - prev["Face"][:, :2]) / dt
            else:
                fvel = np.zeros((fpts.shape[0], 2), np.float32)
            prev["Face"] = fpts.copy()

            # Subsample velocities for face so it stays smooth (every 10th point)
            avg_speed_face = float(np.mean(np.linalg.norm(fvel, axis=1)))
            ai_F = draw_ai(ai_F, fpts, fvel, f"Face avg_speed:{avg_speed_face:.2f}",
                           arrow_scale=40, point_step=10)
            seen_face = True
        else:
            prev["Face"] = None

        # ---------- UI text ----------
        cv2.putText(overlay, f"fps:{fps_smooth:.1f}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        if not any(seen_hand.values()) and not seen_face:
            cv2.putText(overlay, "no face/hands", (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

        # ---------- place the AI panels (top row: Left & Right; second row: Face) ----------
        h, w = overlay.shape[:2]
        side = w // 4  # smaller so two fit side-by-side
        sL = cv2.resize(ai_L, (side, side))
        sR = cv2.resize(ai_R, (side, side))
        sF = cv2.resize(ai_F, (side, side))

        # top row
        safe_paste(overlay, sL, 0, 0)
        safe_paste(overlay, sR, 0, sL.shape[1] + 8)
        # second row
        safe_paste(overlay, sF, sL.shape[0] + 8, 0)

        cv2.imshow("camera + overlay (q to quit)", overlay)
        cv2.imshow("AI Left hand", ai_L)
        cv2.imshow("AI Right hand", ai_R)
        cv2.imshow("AI Face", ai_F)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break

    hands.close()
    face.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
