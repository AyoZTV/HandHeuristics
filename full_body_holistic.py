# Full body realtime tracker (MediaPipe Holistic)
# Tracks: Pose (33), Face (468), Left/Right Hands (21 each)
# Shows: camera overlay + 4 AI panels (Pose, Left, Right, Face)

# --- kill noisy logs BEFORE importing mediapipe/tflite ---
import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2, time, numpy as np

# ===== camera prefs =====
DESIRED_W, DESIRED_H, DESIRED_FPS = 960, 540, 30
MAX_CAM_INDEX = 5
AI_SIDE = 180  # size of each AI panel

# ===== helpers =====
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
    raise SystemExit("No camera. Give Terminal/VS Code camera permission in Settings → Privacy & Security → Camera.")

def np_from_landmarks(lms):
    # list of NormalizedLandmark -> (N,3) float32
    return np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

def heatmap(norm_pts, size=72):
    g = np.zeros((size, size), np.float32)
    if norm_pts is not None:
        for x,y,_ in norm_pts:
            gx = int(np.clip(x*(size-1), 0, size-1))
            gy = int(np.clip(y*(size-1), 0, size-1))
            g[gy, gx] += 1
    g = cv2.GaussianBlur(g, (7,7), 0)
    m = g.max()
    return g/m if m>0 else g

def draw_ai(canvas, pts, vel, title, arrow_scale=50, point_step=1):
    h, w = canvas.shape[:2]
    hm = heatmap(pts)
    hm = cv2.applyColorMap((cv2.resize(hm, (w, h))*255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.addWeighted(hm, 0.6, canvas, 0.4, 0, canvas)
    if pts is not None:
        for i in range(0, len(pts), point_step):
            x,y,_ = pts[i]
            px, py = int(x*w), int(y*h)
            cv2.circle(canvas, (px,py), 2, (255,255,255), -1)
            if vel is not None and i < len(vel):
                vx, vy = vel[i]
                cv2.arrowedLine(canvas, (px,py), (int(px+vx*arrow_scale), int(py+vy*arrow_scale)),
                                (0,255,0), 1, tipLength=0.3)
    cv2.putText(canvas, title, (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return canvas

def safe_paste(dst, src, top, left):
    H,W = dst.shape[:2]; h,w = src.shape[:2]
    if top>=H or left>=W: return
    h_take = min(h, max(0, H-top))
    w_take = min(w, max(0, W-left))
    if h_take<=0 or w_take<=0: return
    dst[top:top+h_take, left:left+w_take] = src[:h_take, :w_take]

def main():
    cv2.setUseOptimized(True)
    try: cv2.ocl.setUseOpenCL(True)
    except: pass

    cap, cam_idx = open_camera()
    print(f"[info] cam index {cam_idx}")

    try:
        import mediapipe as mp
    except Exception as e:
        raise SystemExit("Install deps:\n  python3 -m pip install mediapipe opencv-python numpy\n" + str(e))

    mp_hol   = mp.solutions.holistic
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    hol = mp_hol.Holistic(
        static_image_mode=False,
        model_complexity=1,         # 0=fast, 1=balanced, 2=accurate
        smooth_landmarks=True,
        refine_face_landmarks=True, # better eyes/lips
        enable_segmentation=False,  # set True if you want a people mask
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # previous frames for velocities
    prev = {"pose": None, "left": None, "right": None, "face": None}
    fps_s, t_prev = 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok: break

        now = time.time()
        dt = max(now - t_prev, 1e-6); t_prev = now
        fps = 1.0/dt; fps_s = fps_s*0.9 + fps*0.1

        overlay = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = hol.process(rgb)

        # ---- convert all landmark sets to numpy (normalized) ----
        pose_pts = np_from_landmarks(res.pose_landmarks.landmark) if res.pose_landmarks else None
        left_pts = np_from_landmarks(res.left_hand_landmarks.landmark) if res.left_hand_landmarks else None
        right_pts= np_from_landmarks(res.right_hand_landmarks.landmark) if res.right_hand_landmarks else None
        face_pts = np_from_landmarks(res.face_landmarks.landmark) if res.face_landmarks else None

        # ---- draw on camera overlay ----
        if res.pose_landmarks:
            mp_draw.draw_landmarks(
                overlay, res.pose_landmarks, mp_hol.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
            )
        if res.left_hand_landmarks:
            mp_draw.draw_landmarks(
                overlay, res.left_hand_landmarks, mp_hol.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )
        if res.right_hand_landmarks:
            mp_draw.draw_landmarks(
                overlay, res.right_hand_landmarks, mp_hol.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )
        if res.face_landmarks:
            mp_draw.draw_landmarks(
                overlay, res.face_landmarks, mp_hol.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_style.get_default_face_mesh_tesselation_style()
            )
            mp_draw.draw_landmarks(
                overlay, res.face_landmarks, mp_hol.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_style.get_default_face_mesh_contours_style()
            )

        # ---- velocities (normalized space) ----
        def vel(curr, key):
            if curr is None: 
                prev[key] = None
                return None
            p = prev[key]
            prev[key] = curr.copy()
            if p is None or p.shape != curr.shape:
                return np.zeros((curr.shape[0], 2), np.float32)
            return (curr[:, :2] - p[:, :2]) / dt

        v_pose  = vel(pose_pts,  "pose")
        v_left  = vel(left_pts,  "left")
        v_right = vel(right_pts, "right")
        v_face  = vel(face_pts,  "face")

        # ---- AI panels (pose, L hand, R hand, face) ----
        panel_pose = np.zeros((AI_SIDE, AI_SIDE, 3), np.uint8)
        panel_left = np.zeros((AI_SIDE, AI_SIDE, 3), np.uint8)
        panel_right= np.zeros((AI_SIDE, AI_SIDE, 3), np.uint8)
        panel_face = np.zeros((AI_SIDE, AI_SIDE, 3), np.uint8)

        # subsample for face so it stays smooth
        avg_pose_spd  = 0.0 if v_pose  is None else float(np.mean(np.linalg.norm(v_pose,  axis=1)))
        avg_left_spd  = 0.0 if v_left  is None else float(np.mean(np.linalg.norm(v_left,  axis=1)))
        avg_right_spd = 0.0 if v_right is None else float(np.mean(np.linalg.norm(v_right, axis=1)))
        avg_face_spd  = 0.0 if v_face  is None else float(np.mean(np.linalg.norm(v_face,  axis=1)))

        draw_ai(panel_pose, pose_pts,  v_pose,  f"Pose avg:{avg_pose_spd:.2f}",  arrow_scale=60, point_step=1)
        draw_ai(panel_left, left_pts,  v_left,  f"Left avg:{avg_left_spd:.2f}",  arrow_scale=70, point_step=1)
        draw_ai(panel_right,right_pts, v_right, f"Right avg:{avg_right_spd:.2f}", arrow_scale=70, point_step=1)
        draw_ai(panel_face, face_pts,  v_face,  f"Face avg:{avg_face_spd:.2f}",  arrow_scale=40, point_step=10)

        # ---- HUD ----
        cv2.putText(overlay, f"fps:{fps_s:.1f}", (10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        if pose_pts is None and left_pts is None and right_pts is None and face_pts is None:
            cv2.putText(overlay, "no body/face/hands", (10,56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

        # ---- place panels: top row (Pose, Left), bottom row (Right, Face) ----
        sP = cv2.resize(panel_pose,  (AI_SIDE, AI_SIDE))
        sL = cv2.resize(panel_left,  (AI_SIDE, AI_SIDE))
        sR = cv2.resize(panel_right, (AI_SIDE, AI_SIDE))
        sF = cv2.resize(panel_face,  (AI_SIDE, AI_SIDE))

        safe_paste(overlay, sP, 0, 0)
        safe_paste(overlay, sL, 0, sP.shape[1] + 8)
        safe_paste(overlay, sR, sP.shape[0] + 8, 0)
        safe_paste(overlay, sF, sP.shape[0] + 8, sR.shape[1] + 8)

        # ---- show ----
        cv2.imshow("full body + hands + face (q to quit)", overlay)
        cv2.imshow("AI Pose",  panel_pose)
        cv2.imshow("AI Left",  panel_left)
        cv2.imshow("AI Right", panel_right)
        cv2.imshow("AI Face",  panel_face)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break

    hol.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
