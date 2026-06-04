"""
app.py — Flask application factory for College Management System
"""
import os
from flask import Flask, redirect, url_for, Response
from flask_login import LoginManager, login_required
from flask_mail import Mail

from config import Config
from models.db import db, bcrypt, Student, Staff, HOD, Principal

login_manager = LoginManager()
mail = Mail()


def gen_frames(dept_name):
    """
    Generator yielding MJPEG frames for the live camera feed.
    Includes:
      - Person detection & centroid tracking (Person 1, Person 2, ...)
      - Sleeping detection (EAR + head pitch)
      - Mobile usage, hand raising, standing/sitting detection
      - HUD overlay: student count, sleeping count
    """
    import cv2
    import time
    import math
    import datetime
    import numpy as np
    from collections import OrderedDict
    from camera_service import (
        YOLO_AVAILABLE, MP_AVAILABLE,
        save_snapshot, insert_alert, get_dept_info,
        eye_aspect_ratio, boxes_iou, keypoint_velocity,
        ALERT_COOLDOWN, EAR_THRESHOLD, EAR_CONSEC_FRAMES,
        HEAD_PITCH_FRAMES, FIGHT_IOU_THRESHOLD, POSE_VEL_THRESHOLD,
        MOBILE_CLASS_ID, HAND_RAISE_MARGIN, ALERT_ICONS,
    )

    # ── Department info ──────────────────────────────────────────────────────
    dept_info = get_dept_info(dept_name)
    dept_id   = dept_info["dept_id"] if dept_info else 1

    # ── Open webcam ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"[Flask Camera] Could not open webcam.")
        return

    # ── Load models ──────────────────────────────────────────────────────────
    yolo_det = yolo_pose = face_mesh = None

    if YOLO_AVAILABLE:
        try:
            from ultralytics import YOLO
            yolo_det  = YOLO("yolov8n.pt")
            yolo_pose = YOLO("yolov8n-pose.pt")
        except Exception as e:
            print(f"[Flask Camera] YOLO load error: {e}")

    if MP_AVAILABLE:
        try:
            import mediapipe as mp
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            print(f"[Flask Camera] MediaPipe load error: {e}")

    # ── Simple Centroid Tracker (inline) ────────────────────────────────────
    class CentroidTracker:
        def __init__(self, max_disappeared=30):
            self.next_id     = 1
            self.objects     = OrderedDict()
            self.disappeared = OrderedDict()
            self.max_disappeared = max_disappeared

        def _centroid(self, box):
            return ((box[0]+box[2])//2, (box[1]+box[3])//2)

        def update(self, boxes):
            if not boxes:
                for oid in list(self.disappeared):
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.max_disappeared:
                        del self.objects[oid]
                        del self.disappeared[oid]
                return self.objects

            input_cents = [self._centroid(b) for b in boxes]

            if not self.objects:
                for c in input_cents:
                    self.objects[self.next_id]     = c
                    self.disappeared[self.next_id] = 0
                    self.next_id += 1
            else:
                obj_ids   = list(self.objects.keys())
                obj_cents = list(self.objects.values())
                D = np.zeros((len(obj_cents), len(input_cents)))
                for i, oc in enumerate(obj_cents):
                    for j, ic in enumerate(input_cents):
                        D[i, j] = math.sqrt((oc[0]-ic[0])**2 + (oc[1]-ic[1])**2)
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                used_r, used_c = set(), set()
                for r, c in zip(rows, cols):
                    if r in used_r or c in used_c or D[r,c] > 120:
                        continue
                    oid = obj_ids[r]
                    self.objects[oid]     = input_cents[c]
                    self.disappeared[oid] = 0
                    used_r.add(r); used_c.add(c)
                for r in set(range(D.shape[0])) - used_r:
                    oid = obj_ids[r]
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.max_disappeared:
                        del self.objects[oid]; del self.disappeared[oid]
                for c in set(range(D.shape[1])) - used_c:
                    self.objects[self.next_id]     = input_cents[c]
                    self.disappeared[self.next_id] = 0
                    self.next_id += 1
            return self.objects

    tracker        = CentroidTracker()
    ear_counters   = {}   # person_id → count
    pitch_counters = {}
    person_status  = {}   # person_id → "Active" | "Sleeping"
    last_alert     = {}
    prev_keypoints = None
    frame_count    = 0
    prev_count     = 0

    # ── Multi-signal playing detection constants ────────────────────────
    PLAY_ARM_SPREAD_RATIO  = 0.35
    PLAY_WRIST_RAISE_RATIO = 0.08
    PLAY_ELBOW_RAISE_RATIO = 0.05
    PLAY_WRIST_GAP_RATIO   = 0.40
    PLAY_SCORE_PLAY        = 2
    PLAY_SCORE_DANCE       = 3

    def playing_score(kps, fw, fh):
        """Return (score, signals) for one person's keypoint list."""
        sc, sigs = 0, []
        def kp(i):
            if i >= len(kps): return None
            x, y, c = kps[i]
            return (x, y, c) if c > 0.3 else None
        ls=kp(5); rs=kp(6); le=kp(7); re=kp(8)
        lw=kp(9); rw=kp(10); lh=kp(11); rh=kp(12)
        hip_cx = ((lh[0]+rh[0])/2 if lh and rh else
                  lh[0] if lh else rh[0] if rh else fw/2)
        # arm spread
        if lw and abs(lw[0]-hip_cx)/(fw+1e-6) > PLAY_ARM_SPREAD_RATIO:
            sc+=1; sigs.append("arm_L")
        if rw and abs(rw[0]-hip_cx)/(fw+1e-6) > PLAY_ARM_SPREAD_RATIO:
            sc+=1; sigs.append("arm_R")
        # wrist raise
        twr = PLAY_WRIST_RAISE_RATIO * fh
        if lw and ls and lw[1] < ls[1]-twr: sc+=1; sigs.append("wr_L")
        if rw and rs and rw[1] < rs[1]-twr: sc+=1; sigs.append("wr_R")
        # elbow raise
        ter = PLAY_ELBOW_RAISE_RATIO * fh
        if le and ls and le[1] < ls[1]-ter: sc+=1; sigs.append("er_L")
        if re and rs and re[1] < rs[1]-ter: sc+=1; sigs.append("er_R")
        # wide wrist gap
        if lw and rw and abs(lw[0]-rw[0])/(fw+1e-6) > PLAY_WRIST_GAP_RATIO:
            sc+=1; sigs.append("gap")
        return sc, sigs

    LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33,  160, 158, 133, 153, 144]

    def can_alert(atype):
        return (time.time() - last_alert.get(atype, 0)) > ALERT_COOLDOWN

    def trigger_alert(frame, atype):
        if not can_alert(atype):
            return
        last_alert[atype] = time.time()
        snap = save_snapshot(frame, dept_name, atype)
        insert_alert(dept_id, atype, snap, 0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        h, w = frame.shape[:2]
        person_boxes = []

        # ── 1. YOLOv8 Object Detection ───────────────────────────────────
        if yolo_det:
            try:
                results = yolo_det(frame, verbose=False, conf=0.4)
                for r in results:
                    for box in r.boxes:
                        cls_id   = int(box.cls[0])
                        cls_name = yolo_det.model.names[cls_id].lower()
                        conf     = float(box.conf[0])
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

                        # Fire
                        if "fire" in cls_name or "flame" in cls_name:
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
                            cv2.putText(frame, f"FIRE {conf:.0%}", (x1,y1-8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                            trigger_alert(frame, "fire")

                        # Food / eating
                        if cls_id in range(39, 60):
                            trigger_alert(frame, "eating")

                        # Mobile phone
                        if cls_id == MOBILE_CLASS_ID:
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,50,200), 2)
                            cv2.putText(frame, f"PHONE {conf:.0%}", (x1,y1-8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,50,200), 2)
                            trigger_alert(frame, "mobile_usage")

                        # Person box collection
                        if cls_name == "person":
                            person_boxes.append([x1, y1, x2, y2])

                # Fighting
                if len(person_boxes) >= 2:
                    for i in range(len(person_boxes)):
                        for j in range(i+1, len(person_boxes)):
                            if boxes_iou(person_boxes[i], person_boxes[j]) > FIGHT_IOU_THRESHOLD:
                                cv2.rectangle(frame,
                                              (person_boxes[i][0], person_boxes[i][1]),
                                              (person_boxes[i][2], person_boxes[i][3]),
                                              (0,0,255), 3)
                                cv2.putText(frame, "FIGHTING!", (10, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                                trigger_alert(frame, "fighting")
            except Exception as e:
                print(f"[YOLO] {e}")

        # ── 2. Centroid Tracking ─────────────────────────────────────────
        tracked      = tracker.update(person_boxes)
        person_count = len(tracked)

        if person_count != prev_count:
            print(f"[Flask Camera] {dept_name} — Person count: {person_count}")
            prev_count = person_count

        # Draw tracked persons
        for box, (pid, centroid) in zip(person_boxes, tracked.items()):
            x1, y1, x2, y2 = box
            status = person_status.get(pid, "Active")
            color  = (0, 0, 255) if status == "Sleeping" else (0, 220, 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Person {pid}: {status}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # ── 3. Sleeping Detection (MediaPipe FaceMesh) ───────────────────
        if face_mesh:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)

                if res.multi_face_landmarks:
                    for face_idx, face_lm in enumerate(res.multi_face_landmarks):
                        lm  = face_lm.landmark
                        pid = face_idx  # approximate match by face index

                        # EAR calculation
                        def lm_pt(i):
                            return np.array([lm[i].x, lm[i].y])

                        left_ear  = eye_aspect_ratio(lm, LEFT_EYE)
                        right_ear = eye_aspect_ratio(lm, RIGHT_EYE)
                        ear       = (left_ear + right_ear) / 2.0

                        ear_counters.setdefault(pid, 0)
                        if ear < EAR_THRESHOLD:
                            ear_counters[pid] += 1
                        else:
                            ear_counters[pid] = 0

                        if ear_counters[pid] >= EAR_CONSEC_FRAMES:
                            cv2.putText(frame, "SLEEPING (EYES CLOSED)",
                                        (30, 70 + face_idx * 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                            person_status[pid] = "Sleeping"
                            trigger_alert(frame, "sleeping")
                        else:
                            if person_status.get(pid) == "Sleeping":
                                person_status[pid] = "Active"

                        # Head pitch
                        nose = lm[1]
                        pitch_counters.setdefault(pid, 0)
                        if (nose.y - 0.5) > 0.12:
                            pitch_counters[pid] += 1
                        else:
                            pitch_counters[pid] = 0

                        if pitch_counters[pid] >= HEAD_PITCH_FRAMES:
                            cv2.putText(frame, "SLEEPING (HEAD DOWN)",
                                        (30, 100 + face_idx * 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,80,255), 2)
                            person_status[pid] = "Sleeping"
                            trigger_alert(frame, "sleeping")
                else:
                    ear_counters.clear()
                    pitch_counters.clear()
            except Exception as e:
                print(f"[MediaPipe] {e}")

        # ── 4. Pose Detection (every 5 frames) ───────────────────────────
        if yolo_pose and frame_count % 5 == 0:
            try:
                results = yolo_pose(frame, verbose=False, conf=0.4)
                for r in results:
                    if r.keypoints is None:
                        continue
                    kp_data  = r.keypoints.data
                    boxes_xy = r.boxes.xyxy if r.boxes else None

                    for idx, person_kps in enumerate(kp_data):
                        kps = person_kps.tolist()

                        # Playing / Dancing (multi-signal)
                        ps, _ = playing_score(kps, w, h)
                        vel   = keypoint_velocity(kps, prev_keypoints)
                        if vel > POSE_VEL_THRESHOLD:
                            ps += 1
                        if ps >= PLAY_SCORE_DANCE:
                            cv2.putText(frame, f"DANCING! ({ps})", (10, 140),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 2)
                            trigger_alert(frame, "dancing")
                        elif ps >= PLAY_SCORE_PLAY:
                            cv2.putText(frame, f"PLAYING! ({ps})", (10, 140),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                            trigger_alert(frame, "playing")
                        prev_keypoints = kps

                        # Hand raising (only when NOT playing/dancing)
                        try:
                            l_sh_y = kps[5][1]; r_sh_y = kps[6][1]
                            l_wr_y = kps[9][1]; r_wr_y = kps[10][1]
                            l_wr_c = kps[9][2]; r_wr_c = kps[10][2]

                            if ps < PLAY_SCORE_PLAY:
                                if l_wr_c > 0.4 and l_wr_y < l_sh_y - HAND_RAISE_MARGIN:
                                    lx = int(kps[9][0]); ly = int(kps[9][1])
                                    cv2.putText(frame, "HAND RAISED", (lx, ly-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,180), 2)
                                    trigger_alert(frame, "hand_raising")

                                if r_wr_c > 0.4 and r_wr_y < r_sh_y - HAND_RAISE_MARGIN:
                                    rx = int(kps[10][0]); ry = int(kps[10][1])
                                    cv2.putText(frame, "HAND RAISED", (rx, ry-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,180), 2)
                                    trigger_alert(frame, "hand_raising")
                        except (IndexError, TypeError):
                            pass

                        # Standing / Sitting
                        if boxes_xy is not None and idx < len(boxes_xy):
                            bx1, by1, bx2, by2 = [int(v) for v in boxes_xy[idx]]
                            bh = by2 - by1; bw = bx2 - bx1
                            if bw > 0:
                                ratio = bh / (bw + 1e-6)
                                posture = "Standing" if ratio > 2.0 else ("Sitting" if ratio > 1.0 else "Active")
                                cv2.putText(frame, posture, (bx1, by2 + 18),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,220,0), 2)
            except Exception as e:
                print(f"[Pose] {e}")

        # ── 5. HUD Overlay ───────────────────────────────────────────────
        sleeping_count = sum(1 for s in person_status.values() if s == "Sleeping")
        active_count   = max(0, person_count - sleeping_count)
        ts = datetime.datetime.now().strftime("%H:%M:%S")

        cv2.rectangle(frame, (0, 0), (w, 40), (15, 15, 15), -1)
        hud = (f"Dept: {dept_name}  |  Students: {person_count}  |  "
               f"Active: {active_count}  |  Sleeping: {sleeping_count}  |  {ts}")
        cv2.putText(frame, hud, (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 120), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # ── Extensions ─────────────────────────────────────────────────────────────
    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)

    login_manager.login_view    = "auth.login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "warning"

    # ── Blueprints ─────────────────────────────────────────────────────────────
    from blueprints.auth      import auth_bp
    from blueprints.student   import student_bp
    from blueprints.staff     import staff_bp
    from blueprints.hod       import hod_bp
    from blueprints.principal import principal_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(student_bp,   url_prefix="/student")
    app.register_blueprint(staff_bp,     url_prefix="/staff")
    app.register_blueprint(hod_bp,       url_prefix="/hod")
    app.register_blueprint(principal_bp, url_prefix="/principal")

    # ── Snapshot dir ───────────────────────────────────────────────────────────
    os.makedirs(app.config["SNAPSHOT_DIR"], exist_ok=True)

    # ── Root redirect ──────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return redirect(url_for("auth.login"))

    @app.route("/video_feed/<string:dept_name>")
    @login_required
    def video_feed(dept_name):
        return Response(gen_frames(dept_name),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    return app


@login_manager.user_loader
def load_user(user_id: str):
    """Load any role user from composite ID like 'student_5' or 'staff_12'."""
    if not user_id or "_" not in user_id:
        return None
    role, uid = user_id.split("_", 1)
    try:
        uid = int(uid)
    except ValueError:
        return None

    if role == "student":
        return Student.query.get(uid)
    elif role == "staff":
        return Staff.query.get(uid)
    elif role == "hod":
        return HOD.query.get(uid)
    elif role == "principal":
        return Principal.query.get(uid)
    return None


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        db.create_all()
    app.run(debug=True, host="0.0.0.0", port=5000)
