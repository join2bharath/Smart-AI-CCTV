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
    import cv2
    import time
    import mathP
    import numpy as np
    import datetime
    from camera_service import (
        YOLO_AVAILABLE, MP_AVAILABLE, save_snapshot, insert_alert, get_dept_info
    )

    # Get department info
    dept_info = get_dept_info(dept_name)
    dept_id = dept_info["dept_id"] if dept_info else 1
    
    # Initialize camera 0 (the laptop's webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"[Flask Camera] Could not open laptop camera (index 0)")
        return
        
    # Load YOLO detection models locally if available
    yolo_det = None
    yolo_pose = None
    if YOLO_AVAILABLE:
        try:
            from ultralytics import YOLO
            yolo_det = YOLO("yolov8n.pt")
            yolo_pose = YOLO("yolov8n-pose.pt")
        except Exception as e:
            print(f"[Flask Camera] Error loading YOLO: {e}")
            
    # Load MediaPipe Face Mesh if available
    face_mesh = None
    if MP_AVAILABLE:
        try:
            import mediapipe as mp
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=3,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            print(f"[Flask Camera] Error loading MediaPipe: {e}")

    # Local state for EAR / velocities / cooldowns
    last_alert = {}
    ear_counter = 0
    pitch_counter = 0
    prev_keypoints = None
    frame_count = 0
    
    # Constants
    EAR_THRESHOLD = 0.25
    EAR_CONSEC_FRAMES = 90
    HEAD_PITCH_THRESH = -25.0
    HEAD_PITCH_FRAMES = 60
    FIGHT_IOU_THRESHOLD = 0.15
    POSE_VEL_THRESHOLD = 40.0
    ALERT_COOLDOWN = 30
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        h, w = frame.shape[:2]
        
        # 1. YOLOv8 Detection
        if yolo_det:
            try:
                results = yolo_det(frame, verbose=False, conf=0.4)
                boxes = []
                labels = []
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = yolo_det.model.names[cls_id].lower()
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{cls_name} {conf:.0%}", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        if "fire" in cls_name or "flame" in cls_name:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, "⚠️ FIRE!", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            if time.time() - last_alert.get("fire", 0) > ALERT_COOLDOWN:
                                last_alert["fire"] = time.time()
                                snap_path = save_snapshot(frame, dept_name, "fire")
                                insert_alert(dept_id, "fire", snap_path, 0)
                            
                        if cls_id in range(39, 60):
                            if time.time() - last_alert.get("eating", 0) > ALERT_COOLDOWN:
                                last_alert["eating"] = time.time()
                                snap_path = save_snapshot(frame, dept_name, "eating")
                                insert_alert(dept_id, "eating", snap_path, 0)
                            
                        if cls_name == "person":
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls_name)
                            
                persons = [b for b, l in zip(boxes, labels) if l == "person"]
                if len(persons) >= 2:
                    for i in range(len(persons)):
                        for j in range(i+1, len(persons)):
                            ix1 = max(persons[i][0], persons[j][0])
                            iy1 = max(persons[i][1], persons[j][1])
                            ix2 = min(persons[i][2], persons[j][2])
                            iy2 = min(persons[i][3], persons[j][3])
                            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                            area1 = (persons[i][2] - persons[i][0]) * (persons[i][3] - persons[i][1])
                            area2 = (persons[j][2] - persons[j][0]) * (persons[j][3] - persons[j][1])
                            union = area1 + area2 - inter
                            iou = inter / (union + 1e-6)
                            
                            if iou > FIGHT_IOU_THRESHOLD:
                                cv2.rectangle(frame, (persons[i][0], persons[i][1]), (persons[i][2], persons[i][3]), (0, 0, 255), 3)
                                cv2.rectangle(frame, (persons[j][0], persons[j][1]), (persons[j][2], persons[j][3]), (0, 0, 255), 3)
                                cv2.putText(frame, "⚠️ FIGHTING!", (10, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                if time.time() - last_alert.get("fighting", 0) > ALERT_COOLDOWN:
                                    last_alert["fighting"] = time.time()
                                    snap_path = save_snapshot(frame, dept_name, "fighting")
                                    insert_alert(dept_id, "fighting", snap_path, 0)
            except Exception as e:
                pass
                
        if yolo_pose and frame_count % 5 == 0:
            try:
                results = yolo_pose(frame, verbose=False, conf=0.4)
                for r in results:
                    if r.keypoints is not None:
                        kp_data = r.keypoints.data
                        for person_kps in kp_data:
                            kps = person_kps.tolist()
                            if prev_keypoints is not None:
                                vels = []
                                for (xc, yc, cc), (xp, yp, cp) in zip(kps, prev_keypoints):
                                    if cc > 0.4 and cp > 0.4:
                                        vels.append(math.sqrt((xc-xp)**2 + (yc-yp)**2))
                                vel = float(np.mean(vels)) if vels else 0.0
                                
                                if vel > POSE_VEL_THRESHOLD * 1.5:
                                    cv2.putText(frame, "💃 DANCING!", (10, 120),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
                                    if time.time() - last_alert.get("dancing", 0) > ALERT_COOLDOWN:
                                        last_alert["dancing"] = time.time()
                                        snap_path = save_snapshot(frame, dept_name, "dancing")
                                        insert_alert(dept_id, "dancing", snap_path, 0)
                                elif vel > POSE_VEL_THRESHOLD:
                                    cv2.putText(frame, "🎮 PLAYING!", (10, 120),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                                    if time.time() - last_alert.get("playing", 0) > ALERT_COOLDOWN:
                                        last_alert["playing"] = time.time()
                                        snap_path = save_snapshot(frame, dept_name, "playing")
                                        insert_alert(dept_id, "playing", snap_path, 0)
                            prev_keypoints = kps
            except Exception as e:
                pass
                
        if face_mesh:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    for face_lm in res.multi_face_landmarks:
                        lm = face_lm.landmark
                        def lm_pt(idx):
                            p = lm[idx]
                            return np.array([p.x, p.y])
                        
                        v1 = np.linalg.norm(lm_pt(385) - lm_pt(380))
                        v2 = np.linalg.norm(lm_pt(387) - lm_pt(373))
                        h_dist = np.linalg.norm(lm_pt(362) - lm_pt(263))
                        left_ear = (v1 + v2) / (2.0 * h_dist + 1e-6)
                        
                        v1_r = np.linalg.norm(lm_pt(160) - lm_pt(144))
                        v2_r = np.linalg.norm(lm_pt(158) - lm_pt(153))
                        h_dist_r = np.linalg.norm(lm_pt(33) - lm_pt(133))
                        right_ear = (v1_r + v2_r) / (2.0 * h_dist_r + 1e-6)
                        
                        ear = (left_ear + right_ear) / 2.0
                        
                        if ear < EAR_THRESHOLD:
                            ear_counter += 1
                        else:
                            ear_counter = 0
                            
                        if ear_counter >= EAR_CONSEC_FRAMES:
                            cv2.putText(frame, "🚨 SLEEPING ALERT!", (10, 160),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                            if time.time() - last_alert.get("sleeping", 0) > ALERT_COOLDOWN:
                                last_alert["sleeping"] = time.time()
                                snap_path = save_snapshot(frame, dept_name, "sleeping")
                                insert_alert(dept_id, "sleeping", snap_path, 0)
                else:
                    ear_counter = 0
            except Exception as e:
                pass

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"[LIVE webcam] Dept: {dept_name} | {ts}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
                    
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

    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "warning"

    # ── Blueprints ─────────────────────────────────────────────────────────────
    from blueprints.auth import auth_bp
    from blueprints.student import student_bp
    from blueprints.staff import staff_bp
    from blueprints.hod import hod_bp
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
