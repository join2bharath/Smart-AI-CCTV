"""
camera_service.py — AI Camera Service for College Management System
==================================================================
Runs as a SEPARATE Python process from Flask.
Usage: python camera_service.py [--dept IT] [--cam 0]

Detection capabilities:
  - YOLOv8 (object detection): fire, person count, eating, fighting
  - MediaPipe Face Mesh: sleeping (EAR < 0.25 or head pitch < -25°)
  - YOLOv8-Pose: playing / dancing (keypoint velocity patterns)

On detection:
  - Saves snapshot to /snapshots/<dept>/
  - Inserts row into camera_alerts MySQL table
  - Sends email alert to HOD of that department

NOTE: Camera access is scoped by dept_id. Each department maps to a
      camera index configured in camera_config.json.
"""

import os
import sys
import time
import json
import math
import argparse
import datetime
import logging
import smtplib
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import deque

import cv2
import numpy as np
import pymysql

# ── Optional imports (graceful fallback if model not yet downloaded) ─────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[!] ultralytics not installed. Run: pip install ultralytics")

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[!] mediapipe not installed. Run: pip install mediapipe")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("camera_service")

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG_FILE = Path(__file__).parent / "camera_config.json"
SNAPSHOT_DIR = Path(__file__).parent / "static" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

DB_CFG = dict(
    host="localhost", port=3306,
    user="root", password="root",
    database="college_db", charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)

# SMTP — configure these or set via environment variables
SMTP_HOST     = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("MAIL_PORT", 587))
SMTP_USER     = os.environ.get("MAIL_USERNAME", "college.cms.alerts@gmail.com")
SMTP_PASSWORD = os.environ.get("MAIL_PASSWORD", "")
SMTP_ENABLED  = bool(SMTP_PASSWORD)   # email only if password is set

# Detection thresholds
EAR_THRESHOLD       = 0.25     # Eye Aspect Ratio for sleep detection
EAR_CONSEC_FRAMES   = 300      # ~10s at 30fps
HEAD_PITCH_THRESH   = -25.0    # degrees
HEAD_PITCH_FRAMES   = 150      # ~5s at 30fps
FIGHT_IOU_THRESHOLD = 0.15     # bounding-box overlap
POSE_VEL_THRESHOLD  = 40.0     # pixels/frame for playing/dancing
ALERT_COOLDOWN      = 60       # seconds between same alert type

# Alert icon mapping for log
ALERT_ICONS = {
    "fire": "🔥", "fighting": "👊", "sleeping": "😴",
    "eating": "🍔", "playing": "🎮", "dancing": "💃", "person": "👤",
}

# ── Default camera config ─────────────────────────────────────────────────────
DEFAULT_CAMERA_CONFIG = {
    "IT":    {"camera_index": 0, "enabled": True},
    "CSE":   {"camera_index": 0, "enabled": True},
    "AIDS":  {"camera_index": 0, "enabled": True},
    "ECE":   {"camera_index": 0, "enabled": True},
    "EEE":   {"camera_index": 0, "enabled": True},
    "CIVIL": {"camera_index": 0, "enabled": True},
}


def load_config() -> dict:
    """Load camera configuration from JSON file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    # Write defaults if not exist
    with open(CONFIG_FILE, "w") as f:
        json.dump(DEFAULT_CAMERA_CONFIG, f, indent=2)
    log.info(f"Created default camera_config.json at {CONFIG_FILE}")
    return DEFAULT_CAMERA_CONFIG


# ── Database helpers ──────────────────────────────────────────────────────────
def get_db_connection():
    """Return a fresh pymysql connection."""
    return pymysql.connect(**DB_CFG)


def get_dept_info(dept_name: str) -> dict | None:
    """Fetch dept_id and HOD email for the given department name."""
    try:
        con = get_db_connection()
        with con.cursor() as cur:
            cur.execute("SELECT id FROM departments WHERE name=%s", (dept_name,))
            dept = cur.fetchone()
            if not dept:
                return None
            cur.execute(
                "SELECT email FROM hods WHERE dept_id=%s LIMIT 1",
                (dept["id"],)
            )
            hod = cur.fetchone()
        con.close()
        return {"dept_id": dept["id"], "hod_email": hod["email"] if hod else None}
    except Exception as e:
        log.error(f"DB error (get_dept_info): {e}")
        return None


def insert_alert(dept_id: int, alert_type: str, snapshot_path: str, camera_index: int):
    """Insert a camera alert into the database."""
    try:
        con = get_db_connection()
        with con.cursor() as cur:
            cur.execute("""
                INSERT INTO camera_alerts
                    (dept_id, alert_type, snapshot_path, camera_index, acknowledged)
                VALUES (%s, %s, %s, %s, 0)
            """, (dept_id, alert_type, snapshot_path, camera_index))
        con.close()
        log.info(f"[DB] Alert inserted: {alert_type} | dept_id={dept_id}")
    except Exception as e:
        log.error(f"DB error (insert_alert): {e}")


# ── Email helper ──────────────────────────────────────────────────────────────
def send_email_alert(hod_email: str, dept_name: str, alert_type: str, snapshot_path: str):
    """Send email notification to HOD."""
    if not SMTP_ENABLED:
        log.info(f"[EMAIL] (SMTP disabled) Would send '{alert_type}' alert to {hod_email}")
        return
    try:
        subject = f"🚨 Camera Alert: {alert_type.title()} detected in {dept_name}"
        body = f"""
College CMS — Automated Camera Alert
=====================================
Department : {dept_name}
Alert Type : {alert_type.upper()}
Timestamp  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Snapshot   : {snapshot_path}

Please log in to the HOD dashboard to acknowledge this alert.
        """
        msg = MIMEMultipart()
        msg["From"]    = SMTP_USER
        msg["To"]      = hod_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        log.info(f"[EMAIL] Alert sent to {hod_email}")
    except Exception as e:
        log.error(f"[EMAIL] Failed: {e}")


# ── Snapshot helper ────────────────────────────────────────────────────────────
def save_snapshot(frame: np.ndarray, dept_name: str, alert_type: str) -> str:
    """Save a snapshot image; return relative path for DB storage."""
    dept_dir = SNAPSHOT_DIR / dept_name
    dept_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{alert_type}_{ts}.jpg"
    full_path = dept_dir / filename
    cv2.imwrite(str(full_path), frame)
    # Return path relative to static/ for Flask to serve
    rel_path = f"snapshots/{dept_name}/{filename}"
    return rel_path


# ── Geometry helpers ──────────────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices) -> float:
    """Compute EAR (Eye Aspect Ratio) from MediaPipe face landmarks."""
    def lm(i):
        p = landmarks[i]
        return np.array([p.x, p.y])
    # Vertical distances
    v1 = np.linalg.norm(lm(eye_indices[1]) - lm(eye_indices[5]))
    v2 = np.linalg.norm(lm(eye_indices[2]) - lm(eye_indices[4]))
    # Horizontal distance
    h  = np.linalg.norm(lm(eye_indices[0]) - lm(eye_indices[3]))
    return (v1 + v2) / (2.0 * h + 1e-6)


def boxes_iou(b1, b2) -> float:
    """Compute IoU of two [x1,y1,x2,y2] boxes."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def keypoint_velocity(kp_curr, kp_prev) -> float:
    """Mean velocity of all detected keypoints between frames."""
    if kp_prev is None:
        return 0.0
    vels = []
    for (xc, yc, cc), (xp, yp, cp) in zip(kp_curr, kp_prev):
        if cc > 0.4 and cp > 0.4:
            vels.append(math.sqrt((xc-xp)**2 + (yc-yp)**2))
    return float(np.mean(vels)) if vels else 0.0


# ── Main Camera Detector class ────────────────────────────────────────────────
class DeptCameraDetector:
    """Runs detection on a single department's camera stream."""

    # MediaPipe left/right eye landmark indices (subset for EAR)
    LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33,  160, 158, 133, 153, 144]

    def __init__(self, dept_name: str, camera_index: int):
        self.dept_name    = dept_name
        self.cam_idx      = camera_index
        self.dept_info    = get_dept_info(dept_name)
        self.last_alert   = {}          # alert_type → timestamp
        self.ear_counter  = 0
        self.pitch_counter = 0
        self.prev_keypoints = None

        # Load models
        self.yolo_det  = YOLO("yolov8n.pt")     if YOLO_AVAILABLE else None
        self.yolo_pose = YOLO("yolov8n-pose.pt") if YOLO_AVAILABLE else None

        if MP_AVAILABLE:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.face_mesh = None

    def _can_alert(self, alert_type: str) -> bool:
        """Rate-limit alerts per type."""
        last = self.last_alert.get(alert_type, 0)
        return (time.time() - last) > ALERT_COOLDOWN

    def _trigger_alert(self, frame: np.ndarray, alert_type: str):
        """Save snapshot, insert DB row, send email."""
        if not self._can_alert(alert_type):
            return
        self.last_alert[alert_type] = time.time()

        icon = ALERT_ICONS.get(alert_type, "⚠️")
        log.info(f"{icon} ALERT [{self.dept_name}]: {alert_type.upper()}")

        snap_path = save_snapshot(frame, self.dept_name, alert_type)

        if self.dept_info:
            insert_alert(
                self.dept_info["dept_id"],
                alert_type,
                snap_path,
                self.cam_idx,
            )
            if self.dept_info.get("hod_email"):
                send_email_alert(
                    self.dept_info["hod_email"],
                    self.dept_name,
                    alert_type,
                    snap_path,
                )

    # ── Detection methods ─────────────────────────────────────────────────────

    def detect_fire_eating_fighting(self, frame: np.ndarray):
        """YOLOv8 object detection for fire, eating, fighting."""
        if not self.yolo_det:
            return

        results = self.yolo_det(frame, verbose=False, conf=0.4)
        boxes   = []
        labels  = []

        for r in results:
            for box in r.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.yolo_det.model.names[cls_id].lower()
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

                # ── Fire detection ───────────────────────────────────────────
                if "fire" in cls_name or "flame" in cls_name:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(frame, f"FIRE {conf:.0%}", (x1,y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    self._trigger_alert(frame, "fire")

                # ── Eating detection ─────────────────────────────────────────
                # YOLO COCO: cup(41), fork(42), knife(43), spoon(44), bowl(45), apple..banana (47-52)
                if cls_id in range(39, 60):   # food/kitchen objects
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,100), 2)
                    self._trigger_alert(frame, "eating")

                if cls_name == "person":
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_name)

        # ── Fighting: overlapping person bboxes ──────────────────────────────
        persons = [b for b, l in zip(boxes, labels) if l == "person"]
        if len(persons) >= 2:
            for i in range(len(persons)):
                for j in range(i+1, len(persons)):
                    iou = boxes_iou(persons[i], persons[j])
                    if iou > FIGHT_IOU_THRESHOLD:
                        cv2.rectangle(frame, (persons[i][0], persons[i][1]),
                                      (persons[i][2], persons[i][3]), (0,50,255), 3)
                        self._trigger_alert(frame, "fighting")

    def detect_sleeping(self, frame: np.ndarray):
        """MediaPipe FaceMesh: sleep via EAR and head pitch."""
        if not self.face_mesh:
            return

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = self.face_mesh.process(rgb)
        h, w  = frame.shape[:2]

        if not res.multi_face_landmarks:
            self.ear_counter   = 0
            self.pitch_counter = 0
            return

        for face_lm in res.multi_face_landmarks:
            lm = face_lm.landmark

            # EAR
            left_ear  = eye_aspect_ratio(lm, self.LEFT_EYE)
            right_ear = eye_aspect_ratio(lm, self.RIGHT_EYE)
            ear       = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                self.ear_counter += 1
            else:
                self.ear_counter = 0

            if self.ear_counter >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "SLEEPING (EAR)", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                self._trigger_alert(frame, "sleeping")

            # Head pitch using nose tip vs forehead/chin landmarks
            nose  = lm[1]
            chin  = lm[152]
            fhead = lm[10]
            # Compute rough pitch angle
            dy = (chin.y - fhead.y) * h
            dz = abs(nose.z - chin.z) * h
            pitch = math.degrees(math.atan2(dz, max(dy, 1)))
            # Negative pitch → head drooping forward
            if (nose.y - 0.5) > 0.1:   # head tilted down heuristic
                self.pitch_counter += 1
            else:
                self.pitch_counter = 0

            if self.pitch_counter >= HEAD_PITCH_FRAMES:
                cv2.putText(frame, "SLEEPING (HEAD)", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,255), 2)
                self._trigger_alert(frame, "sleeping")

    def detect_playing_dancing(self, frame: np.ndarray):
        """YOLOv8-Pose: detect rapid keypoint movement (playing/dancing)."""
        if not self.yolo_pose:
            return

        results = self.yolo_pose(frame, verbose=False, conf=0.4)

        for r in results:
            if r.keypoints is None:
                continue
            kp_data = r.keypoints.data  # (N_persons, 17, 3)
            for person_kps in kp_data:
                kps = person_kps.tolist()  # list of (x, y, conf)
                vel = keypoint_velocity(kps, self.prev_keypoints)

                if vel > POSE_VEL_THRESHOLD * 1.5:
                    self._trigger_alert(frame, "dancing")
                elif vel > POSE_VEL_THRESHOLD:
                    self._trigger_alert(frame, "playing")

                self.prev_keypoints = kps

    def run(self):
        """Main detection loop for this department's camera."""
        log.info(f"[{self.dept_name}] Starting camera on index {self.cam_idx} …")
        cap = cv2.VideoCapture(self.cam_idx)

        if not cap.isOpened():
            log.error(f"[{self.dept_name}] Could not open camera index {self.cam_idx}")
            return

        log.info(f"[{self.dept_name}] Camera opened. Press Q in window to quit.")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning(f"[{self.dept_name}] Frame read failed — retrying …")
                time.sleep(0.5)
                continue

            frame_count += 1

            # Run all detectors every frame
            self.detect_fire_eating_fighting(frame)
            self.detect_sleeping(frame)

            # Pose detection every 5 frames (performance)
            if frame_count % 5 == 0:
                self.detect_playing_dancing(frame)

            # Overlay info
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"[{self.dept_name}] {ts}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,220,100), 2)

            cv2.imshow(f"College CMS — {self.dept_name}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        log.info(f"[{self.dept_name}] Camera service stopped.")


# ── Test mode: fire detection on a sample image ───────────────────────────────
def test_fire_detection():
    """Test fire detection on a sample image (creates a test image if absent)."""
    log.info("=== Fire Detection Test ===")
    if not YOLO_AVAILABLE:
        log.error("ultralytics not installed. Cannot test.")
        return

    model = YOLO("yolov8n.pt")

    # Create a simple red-orange test image (simulates warm colors)
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw orange/red rectangle simulating fire region
    cv2.rectangle(test_img, (150, 100), (450, 380), (0, 100, 255), -1)
    cv2.putText(test_img, "FIRE TEST IMAGE", (80, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

    test_path = SNAPSHOT_DIR / "test_fire_input.jpg"
    cv2.imwrite(str(test_path), test_img)
    log.info(f"Test image saved to: {test_path}")

    # Run YOLO detection
    results = model(test_img, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_name = model.model.names[int(box.cls[0])]
            conf     = float(box.conf[0])
            detections.append(f"{cls_name} ({conf:.0%})")

    if detections:
        log.info(f"Detections: {', '.join(detections)}")
    else:
        log.info("No objects detected in test image (expected for synthetic image).")

    log.info("Fire detection test complete. YOLOv8 model loaded successfully ✓")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="College CMS — AI Camera Service")
    parser.add_argument("--dept", default="IT", help="Department name (IT/CSE/AIDS/ECE/EEE/CIVIL)")
    parser.add_argument("--cam",  type=int, default=-1, help="Camera index override (-1 = use config)")
    parser.add_argument("--test", action="store_true", help="Run fire detection test and exit")
    args = parser.parse_args()

    if args.test:
        test_fire_detection()
        return

    config = load_config()
    dept_name = args.dept.upper()

    if dept_name not in config:
        log.error(f"Department '{dept_name}' not in camera_config.json. Options: {list(config.keys())}")
        sys.exit(1)

    dept_cfg = config[dept_name]
    if not dept_cfg.get("enabled", True):
        log.info(f"[{dept_name}] Camera disabled in config. Exiting.")
        sys.exit(0)

    cam_idx = args.cam if args.cam >= 0 else dept_cfg.get("camera_index", 0)

    log.info(f"College CMS — AI Camera Service")
    log.info(f"  Department  : {dept_name}")
    log.info(f"  Camera Index: {cam_idx}")
    log.info(f"  YOLOv8      : {'✓' if YOLO_AVAILABLE else '✗'}")
    log.info(f"  MediaPipe   : {'✓' if MP_AVAILABLE else '✗'}")
    log.info(f"  SMTP Email  : {'✓ Enabled' if SMTP_ENABLED else '✗ Disabled (set MAIL_PASSWORD env var)'}")

    detector = DeptCameraDetector(dept_name, cam_idx)
    detector.run()


if __name__ == "__main__":
    main()
