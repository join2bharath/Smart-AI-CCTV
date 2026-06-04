"""
camera_service.py — AI Camera Service for College Management System
==================================================================
Runs as a SEPARATE Python process from Flask.
Usage: python camera_service.py [--dept IT] [--cam 0]

Detection capabilities:
  - YOLOv8 (object detection): fire, person count, eating, fighting, mobile usage
  - MediaPipe Face Mesh: sleeping (EAR < 0.25 or head pitch < -25°)
  - YOLOv8-Pose: playing / dancing (keypoint velocity), hand raising, standing/sitting

Person Tracking:
  - Simple centroid tracker assigns Person 1, Person 2, ... IDs
  - Count displayed live on camera feed

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
from email.mime.image import MIMEImage
import threading
from collections import deque, OrderedDict

try:
    import google.generativeai as genai
    from PIL import Image
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False
    print("[!] google-generativeai or PIL not installed.")

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
CONFIG_FILE   = Path(__file__).parent / "camera_config.json"
SNAPSHOT_DIR  = Path(__file__).parent / "static" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

DB_CFG = dict(
    host="localhost", port=3306,
    user="root", password="root",
    database="college_db", charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)

# SMTP — configure these or set via environment variables
SMTP_HOST     = os.environ.get("MAIL_SERVER",   "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("MAIL_PORT", 587))
SMTP_USER     = os.environ.get("MAIL_USERNAME", "college.cms.alerts@gmail.com")
SMTP_PASSWORD = os.environ.get("MAIL_PASSWORD", "")
SMTP_ENABLED  = bool(SMTP_PASSWORD)

# ── Detection thresholds ──────────────────────────────────────────────────────
EAR_THRESHOLD       = 0.25      # Eye Aspect Ratio for sleep detection
EAR_CONSEC_FRAMES   = 90        # ~3s at 30fps
HEAD_PITCH_FRAMES   = 60        # ~2s at 30fps
FIGHT_IOU_THRESHOLD = 0.45      # strict bounding-box overlap for pure-BBox fighting
POSE_VEL_THRESHOLD  = 40.0      # pixels/frame for playing/dancing
ALERT_COOLDOWN      = 60        # seconds between same alert type
MOBILE_CLASS_ID     = 67        # YOLO COCO 'cell phone'
HAND_RAISE_MARGIN   = 20        # wrist y < shoulder y - margin (pixels)

# ── Eating detection ─────────────────────────────────────────────────────────
# COCO class IDs for food items and eating utensils
FOOD_CLASS_IDS = {
    39,   # bottle
    40,   # wine glass
    41,   # cup
    42,   # fork
    43,   # knife
    44,   # spoon
    45,   # bowl
    46,   # banana
    47,   # apple
    48,   # sandwich
    49,   # orange
    50,   # broccoli
    51,   # carrot
    52,   # hot dog
    53,   # pizza
    54,   # donut
    55,   # cake
    60,   # dining table  (strong eating context)
}
# Subset: actual food objects (higher confidence than just utensils)
PURE_FOOD_IDS = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55}
DRINKING_IDS  = {39, 40, 41}    # bottle / glass / cup
UTENSIL_IDS   = {42, 43, 44}    # fork / knife / spoon

# Wrist-to-nose distance threshold for hand-to-mouth eating gesture
# expressed as a fraction of the person's bounding-box height
EATING_WRIST_NOSE_RATIO = 0.35
# Minimum consecutive pose frames before triggering eating alert
EATING_POSE_FRAMES      = 8

# ── Alert icon mapping for log ────────────────────────────────────────────────
ALERT_ICONS = {
    "fire":         "🔥",
    "fighting":     "👊",
    "sleeping":     "😴",
    "eating":       "🍔",
    "playing":      "🎮",
    "dancing":      "💃",
    "mobile_usage": "📱",
    "hand_raising": "✋",
    "standing":     "🧍",
    "sitting":      "🪑",
    "person":       "👤",
}

# ── Gemini Configuration ──────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_ENABLED = bool(GEMINI_API_KEY) and GEMINI_SDK_AVAILABLE

if GEMINI_ENABLED:
    genai.configure(api_key=GEMINI_API_KEY)
    log.info("[GEMINI] Gemini API configured successfully.")

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


def send_email_alert(hod_email: str | None, dept_name: str, alert_type: str, snapshot_path: str):
    """Send email notification with photo attachment."""
    if not SMTP_ENABLED:
        log.info(f"[EMAIL] (SMTP disabled) Would send '{alert_type}' alert to {hod_email}")
        return
        
    recipients = []
    if hod_email:
        recipients.append(hod_email)
        
    # Send fighting, sleeping, and eating alerts to the requested mail id
    if alert_type in ["fighting", "sleeping", "eating"]:
        recipients.append("join2bharath2003@gmail.com")
        
    recipients = list(set(recipients))
    if not recipients:
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

Please log in to the HOD/Principal dashboard to acknowledge this alert.
        """
        msg = MIMEMultipart()
        msg["From"]    = SMTP_USER
        msg["To"]      = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        
        # Attach the photo
        full_snap_path = Path(__file__).parent / "static" / snapshot_path
        if full_snap_path.exists():
            with open(full_snap_path, "rb") as f:
                img_data = f.read()
            image = MIMEImage(img_data, name=os.path.basename(snapshot_path))
            image.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(snapshot_path)}"')
            msg.attach(image)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        log.info(f"[EMAIL] Alert sent to {recipients}")
    except Exception as e:
        log.error(f"[EMAIL] Failed: {e}")


# ── Snapshot helper ────────────────────────────────────────────────────────────
def save_snapshot(frame: np.ndarray, dept_name: str, alert_type: str) -> str:
    """Save a snapshot image; return relative path for DB storage."""
    dept_dir = SNAPSHOT_DIR / dept_name
    dept_dir.mkdir(exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{alert_type}_{ts}.jpg"
    full_path = dept_dir / filename
    cv2.imwrite(str(full_path), frame)
    return f"snapshots/{dept_name}/{filename}"


# ── Geometry helpers ──────────────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices) -> float:
    """Compute EAR (Eye Aspect Ratio) from MediaPipe face landmarks."""
    def lm(i):
        p = landmarks[i]
        return np.array([p.x, p.y])
    v1 = np.linalg.norm(lm(eye_indices[1]) - lm(eye_indices[5]))
    v2 = np.linalg.norm(lm(eye_indices[2]) - lm(eye_indices[4]))
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


# ── Simple Centroid Tracker ───────────────────────────────────────────────────
class CentroidTracker:
    """
    Assigns persistent Person IDs to detected bounding boxes
    using centroid distance matching across frames.
    """
    def __init__(self, max_disappeared=30):
        self.next_id      = 1
        self.objects      = OrderedDict()   # id → centroid
        self.disappeared  = OrderedDict()   # id → frames missing
        self.max_disappeared = max_disappeared

    def _centroid(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, boxes):
        """
        boxes: list of [x1,y1,x2,y2]
        Returns OrderedDict: id → centroid
        """
        if not boxes:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]
            return self.objects

        input_centroids = [self._centroid(b) for b in boxes]

        if not self.objects:
            for c in input_centroids:
                self.objects[self.next_id]     = c
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        else:
            obj_ids    = list(self.objects.keys())
            obj_cents  = list(self.objects.values())

            # Compute distance matrix
            D = np.zeros((len(obj_cents), len(input_centroids)), dtype="float")
            for i, oc in enumerate(obj_cents):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = math.sqrt((oc[0]-ic[0])**2 + (oc[1]-ic[1])**2)

            # Greedy match: row=smallest dist first
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                if D[r, c] > 120:   # too far → new person
                    continue
                oid = obj_ids[r]
                self.objects[oid]     = input_centroids[c]
                self.disappeared[oid] = 0
                used_rows.add(r)
                used_cols.add(c)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for r in unused_rows:
                oid = obj_ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]

            for c in unused_cols:
                self.objects[self.next_id]     = input_centroids[c]
                self.disappeared[self.next_id] = 0
                self.next_id += 1

        return self.objects


# ── Main Camera Detector class ────────────────────────────────────────────────
class DeptCameraDetector:
    """Runs detection on a single department's camera stream."""

    LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33,  160, 158, 133, 153, 144]

    def __init__(self, dept_name: str, camera_index: int):
        self.dept_name      = dept_name
        self.cam_idx        = camera_index
        self.dept_info      = get_dept_info(dept_name)
        self.last_alert     = {}      # alert_type → timestamp
        self.tracker        = CentroidTracker(max_disappeared=30)
        self.person_status  = {}      # person_id → activity string
        self.ear_counters   = {}      # person_id → ear frame count
        self.pitch_counters = {}      # person_id → pitch frame count
        self.eating_pose_counters = {} # person_idx → hand-to-mouth frame count
        self.sleeping_pose_counters = {} # person_idx → sleeping posture frame count
        self.fight_pose_counters = {}    # pair_id → aggressive posture frame count
        self._eating_food_in_frame = False   # set True when food object detected
        self.prev_keypoints = None
        self.prev_count     = 0
        self.running        = False
        self.latest_frame   = None

        log.info(f"Loading YOLO models …")
        self.yolo_det  = YOLO("yolov8n.pt")      if YOLO_AVAILABLE else None
        self.yolo_pose = YOLO("yolov8n-pose.pt")  if YOLO_AVAILABLE else None

        if MP_AVAILABLE:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.face_mesh = None

    def _can_alert(self, alert_type: str) -> bool:
        last = self.last_alert.get(alert_type, 0)
        return (time.time() - last) > ALERT_COOLDOWN

    def _trigger_alert(self, frame: np.ndarray, alert_type: str):
        if not self._can_alert(alert_type):
            return
        self.last_alert[alert_type] = time.time()
        icon = ALERT_ICONS.get(alert_type, "⚠️")
        log.info(f"{icon} ALERT [{self.dept_name}]: {alert_type.upper()}")
        snap_path = save_snapshot(frame, self.dept_name, alert_type)
        if self.dept_info:
            insert_alert(self.dept_info["dept_id"], alert_type, snap_path, self.cam_idx)
            hod_email = self.dept_info.get("hod_email")
            send_email_alert(hod_email, self.dept_name, alert_type, snap_path)

    # ── Detection methods ─────────────────────────────────────────────────────

    def detect_objects(self, frame: np.ndarray) -> list:
        """
        YOLOv8 detection.
        Returns list of person bounding boxes [x1,y1,x2,y2].
        Also fires alerts for fire, eating, mobile usage.
        """
        person_boxes = []
        if not self.yolo_det:
            return person_boxes

        results = self.yolo_det(frame, verbose=False, conf=0.4)
        for r in results:
            for box in r.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.yolo_det.model.names[cls_id].lower()
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

                # Fire
                if "fire" in cls_name or "flame" in cls_name:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(frame, f"FIRE {conf:.0%}", (x1,y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    self._trigger_alert(frame, "fire")

                # ── Food / eating objects ──────────────────────────────────
                if cls_id in FOOD_CLASS_IDS:
                    # Choose label colour based on food type
                    if cls_id in PURE_FOOD_IDS:
                        food_color = (0, 200, 100)   # green — actual food
                    elif cls_id in DRINKING_IDS:
                        food_color = (255, 160, 0)   # orange — drink
                    else:
                        food_color = (100, 220, 255) # cyan — utensil/table

                    food_label = self.yolo_det.model.names[cls_id]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), food_color, 2)
                    cv2.putText(frame, f"{food_label} {conf:.0%}",
                                (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, food_color, 2)

                    # High-confidence alert for actual food; utensils/table alone = no alert
                    if cls_id in PURE_FOOD_IDS or cls_id in DRINKING_IDS:
                        self._eating_food_in_frame = True
                        self._trigger_alert(frame, "eating")

                # Mobile usage
                if cls_id == MOBILE_CLASS_ID:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,50,200), 2)
                    cv2.putText(frame, "PHONE", (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,50,200), 2)
                    self._trigger_alert(frame, "mobile_usage")

                # Person
                if cls_name == "person":
                    person_boxes.append([x1, y1, x2, y2])

        # Fighting: overlapping bounding boxes
        if len(person_boxes) >= 2:
            for i in range(len(person_boxes)):
                for j in range(i+1, len(person_boxes)):
                    if boxes_iou(person_boxes[i], person_boxes[j]) > FIGHT_IOU_THRESHOLD:
                        cv2.rectangle(frame,
                                      (person_boxes[i][0], person_boxes[i][1]),
                                      (person_boxes[i][2], person_boxes[i][3]),
                                      (0,50,255), 3)
                        self._trigger_alert(frame, "fighting")

        return person_boxes

    def detect_sleeping_faces(self, frame: np.ndarray):
        """MediaPipe FaceMesh: sleeping via EAR and head pitch (global counter)."""
        if not self.face_mesh:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        h, w = frame.shape[:2]

        if not res.multi_face_landmarks:
            self.ear_counters  = {}
            self.pitch_counters = {}
            return

        for face_idx, face_lm in enumerate(res.multi_face_landmarks):
            pid = face_idx  # approximate face index
            lm  = face_lm.landmark

            # EAR
            left_ear  = eye_aspect_ratio(lm, self.LEFT_EYE)
            right_ear = eye_aspect_ratio(lm, self.RIGHT_EYE)
            ear       = (left_ear + right_ear) / 2.0

            self.ear_counters.setdefault(pid, 0)
            if ear < EAR_THRESHOLD:
                self.ear_counters[pid] += 1
            else:
                self.ear_counters[pid] = 0

            if self.ear_counters[pid] >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "SLEEPING (EYES)", (30, 60 + face_idx * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                self._trigger_alert(frame, "sleeping")

            # Head pitch heuristic
            nose = lm[1]
            self.pitch_counters.setdefault(pid, 0)
            if (nose.y - 0.5) > 0.12:   # head drooping down
                self.pitch_counters[pid] += 1
            else:
                self.pitch_counters[pid] = 0

            if self.pitch_counters[pid] >= HEAD_PITCH_FRAMES:
                cv2.putText(frame, "SLEEPING (HEAD)", (30, 90 + face_idx * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,100,255), 2)
                self._trigger_alert(frame, "sleeping")

    # ── Playing detection thresholds (multi-signal) ───────────────────────────
    PLAY_ARM_SPREAD_RATIO  = 0.35   # wrist x-dist from hip-center > 35% of frame width
    PLAY_WRIST_RAISE_RATIO = 0.08   # wrist y < shoulder y – 8% frame height
    PLAY_ELBOW_RAISE_RATIO = 0.05   # elbow y < shoulder y – 5% frame height
    PLAY_WRIST_GAP_RATIO   = 0.40   # |left_wrist_x – right_wrist_x| > 40% frame width
    PLAY_SCORE_PLAY        = 2      # ≥2 signals → playing
    PLAY_SCORE_DANCE       = 3      # ≥3 signals → dancing

    def _playing_score(self, kps: list, frame_w: int, frame_h: int):
        """
        Multi-signal playing score for one person's keypoints.
        Returns (score, signals_list).
        Keypoint indices (COCO): 5=L_shoulder, 6=R_shoulder,
          7=L_elbow, 8=R_elbow, 9=L_wrist, 10=R_wrist, 11=L_hip, 12=R_hip.
        """
        score = 0
        signals = []

        def kp(idx):
            if idx >= len(kps): return None
            x, y, c = kps[idx]
            return (x, y, c) if c > 0.3 else None

        ls = kp(5);  rs = kp(6)
        le = kp(7);  re = kp(8)
        lw = kp(9);  rw = kp(10)
        lh = kp(11); rh = kp(12)

        # Hip centre x
        if lh and rh:
            hip_cx = (lh[0] + rh[0]) / 2.0
        elif lh:
            hip_cx = lh[0]
        elif rh:
            hip_cx = rh[0]
        else:
            hip_cx = frame_w / 2.0

        # Signal 1: arm spread — wrist far from hip centre
        if lw and abs(lw[0] - hip_cx) / (frame_w + 1e-6) > self.PLAY_ARM_SPREAD_RATIO:
            score += 1; signals.append("arm_spread_L")
        if rw and abs(rw[0] - hip_cx) / (frame_w + 1e-6) > self.PLAY_ARM_SPREAD_RATIO:
            score += 1; signals.append("arm_spread_R")

        # Signal 2: wrist raised above shoulder
        thresh_wr = self.PLAY_WRIST_RAISE_RATIO * frame_h
        if lw and ls and lw[1] < ls[1] - thresh_wr:
            score += 1; signals.append("wrist_raise_L")
        if rw and rs and rw[1] < rs[1] - thresh_wr:
            score += 1; signals.append("wrist_raise_R")

        # Signal 3: elbow raised above shoulder
        thresh_er = self.PLAY_ELBOW_RAISE_RATIO * frame_h
        if le and ls and le[1] < ls[1] - thresh_er:
            score += 1; signals.append("elbow_raise_L")
        if re and rs and re[1] < rs[1] - thresh_er:
            score += 1; signals.append("elbow_raise_R")

        # Signal 4: wide wrist gap (arms outstretched)
        if lw and rw:
            gap = abs(lw[0] - rw[0]) / (frame_w + 1e-6)
            if gap > self.PLAY_WRIST_GAP_RATIO:
                score += 1; signals.append("wide_wrist_gap")

        return score, signals

    def _eating_pose_score(self, kps: list, box_h: float) -> tuple:
        """
        Detect hand-to-mouth eating gesture via wrist-to-nose proximity.
        Returns (eating, signal_name) where eating is True/False.

        Keypoints used (COCO):
          0 = nose,  9 = left_wrist,  10 = right_wrist
          5 = left_shoulder, 6 = right_shoulder (used to filter out raised hands)

        Logic from the eating images:
          - Student holds food with ONE hand raised to face level
          - Wrist close to nose/mouth (within EATING_WRIST_NOSE_RATIO * person_height)
          - Elbow NOT above shoulder (distinguishes eating from hand-raising)
        """
        def kp(i):
            if i >= len(kps): return None
            x, y, c = kps[i]
            return (x, y, c) if c > 0.35 else None

        nose = kp(0)
        lw   = kp(9);  rw  = kp(10)
        ls   = kp(5);  rs  = kp(6)
        le   = kp(7);  re  = kp(8)

        if not nose or box_h < 1:
            return False, None

        thresh = EATING_WRIST_NOSE_RATIO * box_h

        for wrist, shoulder, elbow, side in [
            (lw, ls, le, "left"), (rw, rs, re, "right")
        ]:
            if not wrist:
                continue
            dist = math.sqrt((wrist[0] - nose[0])**2 + (wrist[1] - nose[1])**2)
            if dist < thresh:
                # Exclude: elbow above shoulder = hand-raising, not eating
                if elbow and shoulder and elbow[1] < shoulder[1] - 15:
                    continue   # this is hand-raising posture
                return True, f"wrist_near_nose_{side}"

        return False, None

    def _sleeping_pose_score(self, kps: list, box_h: float) -> tuple:
        """
        Detect sleeping based on posture (head resting on desk or hand).
        Returns (sleeping, signal_name).
        """
        def kp(i):
            if i >= len(kps): return None
            x, y, c = kps[i]
            return (x, y, c) if c > 0.35 else None

        nose = kp(0)
        lear = kp(3); rear = kp(4)
        ls = kp(5); rs = kp(6)
        lw = kp(9); rw = kp(10)

        if box_h < 1:
            return False, None
            
        # Get head y-coordinate (prefer nose, fallback to ears)
        head_y = None
        if nose: head_y = nose[1]
        elif lear and rear: head_y = (lear[1] + rear[1]) / 2.0
        elif lear: head_y = lear[1]
        elif rear: head_y = rear[1]
        
        if head_y is None:
            return False, None

        # Get shoulder y-coordinate
        shoulder_y = None
        if ls and rs: shoulder_y = (ls[1] + rs[1]) / 2.0
        elif ls: shoulder_y = ls[1]
        elif rs: shoulder_y = rs[1]

        if shoulder_y is None:
            return False, None

        # Posture 1: Head is completely down (head_y is very close to or below shoulder_y)
        head_shoulder_dist = shoulder_y - head_y
        if head_shoulder_dist < (0.15 * box_h):
            return True, "head_on_desk"
            
        # Posture 2: Head resting on hand
        # Distance from wrist to head is very small, and head_shoulder_dist is small (slouched)
        thresh_wrist = 0.25 * box_h
        if head_shoulder_dist < (0.3 * box_h):
            for wrist, side in [(lw, "left"), (rw, "right")]:
                if wrist:
                    dist_to_head = math.sqrt((wrist[0] - (nose[0] if nose else wrist[0]))**2 + 
                                             (wrist[1] - head_y)**2)
                    if dist_to_head < thresh_wrist:
                        return True, f"head_on_{side}_hand"

        return False, None

    def _detect_fighting_posture(self, kps1: list, kps2: list) -> bool:
        """
        Check for aggressive posture between two overlapping people.
        Indicators: A wrist is raised and extremely close to the other person's head/neck/shoulder
        (e.g., punching, grabbing collar, pulling hair, headlock).
        """
        def kp(kps, idx):
            if idx >= len(kps): return None
            x, y, c = kps[idx]
            return (x, y, c) if c > 0.4 else None

        def wrist_near_head(p_attacker, p_victim):
            a_lw = kp(p_attacker, 9); a_rw = kp(p_attacker, 10)
            v_nose = kp(p_victim, 0)
            v_ls = kp(p_victim, 5); v_rs = kp(p_victim, 6)
            
            targets = [v for v in [v_nose, v_ls, v_rs] if v is not None]
            if not targets:
                return False
                
            for wrist in [a_lw, a_rw]:
                if not wrist: continue
                
                # Attacker's wrist must be raised (above their own elbow or hip)
                a_ls = kp(p_attacker, 5); a_rs = kp(p_attacker, 6)
                if a_ls and a_rs:
                    shoulder_y = (a_ls[1] + a_rs[1]) / 2.0
                    if wrist[1] > shoulder_y + 80: # Wrist too low to be a punch/grab to upper body
                        continue
                        
                for target in targets:
                    dist = math.sqrt((wrist[0] - target[0])**2 + (wrist[1] - target[1])**2)
                    if dist < 65: # Within 65 pixels = physical contact / grabbing
                        return True
            return False

        # Check if either person is attacking the other
        if wrist_near_head(kps1, kps2) or wrist_near_head(kps2, kps1):
            return True
            
        return False

    def detect_pose_activities(self, frame: np.ndarray):
        """YOLOv8-Pose: dancing, playing (multi-signal), hand raising, standing/sitting."""
        if not self.yolo_pose:
            return

        results = self.yolo_pose(frame, verbose=False, conf=0.4)
        h, w    = frame.shape[:2]

        for r in results:
            if r.keypoints is None:
                continue
            kp_data  = r.keypoints.data        # (N, 17, 3)
            boxes_xy = r.boxes.xyxy if r.boxes is not None else None

            for idx, person_kps in enumerate(kp_data):
                kps = person_kps.tolist()

                # ── Eating: pose-based wrist-to-nose detection ───────────────
                box_h = 0.0
                if boxes_xy is not None and idx < len(boxes_xy):
                    bvals = boxes_xy[idx]
                    box_h = float(bvals[3]) - float(bvals[1])

                eating, eat_signal = self._eating_pose_score(kps, box_h)
                self.eating_pose_counters.setdefault(idx, 0)
                if eating:
                    self.eating_pose_counters[idx] += 1
                else:
                    self.eating_pose_counters[idx] = max(0, self.eating_pose_counters[idx] - 1)

                if self.eating_pose_counters[idx] >= EATING_POSE_FRAMES:
                    # Draw eating label near the person's face area
                    if boxes_xy is not None and idx < len(boxes_xy):
                        bx1 = int(boxes_xy[idx][0])
                        by1 = int(boxes_xy[idx][1])
                        cv2.putText(frame, "EATING", (bx1, by1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 100), 2)
                    self._trigger_alert(frame, "eating")
                    log.info(f"[{self.dept_name}] Eating gesture detected (signal={eat_signal})")

                # ── Sleeping: pose-based wrist-to-head / head-to-desk ────────
                sleeping, sleep_signal = self._sleeping_pose_score(kps, box_h)
                self.sleeping_pose_counters.setdefault(idx, 0)
                if sleeping:
                    self.sleeping_pose_counters[idx] += 1
                else:
                    self.sleeping_pose_counters[idx] = max(0, self.sleeping_pose_counters[idx] - 2)

                if self.sleeping_pose_counters[idx] >= 45: # ~1.5s
                    if boxes_xy is not None and idx < len(boxes_xy):
                        bx1 = int(boxes_xy[idx][0])
                        by1 = int(boxes_xy[idx][1])
                        cv2.putText(frame, "SLEEPING", (bx1, by1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self._trigger_alert(frame, "sleeping")
                    # Log only occasionally to prevent spam
                    if self.sleeping_pose_counters[idx] == 45:
                        log.info(f"[{self.dept_name}] Sleeping posture detected (signal={sleep_signal})")

                # ── Playing / Dancing (multi-signal heuristic) ───────────────
                play_score, signals = self._playing_score(kps, w, h)

                # Velocity as one extra bonus signal
                vel = keypoint_velocity(kps, self.prev_keypoints)
                if vel > POSE_VEL_THRESHOLD:
                    play_score += 1
                    signals.append("velocity")

                if play_score >= self.PLAY_SCORE_DANCE:
                    cv2.putText(frame, f"DANCING! ({play_score})", (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                    self._trigger_alert(frame, "dancing")
                elif play_score >= self.PLAY_SCORE_PLAY:
                    cv2.putText(frame, f"PLAYING! ({play_score})", (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    self._trigger_alert(frame, "playing")

                self.prev_keypoints = kps

                # ── Hand Raising ──────────────────────────────────────────
                # Keypoints: 5=left_shoulder, 6=right_shoulder, 9=left_wrist, 10=right_wrist
                try:
                    l_shoulder_y = kps[5][1]
                    r_shoulder_y = kps[6][1]
                    l_wrist_y    = kps[9][1]
                    r_wrist_y    = kps[10][1]
                    l_wrist_conf = kps[9][2]
                    r_wrist_conf = kps[10][2]

                    # Only trigger hand raise if NOT already classified as playing/dancing
                    if play_score < self.PLAY_SCORE_PLAY:
                        if (l_wrist_conf > 0.4 and
                                l_wrist_y < l_shoulder_y - HAND_RAISE_MARGIN):
                            self._trigger_alert(frame, "hand_raising")
                            lx, ly = int(kps[9][0]), int(kps[9][1])
                            cv2.putText(frame, "HAND UP", (lx, ly - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,200), 2)

                        if (r_wrist_conf > 0.4 and
                                r_wrist_y < r_shoulder_y - HAND_RAISE_MARGIN):
                            self._trigger_alert(frame, "hand_raising")
                            rx, ry = int(kps[10][0]), int(kps[10][1])
                            cv2.putText(frame, "HAND UP", (rx, ry - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,200), 2)
                except (IndexError, TypeError):
                    pass

                # ── Standing / Sitting ────────────────────────────────────
                if boxes_xy is not None and idx < len(boxes_xy):
                    bx1, by1, bx2, by2 = [int(v) for v in boxes_xy[idx]]
                    bh = by2 - by1
                    bw = bx2 - bx1
                    if bw > 0:
                        ratio = bh / (bw + 1e-6)
                        if ratio > 2.0:
                            activity = "Standing"
                        elif ratio > 1.0:
                            activity = "Sitting"
                        else:
                            activity = "Active"
                        cv2.putText(frame, activity,
                                    (bx1, by2 + 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,220,0), 2)

            # ── Multi-person Interactions (Fighting / Grappling) ─────────────
            num_people = len(kp_data)
            if num_people >= 2 and boxes_xy is not None:
                for i in range(num_people):
                    for j in range(i+1, num_people):
                        box_i = boxes_xy[i].tolist()
                        box_j = boxes_xy[j].tolist()
                        iou = boxes_iou(box_i, box_j)
                        
                        # Soft IoU threshold for proximity (people standing close)
                        if iou > 0.12:
                            kps_i = kp_data[i].tolist()
                            kps_j = kp_data[j].tolist()
                            
                            is_fighting = self._detect_fighting_posture(kps_i, kps_j)
                            
                            pair_id = f"{min(i,j)}_{max(i,j)}"
                            self.fight_pose_counters.setdefault(pair_id, 0)
                            if is_fighting:
                                self.fight_pose_counters[pair_id] += 1
                            else:
                                self.fight_pose_counters[pair_id] = max(0, self.fight_pose_counters[pair_id] - 1)
                                
                            # Require ~5 frames of continuous physical contact
                            if self.fight_pose_counters[pair_id] >= 5:
                                cv2.rectangle(frame, (int(box_i[0]), int(box_i[1])), (int(box_i[2]), int(box_i[3])), (0,0,255), 3)
                                cv2.rectangle(frame, (int(box_j[0]), int(box_j[1])), (int(box_j[2]), int(box_j[3])), (0,0,255), 3)
                                cv2.putText(frame, "FIGHTING (GRAPPLE)!", (int(box_i[0]), int(box_i[1])-15), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                                self._trigger_alert(frame, "fighting")
                                if self.fight_pose_counters[pair_id] == 5:
                                    log.info(f"[{self.dept_name}] Fighting posture detected (grappling/contact)")

    def gemini_worker(self):
        """Background thread that sends 1 frame every 5 seconds to Gemini API."""
        if not GEMINI_ENABLED:
            return
            
        log.info(f"[{self.dept_name}] Gemini background analysis thread started (1 frame / 5s).")
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config={"response_mime_type": "application/json"})
        prompt = """
Analyze this classroom image. Tell me if any of the following activities are clearly happening right now.
Respond strictly in this JSON format (use true or false):
{
  "fighting": false,
  "sleeping": false,
  "eating": false,
  "dancing": false
}
"""
        while self.running:
            time.sleep(5)
            
            if self.latest_frame is None:
                continue

            # Work on a copy of the frame to avoid race conditions
            frame_copy = self.latest_frame.copy()
            try:
                rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                response = model.generate_content([prompt, pil_img])
                data = json.loads(response.text)
                
                detected_activities = [k for k, v in data.items() if v is True]
                if detected_activities:
                    log.info(f"[GEMINI] Cloud AI detected: {', '.join(detected_activities).upper()} in {self.dept_name}!")
                    
                for activity in detected_activities:
                    if activity in ["fighting", "sleeping", "eating", "dancing"]:
                        # Draw alert overlay for the live feed window
                        h, w = self.latest_frame.shape[:2]
                        cv2.putText(self.latest_frame, f"GEMINI DETECTED: {activity.upper()}", 
                                    (w//2 - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        # Trigger standard alert (saves snapshot, sends email, updates DB)
                        self._trigger_alert(frame_copy, activity)
                        
            except Exception as e:
                log.error(f"[GEMINI] Analysis failed: {e}")

    def run(self):
        """Main detection loop for this department's camera."""
        log.info(f"[{self.dept_name}] Starting camera on index {self.cam_idx} …")
        cap = cv2.VideoCapture(self.cam_idx)

        if not cap.isOpened():
            log.error(f"[{self.dept_name}] Could not open camera index {self.cam_idx}")
            return

        log.info(f"[{self.dept_name}] Camera opened. Press Q in window to quit.")
        frame_count = 0
        
        self.running = True
        gemini_thread = threading.Thread(target=self.gemini_worker, daemon=True)
        gemini_thread.start()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                log.warning(f"[{self.dept_name}] Frame read failed — retrying …")
                time.sleep(0.5)
                continue

            self.latest_frame = frame
            frame_count += 1
            h, w = frame.shape[:2]

            # ── Object detection (persons, fire, food, phone) ──────────────
            person_boxes = self.detect_objects(frame)

            # ── Update person tracker ──────────────────────────────────────
            tracked = self.tracker.update(person_boxes)
            person_count = len(tracked)

            # Log count changes
            if person_count != self.prev_count:
                log.info(f"[{self.dept_name}] Person count changed: {person_count}")
                for pid in tracked:
                    log.info(f"  Person {pid}: detected")
                self.prev_count = person_count

            # Draw person IDs on frame
            for box, (pid, centroid) in zip(person_boxes, tracked.items()):
                x1, y1, x2, y2 = box
                status = self.person_status.get(pid, "Active")
                color  = (0,0,255) if status == "Sleeping" else (0,220,100)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                label = f"Person {pid}: {status}"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # ── Sleeping detection ─────────────────────────────────────────
            self.detect_sleeping_faces(frame)

            # ── Pose activities every 5 frames ─────────────────────────────
            if frame_count % 5 == 0:
                self.detect_pose_activities(frame)

            # ── HUD overlay ────────────────────────────────────────────────
            sleeping_count = sum(1 for s in self.person_status.values() if s == "Sleeping")
            ts = datetime.datetime.now().strftime("%H:%M:%S")

            # Top banner
            cv2.rectangle(frame, (0, 0), (w, 38), (20, 20, 20), -1)
            cv2.putText(frame, f"[{self.dept_name}]  Students: {person_count}  |  Sleeping: {sleeping_count}  |  Active: {max(0, person_count - sleeping_count)}  |  {ts}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 120), 2)

            cv2.imshow(f"College CMS — {self.dept_name}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cap.release()
        cv2.destroyAllWindows()
        log.info(f"[{self.dept_name}] Camera service stopped.")


# ── Test mode ─────────────────────────────────────────────────────────────────
def test_fire_detection():
    log.info("=== Fire Detection Test ===")
    if not YOLO_AVAILABLE:
        log.error("ultralytics not installed. Cannot test.")
        return
    model    = YOLO("yolov8n.pt")
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (150, 100), (450, 380), (0, 100, 255), -1)
    cv2.putText(test_img, "FIRE TEST IMAGE", (80, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
    test_path = SNAPSHOT_DIR / "test_fire_input.jpg"
    cv2.imwrite(str(test_path), test_img)
    results    = model(test_img, verbose=False)
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

    config    = load_config()
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
