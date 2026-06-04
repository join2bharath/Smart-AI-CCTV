"""
test_playing_detection.py
=========================
Analyzes a classroom image for "playing" activity using:
  - YOLOv8 person detection
  - YOLOv8-Pose keypoint analysis
  - Custom playing heuristics based on:
      * High arm extension (wrists far from body center)
      * Multiple people clustered close together
      * Arms raised or spread outward
      * Object interaction (arms reaching toward same point)

Usage:
    python test_playing_detection.py --image <path_to_image>
"""

import cv2
import numpy as np
import math
import argparse
import datetime
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[!] ultralytics not installed. Run: pip install ultralytics")
    YOLO_AVAILABLE = False

# ── Playing Detection Heuristics ─────────────────────────────────────────────
# YOLOv8-Pose keypoint indices (COCO format):
# 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
# 5=left_shoulder, 6=right_shoulder
# 7=left_elbow, 8=right_elbow
# 9=left_wrist, 10=right_wrist
# 11=left_hip, 12=right_hip
# 13=left_knee, 14=right_knee
# 15=left_ankle, 16=right_ankle

ARM_SPREAD_THRESHOLD    = 0.40   # wrist x-distance from hip center > 40% of body width
ARM_RAISE_THRESHOLD     = 0.10   # wrist y < shoulder y - 10% frame height
GROUP_DISTANCE_THRESHOLD = 200   # pixels — persons within this are "clustered"
PLAYING_SCORE_THRESHOLD  = 2     # need at least 2 playing signals to flag


def get_kp(kps, idx, frame_w, frame_h):
    """Return (x, y, conf) in pixel coordinates, or None if low conf."""
    if idx >= len(kps):
        return None
    x, y, c = kps[idx]
    if c < 0.3:
        return None
    return int(x), int(y), c


def compute_playing_score(kps, frame_w, frame_h):
    """
    Returns (score, reasons) where score >= PLAYING_SCORE_THRESHOLD → playing.
    Reasons is a list of detected playing signals.
    """
    score   = 0
    reasons = []

    l_shoulder = get_kp(kps, 5,  frame_w, frame_h)
    r_shoulder = get_kp(kps, 6,  frame_w, frame_h)
    l_wrist    = get_kp(kps, 9,  frame_w, frame_h)
    r_wrist    = get_kp(kps, 10, frame_w, frame_h)
    l_hip      = get_kp(kps, 11, frame_w, frame_h)
    r_hip      = get_kp(kps, 12, frame_w, frame_h)
    l_elbow    = get_kp(kps, 7,  frame_w, frame_h)
    r_elbow    = get_kp(kps, 8,  frame_w, frame_h)

    # Body center X
    if l_hip and r_hip:
        body_cx = (l_hip[0] + r_hip[0]) / 2
        body_w  = abs(r_hip[0] - l_hip[0]) + 1
    elif l_shoulder and r_shoulder:
        body_cx = (l_shoulder[0] + r_shoulder[0]) / 2
        body_w  = abs(r_shoulder[0] - l_shoulder[0]) + 1
    else:
        body_cx = frame_w / 2
        body_w  = frame_w * 0.2

    # 1. Left wrist spread (arm extended outward)
    if l_wrist:
        spread = abs(l_wrist[0] - body_cx) / (body_w + 1)
        if spread > ARM_SPREAD_THRESHOLD:
            score += 1
            reasons.append(f"L-arm spread {spread:.1f}x")

    # 2. Right wrist spread
    if r_wrist:
        spread = abs(r_wrist[0] - body_cx) / (body_w + 1)
        if spread > ARM_SPREAD_THRESHOLD:
            score += 1
            reasons.append(f"R-arm spread {spread:.1f}x")

    # 3. Left wrist raised above shoulder
    if l_wrist and l_shoulder:
        if l_wrist[1] < l_shoulder[1] - (frame_h * ARM_RAISE_THRESHOLD):
            score += 1
            reasons.append("L-wrist above shoulder")

    # 4. Right wrist raised above shoulder
    if r_wrist and r_shoulder:
        if r_wrist[1] < r_shoulder[1] - (frame_h * ARM_RAISE_THRESHOLD):
            score += 1
            reasons.append("R-wrist above shoulder")

    # 5. Arms spread wide from each other (both wrists far apart)
    if l_wrist and r_wrist:
        wrist_spread = abs(l_wrist[0] - r_wrist[0])
        if l_shoulder and r_shoulder:
            shoulder_w = abs(l_shoulder[0] - r_shoulder[0]) + 1
            if wrist_spread > shoulder_w * 1.5:
                score += 1
                reasons.append(f"Wide wrist spread {wrist_spread}px")

    # 6. Elbow raised high (arm bent upward)
    if l_elbow and l_shoulder:
        if l_elbow[1] < l_shoulder[1]:
            score += 1
            reasons.append("L-elbow raised")
    if r_elbow and r_shoulder:
        if r_elbow[1] < r_shoulder[1]:
            score += 1
            reasons.append("R-elbow raised")

    return score, reasons


def persons_are_clustered(boxes, threshold=GROUP_DISTANCE_THRESHOLD):
    """
    Returns True if at least 2 persons are within threshold distance of each other.
    This signals group interaction / playing together.
    """
    centroids = [((b[0]+b[2])//2, (b[1]+b[3])//2) for b in boxes]
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            dx = centroids[i][0] - centroids[j][0]
            dy = centroids[i][1] - centroids[j][1]
            if math.sqrt(dx*dx + dy*dy) < threshold:
                return True
    return False


def analyze_image(image_path: str, output_path: str = None):
    """
    Run playing detection on a still image.
    Draws bounding boxes, keypoints, playing scores, and group flags.
    Saves the annotated output image.
    """
    if not YOLO_AVAILABLE:
        print("[ERROR] YOLO not available.")
        return

    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return

    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    h, w = frame.shape[:2]
    print(f"[INFO] Image size: {w}x{h}")
    print(f"[INFO] Loading YOLOv8 models …")

    yolo_det  = YOLO("yolov8n.pt")
    yolo_pose = YOLO("yolov8n-pose.pt")

    # ── 1. Person Detection ────────────────────────────────────────────────
    print("[INFO] Running object detection …")
    det_results = yolo_det(frame, verbose=False, conf=0.3)
    person_boxes = []
    for r in det_results:
        for box in r.boxes:
            cls_id   = int(box.cls[0])
            cls_name = yolo_det.model.names[cls_id].lower()
            conf     = float(box.conf[0])
            if cls_name == "person":
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                person_boxes.append([x1, y1, x2, y2])

    print(f"[INFO] Persons detected: {len(person_boxes)}")

    # ── 2. Group Clustering Check ─────────────────────────────────────────
    group_playing = persons_are_clustered(person_boxes)
    if group_playing:
        print("[INFO] Group clustering detected → GROUP PLAYING likely!")

    # ── 3. Pose Detection & Playing Analysis ─────────────────────────────
    print("[INFO] Running pose detection …")
    pose_results = yolo_pose(frame, verbose=False, conf=0.3)

    playing_persons = 0
    total_persons   = 0

    for r in pose_results:
        if r.keypoints is None:
            continue
        kp_data  = r.keypoints.data
        boxes_xy = r.boxes.xyxy if r.boxes is not None else None

        for idx, person_kps in enumerate(kp_data):
            total_persons += 1
            kps   = person_kps.tolist()
            score, reasons = compute_playing_score(kps, w, h)

            # Factor in group clustering
            if group_playing:
                score  += 1
                reasons.append("Group clustering")

            is_playing = score >= PLAYING_SCORE_THRESHOLD
            if is_playing:
                playing_persons += 1

            # Draw bounding box
            if boxes_xy is not None and idx < len(boxes_xy):
                bx1, by1, bx2, by2 = [int(v) for v in boxes_xy[idx]]
                color = (0, 80, 255) if is_playing else (0, 200, 80)
                label = f"Person {idx+1}: {'PLAYING!' if is_playing else 'Active'}"
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 3)
                cv2.putText(frame, label, (bx1, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                # Show score
                cv2.putText(frame, f"Score: {score}",
                            (bx1, by2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Print reasons
                print(f"  Person {idx+1}: score={score}, playing={is_playing}")
                for r_ in reasons:
                    print(f"    ↳ {r_}")

            # Draw keypoints
            for ki, (kx, ky, kc) in enumerate(kps):
                if kc > 0.3:
                    cv2.circle(frame, (int(kx), int(ky)), 5, (255, 200, 0), -1)

    # ── 4. HUD Banner ────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 15), -1)

    group_text = "GROUP PLAYING DETECTED!" if group_playing else "No group play"
    hud = (f"Persons: {total_persons}  |  Playing: {playing_persons}  |  "
           f"{group_text}  |  {ts}")
    cv2.putText(frame, hud, (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 120), 2)

    # Playing alert banner
    if playing_persons > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-60), (w, h), (0, 0, 200), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.putText(frame, f"🎮  PLAYING ALERT!  {playing_persons} student(s) playing in class!",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # ── 5. Save Output ───────────────────────────────────────────────────
    if output_path is None:
        ts_str      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(img_path.parent / f"playing_detected_{ts_str}.jpg")

    cv2.imwrite(output_path, frame)
    print(f"\n[✓] Annotated image saved: {output_path}")
    print(f"[✓] Total persons : {total_persons}")
    print(f"[✓] Playing       : {playing_persons}")
    print(f"[✓] Group playing : {group_playing}")

    return output_path, playing_persons, group_playing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Playing Detection Test")
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--output", default=None,  help="Path to save annotated output")
    args = parser.parse_args()
    analyze_image(args.image, args.output)
