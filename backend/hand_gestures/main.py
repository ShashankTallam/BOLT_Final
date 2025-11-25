#!/usr/bin/env python3
"""
hand_gestures/main.py

Robust hand gesture recognition using MediaPipe + optional ML model + rule-based 3D detectors.
Writes results to output.txt and frame.jpg, stops when stop.txt exists.
Compatible with existing CSV layout: label,x0..x20,y0..y20
"""

import os
import time
import csv
import shutil
import cv2
import joblib
import numpy as np
import mediapipe as mp
from collections import deque, Counter

# optional controller helper (if present)
try:
    import controller as cnt
except Exception:
    cnt = None

# ---------------- CONFIG ----------------
THIS_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(THIS_DIR, "gesture_data.csv")
MODEL_FILE = os.path.join(THIS_DIR, "gesture_model.pkl")
OUTPUT_FILE = os.path.join(THIS_DIR, "output.txt")
STOP_FILE = os.path.join(THIS_DIR, "stop.txt")
FRAME_FILE = os.path.join(THIS_DIR, "frame.jpg")

# runtime toggles
TRAINING_MODE = False    # set True only when collecting labeled samples interactively
USE_MODEL = True        # set False to force rule-based detection (useful for debugging)
MODEL_CONF_THRESHOLD = 0.60
SMOOTH_WINDOW = 3       # frames for majority vote smoothing

time.sleep(0.2)

# ---------------- helpers ----------------
def open_camera(device=0, retries=6, delay=0.5):
    for attempt in range(1, retries+1):
        for backend in (getattr(cv2, 'CAP_DSHOW', None), None):
            try:
                if backend is None:
                    cap = cv2.VideoCapture(device)
                else:
                    cap = cv2.VideoCapture(device, backend)
                if cap.isOpened():
                    print(f"[INFO] Camera opened backend={backend}")
                    return cap
                try:
                    cap.release()
                except Exception:
                    pass
            except Exception as e:
                print("[WARN] Camera open failed:", e)
        time.sleep(delay)
    print("[ERROR] Failed to open camera")
    return None

video = open_camera(0)

# ---------------- CSV helpers ----------------
def save_landmarks(label, lmList):
    """
    Save landmarks to CSV. Compatibility maintained: save x and y pixel coords only.
    lmList elements are [id, x_pixel, y_pixel, z_rel]
    """
    file_exists = os.path.exists(DATA_FILE)
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
            writer.writerow(header)
        if len(lmList) != 21:
            print(f"[WARN] save_landmarks skipped: expected 21 got {len(lmList)}")
            return
        row = [label] + [p[1] for p in lmList] + [p[2] for p in lmList]
        writer.writerow(row)
        print(f"[INFO] Saved sample: {label}")

def clean_dataset_file():
    if not os.path.exists(DATA_FILE):
        return DATA_FILE
    tmp_rows = []
    kept, fixed, skipped = 0,0,0
    with open(DATA_FILE, newline="") as fin:
        reader = csv.reader(fin)
        try:
            _ = next(reader)
        except StopIteration:
            pass
        header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
        tmp_rows.append(header)
        for row in reader:
            n = len(row)
            if n == 43:
                tmp_rows.append(row); kept += 1
            elif n == 85:
                label = row[0]
                xs = row[1:1+42][:21]
                ys = row[1+42:1+42+42][:21]
                tmp = [label] + xs + ys
                tmp_rows.append(tmp); fixed += 1
            else:
                skipped += 1
    try:
        shutil.copy2(DATA_FILE, DATA_FILE + ".bak")
    except Exception:
        pass
    with open(DATA_FILE, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerows(tmp_rows)
    print(f"[INFO] Cleaned dataset kept={kept} fixed={fixed} skipped={skipped}")
    return DATA_FILE

# ---------------- Model helpers ----------------
def inspect_model(model_path):
    try:
        m = joblib.load(model_path)
        print("[INFO] model type:", type(m))
        if hasattr(m, "classes_"):
            print("[INFO] classes:", list(m.classes_))
        print("[INFO] has predict_proba:", hasattr(m, "predict_proba"))
        return m
    except Exception as e:
        print("[WARN] inspect_model failed:", e)
        return None

def try_load_model():
    if not USE_MODEL:
        print("[INFO] USE_MODEL=False -> skipping model load")
        return None
    if not os.path.exists(MODEL_FILE):
        print(f"[INFO] model file not found: {MODEL_FILE}")
        return None
    try:
        m = joblib.load(MODEL_FILE)
        print("[INFO] Loaded model from", MODEL_FILE)
        return m
    except Exception as e:
        print("[WARN] load model failed:", e)
        return None

# ---------------- Feature normalizer (pixel) ----------------
def normalize_landmarks(lmList):
    if len(lmList) != 21:
        return None
    pts = np.array([[p[1], p[2]] for p in lmList], dtype=np.float32)
    wrist = pts[0].copy()
    pts -= wrist
    ref = np.linalg.norm(pts[9])
    if ref < 1.0:
        bbox = pts.max(axis=0) - pts.min(axis=0)
        ref = max(np.linalg.norm(bbox), 1.0)
    pts /= ref
    xs = pts[:,0].tolist(); ys = pts[:,1].tolist()
    return xs + ys

# ---------------- 3D RULE-BASED DETECTION ----------------
def finger_states_3d(lmList, z_thresh=0.02):
    """
    Return [thumb,index,middle,ring,pinky] booleans (True = extended).
    lmList: [id, x_px, y_px, z_rel]
    Uses z (relative depth) because it is robust to hand rotation.
    """
    if len(lmList) != 21:
        return None
    pts = {p[0]: (p[1], p[2], p[3]) for p in lmList}
    states = []
    # thumb
    thumb_tip_z = pts[4][2]; thumb_ip_z = pts[3][2]
    states.append(thumb_tip_z < thumb_ip_z - z_thresh)
    # other fingers: tip vs pip
    for tip_idx, pip_idx in ((8,6),(12,10),(16,14),(20,18)):
        tip_z = pts[tip_idx][2]; pip_z = pts[pip_idx][2]
        states.append(tip_z < pip_z - z_thresh)
    return states

def detect_fist(lmList):
    s = finger_states_3d(lmList)
    return False if s is None else (not any(s))

def detect_open_hand(lmList):
    s = finger_states_3d(lmList)
    return False if s is None else all(s)

def detect_L_gesture(lmList):
    s = finger_states_3d(lmList)
    if s is None: return False
    thumb,index,middle,ring,pinky = s
    return thumb and index and (not middle) and (not ring) and (not pinky)

def detect_peace(lmList):
    s = finger_states_3d(lmList)
    if s is None: return False
    thumb,index,middle,ring,pinky = s
    return index and middle and (not ring) and (not pinky)

def detect_pointing(lmList):
    s = finger_states_3d(lmList)
    if s is None: return False
    thumb,index,middle,ring,pinky = s
    return index and (not middle) and (not ring) and (not pinky)

def detect_ok(lmList, pix_thresh_ratio=0.35):
    if len(lmList) != 21: return False
    pts = {p[0]: (p[1], p[2], p[3]) for p in lmList}
    tx,ty = pts[4][0], pts[4][1]; ix,iy = pts[8][0], pts[8][1]
    wx,wy = pts[0][0], pts[0][1]; mx,my = pts[9][0], pts[9][1]
    hand_scale = np.hypot(mx-wx, my-wy)
    if hand_scale < 1.0: hand_scale = 1.0
    dist = np.hypot(tx-ix, ty-iy)
    s = finger_states_3d(lmList)
    if s is None: return False
    thumb,index,middle,ring,pinky = s
    others_folded = (not middle) and (not ring) and (not pinky)
    return bool(dist < hand_scale * pix_thresh_ratio and others_folded)

# ---------------- Number gestures 1-5 ----------------
def detect_number_1(lmList):
    s = finger_states_3d(lmList)
    if s is None: return False
    thumb,index,middle,ring,pinky = s
    return index and (not middle) and (not ring) and (not pinky)

def detect_number_2(lmList):
    s = finger_states_3d(lmList)
    if s is None: return False
    thumb,index,middle,ring,pinky = s
    return index and middle and (not ring) and (not pinky)

def detect_number_3(lmList):
    s = finger_states_3d(lmList)
    if s is None: return False
    thumb,index,middle,ring,pinky = s
    return index and middle and ring and (not pinky)

def detect_number_4(lmList):
    s = finger_states_3d(lmList)
    if s is None: return False
    thumb,index,middle,ring,pinky = s
    return (not thumb) and index and middle and ring and pinky

def detect_number_5(lmList):
    return detect_open_hand(lmList)

# ---------------- Main loop ----------------
mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

pred_queue = deque(maxlen=SMOOTH_WINDOW)
model = try_load_model()

try:
    with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        last_text = ""
        while True:
            if os.path.exists(STOP_FILE):
                print("[INFO] stop file found -> exit")
                break

            if video is None or not getattr(video, "isOpened", lambda: False)():
                print("[WARN] camera not opened, retrying...")
                video = open_camera(0)
                if video is None:
                    time.sleep(0.5)
                    continue

            ret, frame = video.read()
            if not ret or frame is None:
                time.sleep(0.2)
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            lmList = []
            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)
                first_hand = results.multi_hand_landmarks[0]
                h,w,_ = image.shape
                for id, lm in enumerate(first_hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cz = lm.z
                    lmList.append([id, cx, cy, cz])

            gesture = None
            if len(lmList) == 21:
                # 1) try model (if available)
                model_used = False
                if model is not None:
                    try:
                        feat = normalize_landmarks(lmList)
                        if feat is not None and hasattr(model, "predict_proba"):
                            probs = model.predict_proba([feat])[0]
                            top_idx = int(np.argmax(probs))
                            pred = model.classes_[top_idx]
                            prob = float(probs[top_idx])
                            print(f"[DEBUG] model pred={pred} p={prob:.2f}")
                            if prob >= MODEL_CONF_THRESHOLD:
                                gesture = pred
                                model_used = True
                        elif feat is not None:
                            pred = model.predict([feat])[0]
                            print(f"[DEBUG] model pred (no-proba)={pred}")
                            gesture = pred
                            model_used = True
                    except Exception as e:
                        print("[WARN] model predict failed:", e)
                        gesture = None

                # 2) fallback rule-based if no confident model result
                if gesture is None:
                    try:
                        if detect_fist(lmList): gesture = "Fist"
                        elif detect_open_hand(lmList): gesture = "Open"
                        elif detect_L_gesture(lmList): gesture = "L"
                        elif detect_peace(lmList): gesture = "Peace"
                        elif detect_pointing(lmList): gesture = "Pointing"
                        elif detect_ok(lmList): gesture = "OK"
                        elif detect_number_1(lmList): gesture = "1"
                        elif detect_number_2(lmList): gesture = "2"
                        elif detect_number_3(lmList): gesture = "3"
                        elif detect_number_4(lmList): gesture = "4"
                        elif detect_number_5(lmList): gesture = "5"
                    except Exception as e:
                        print("[WARN] rule-based detection error:", e)
                        gesture = None

                # 3) smoothing
                if gesture is not None:
                    pred_queue.append(gesture)
                    gesture = Counter(pred_queue).most_common(1)[0][0]
                else:
                    pred_queue.clear()

                # 4) hardware mapping
                try:
                    if cnt is not None:
                        if gesture == "L": cnt.led(1)
                        elif gesture == "ThumbsUp": cnt.led(5)
                        elif gesture == "Peace": cnt.led(2)
                        elif gesture == "Fist": cnt.led(3)
                        elif gesture == "Open": cnt.led(4)
                        elif gesture == "Pointing": cnt.led(6)
                        elif gesture == "OK": cnt.led(7)
                        elif gesture == "1": cnt.led(11)
                        elif gesture == "2": cnt.led(12)
                        elif gesture == "3": cnt.led(13)
                        elif gesture == "4": cnt.led(14)
                        elif gesture == "5": cnt.led(15)
                except Exception:
                    pass

                # 5) write output only when changed
                if gesture is not None:
                    new_text = f"Recognized Gesture: {gesture}"
                    if new_text != last_text:
                        try:
                            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                                f.write(new_text); f.flush()
                                try: os.fsync(f.fileno())
                                except Exception: pass
                        except Exception as e:
                            print("[WARN] write output failed:", e)
                        last_text = new_text

                # annotate
                if gesture is not None:
                    try:
                        cv2.putText(image, f"Gesture: {gesture}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)
                    except Exception:
                        pass

            # atomic frame write
            try:
                tmp = FRAME_FILE + ".tmp"
                try:
                    from PIL import Image
                    im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    im.save(tmp, format="JPEG", quality=85)
                except Exception:
                    cv2.imwrite(tmp, image)
                try:
                    os.replace(tmp, FRAME_FILE)
                except Exception:
                    os.rename(tmp, FRAME_FILE)
            except Exception as e:
                print("[WARN] frame write failed:", e)

finally:
    try:
        if video is not None:
            video.release()
    except Exception:
        pass
    try:
        if cnt is not None and hasattr(cnt, "cleanup"):
            cnt.cleanup()
    except Exception:
        pass
    print("[INFO] hand_gestures exiting")
