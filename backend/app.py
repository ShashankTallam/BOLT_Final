
from flask import Flask, jsonify, Response
import subprocess
import threading
from flask_cors import CORS
import os
import cv2
import time
import sys

app = Flask(__name__)
CORS(app)

# To store last recognized output
latest_result = {"type": None, "text": ""}
_threads = {}

# --------------------- #
# Helper Functions
# --------------------- #
def read_output_file(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print("Error reading output file:", e)
    return ""

def ensure_removed(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# --------------------- #
# LIP READING
# --------------------- #
def run_lip_reading():
    global latest_result
    base_dir = os.path.dirname(__file__)
    venv_python = os.path.join(base_dir, "LH", "Scripts", "python.exe")
    # robust path resolution for predict_script
    candidate1 = os.path.join(base_dir, "lip reading", "src", "predict.py")
    candidate2 = "/mnt/data/predict.py"  # uploaded/test file location
    if os.path.exists(candidate1):
        predict_script = candidate1
    elif os.path.exists(candidate2):
        predict_script = candidate2
    else:
        # keep candidate1 as default so existing behavior & error handling remain
        predict_script = candidate1
    print(f"[DEBUG] Using predict_script path: {predict_script}")
    output_file = os.path.join(base_dir, "lip reading", "output.txt")
    stop_file = os.path.join(base_dir, "lip reading", "stop.txt")

    ensure_removed(output_file)
    ensure_removed(stop_file)

    if not os.path.exists(venv_python):
        print("⚠️  venv python not found, will fallback to system python if available:", venv_python)

    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "lip_reading.log")
    try:
        print("▶️ Starting Lip Reading subprocess (Popen)...")
        with open(log_path, "a", encoding="utf-8") as lf:
            # choose the python executable: prefer venv, else use the running interpreter
            python_exec = venv_python if os.path.exists(venv_python) else sys.executable
            print(f"[DEBUG] lip_reading will use python_exec={python_exec}")
            # write debug info to the lip_reading log to help diagnose startup failures
            try:
                lf.write(f"[DEBUG] lip_reading will use python_exec={python_exec}\n")
                lf.write(f"[DEBUG] predict_script={predict_script}\n")
                lf.write(f"[DEBUG] predict_script exists={os.path.exists(predict_script)}\n")
                lf.write(f"[DEBUG] predict_script dirname exists={os.path.isdir(os.path.dirname(predict_script))}\n")
                lf.flush()
            except Exception:
                pass
            cmd = [python_exec, predict_script]
            # Set UTF-8 encoding to prevent UnicodeEncodeError from emoji/UTF-8 output
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env.setdefault('LANG', 'en_US.UTF-8')
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=os.path.dirname(predict_script) if os.path.isdir(os.path.dirname(predict_script)) else None, env=env)
            # store process so it can be inspected by other endpoints if needed
            _threads['lip_reading_proc'] = proc
            print(f"LipReading Popen started pid={proc.pid}")
    except Exception as e:
        print("❌ Lip Reading error:", e)

    text = read_output_file(output_file)
    latest_result = {"type": "lip-reading", "text": text or "No output"}


# --------------------- #
# HAND GESTURES
# --------------------- #
def run_hand_gestures():
    global latest_result
    base_dir = os.path.dirname(__file__)
    venv_python = os.path.join(base_dir, "LH", "Scripts", "python.exe")
    main_script = os.path.join(base_dir, "hand_gestures", "main.py")
    output_file = os.path.join(base_dir, "hand_gestures", "output.txt")
    stop_file = os.path.join(base_dir, "hand_gestures", "stop.txt")
    frame_file = os.path.join(base_dir, "hand_gestures", "frame.jpg")

    ensure_removed(output_file)
    ensure_removed(stop_file)

    if not os.path.exists(venv_python):
        print("⚠️  venv python not found, will fallback to system python if available:", venv_python)

    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "hand_gestures.log")
    try:
        print("▶️ Starting Hand Gesture subprocess (Popen)...")
        with open(log_path, "a", encoding="utf-8") as lf:
            python_exec = venv_python if os.path.exists(venv_python) else sys.executable
            print(f"[DEBUG] hand_gestures will use python_exec={python_exec}")
            cmd = [python_exec, main_script]
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=os.path.dirname(main_script))
            _threads['hand_gestures_proc'] = proc
            print(f"HandGesture Popen started pid={proc.pid}")
            # create a small placeholder frame so frontend has something to display immediately
            try:
                import numpy as _np
                placeholder = _np.zeros((240, 320, 3), dtype=_np.uint8)
                try:
                    import cv2 as _cv2
                    _cv2.putText(placeholder, 'Starting...', (10, 120), _cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                except Exception:
                    pass
                # Try Pillow first for reliable JPEG writing
                try:
                    from PIL import Image as _Image
                    im = _Image.fromarray(placeholder[:, :, ::-1])
                    tmp = frame_file + ".tmp"
                    im.save(tmp, format='JPEG', quality=85)
                    try:
                        os.replace(tmp, frame_file)
                    except Exception:
                        os.rename(tmp, frame_file)
                except Exception:
                    try:
                        import cv2 as _cv2
                        _cv2.imwrite(frame_file, placeholder)
                    except Exception as e:
                        print('[WARN] Could not write placeholder frame:', e)
            except Exception as e:
                print('[WARN] Could not write placeholder frame:', e)
    except Exception as e:
        print("❌ Hand gesture error:", e)

    text = read_output_file(output_file)
    latest_result = {"type": "hand-gesture", "text": text or "No output"}


# --------------------- #
# FLASK ROUTES
# --------------------- #
@app.route("/")
def home():
    return jsonify({"message": "Backend running successfully 🚀"})


@app.route("/start_lip_reading", methods=["GET"])
def start_lip_reading():
    t = threading.Thread(target=run_lip_reading, daemon=True)
    t.start()
    _threads["lip_reading"] = t
    return jsonify({"status": "Lip Reading started"})


@app.route("/stop_lip_reading", methods=["GET"])
def stop_lip_reading():
    base_dir = os.path.dirname(__file__)
    stop_file = os.path.join(base_dir, "lip reading", "stop.txt")
    try:
        os.makedirs(os.path.dirname(stop_file), exist_ok=True)
        with open(stop_file, "w") as f:
            f.write("stop")
        return jsonify({"status": "stop signal written for lip reading"})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500


@app.route("/start_hand_gestures", methods=["GET"])
def start_hand_gestures():
    t = threading.Thread(target=run_hand_gestures, daemon=True)
    t.start()
    _threads["hand_gestures"] = t
    return jsonify({"status": "Hand Gesture recognition started"})


@app.route("/stop_hand_gestures", methods=["GET"])
def stop_hand_gestures():
    base_dir = os.path.dirname(__file__)
    stop_file = os.path.join(base_dir, "hand_gestures", "stop.txt")
    try:
        os.makedirs(os.path.dirname(stop_file), exist_ok=True)
        with open(stop_file, "w") as f:
            f.write("stop")
        # also try to terminate the subprocess if it was started
        try:
            proc = _threads.get('hand_gestures_proc')
            if proc is not None:
                print(f"[INFO] Attempting to terminate hand_gestures proc pid={getattr(proc, 'pid', None)}")
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        except Exception as e:
            print('[WARN] Failed to terminate hand_gestures subprocess:', e)
        return jsonify({"status": "stop signal written for hand gestures"})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500


@app.route("/latest_result", methods=["GET"])
def get_latest_result():
    # Prefer reading the hand gesture output file (most recent), then lip reading.
    base_dir = os.path.dirname(__file__)
    hg_out = os.path.join(base_dir, "hand_gestures", "output.txt")
    lr_out = os.path.join(base_dir, "lip reading", "output.txt")

    # If hand gesture output exists and has content, return it
    try:
        if os.path.exists(hg_out):
            text = read_output_file(hg_out)
            return jsonify({"type": "hand-gesture", "text": text or "No output"})
    except Exception as e:
        print('[WARN] Error reading hand gesture output:', e)

    # Otherwise check lip reading
    try:
        if os.path.exists(lr_out):
            text = read_output_file(lr_out)
            return jsonify({"type": "lip-reading", "text": text or "No output"})
    except Exception as e:
        print('[WARN] Error reading lip reading output:', e)

    # Fallback to the in-memory latest_result if present
    return jsonify(latest_result)


# --------------------- #
# VIDEO STREAM ROUTE
# --------------------- #
camera = None

def generate_frames():
    global camera
    base_dir = os.path.dirname(__file__)
    frame_file = os.path.join(base_dir, "hand_gestures", "frame.jpg")

    # If a processed frame file exists (written by hand_gestures/main.py), stream it.
    # Otherwise fall back to direct camera capture.
    while True:
        try:
            if os.path.exists(frame_file):
                # Try a few times to read the file to handle transient locks (Windows/OneDrive/AV)
                data = None
                max_attempts = 5
                for attempt in range(max_attempts):
                    try:
                        with open(frame_file, 'rb') as f:
                            data = f.read()
                        break
                    except PermissionError as e:
                        # transient permission error; retry a few times
                        if attempt < max_attempts - 1:
                            time.sleep(0.05)
                            continue
                        else:
                            print("[WARN] Failed to read frame file (permission denied):", e)
                            data = None
                    except Exception as e:
                        # non-permission error, log and stop retrying
                        print("[WARN] Failed to read frame file:", e)
                        data = None
                        break

                if data:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
                else:
                    # couldn't read the on-disk frame; fall back to camera for this iteration
                    time.sleep(0.05)
                    continue
            else:
                # fallback to reading directly from camera
                if camera is None:
                    camera = cv2.VideoCapture(0)
                success, frame = camera.read()
                if not success or frame is None:
                    time.sleep(0.05)
                    continue
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except GeneratorExit:
            # client disconnected
            break
        except Exception as e:
            print("[ERROR] generate_frames error:", e)
            time.sleep(0.1)

@app.route("/video_feed")
def video_feed():
    """Stream webcam feed to frontend"""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed_lip")
def video_feed_lip():
    """Stream the MJPEG frames produced by the lip-reading subprocess."""
    base_dir = os.path.dirname(__file__)
    frame_file = os.path.join(base_dir, "lip reading", "frame.jpg")

    def gen():
        while True:
            try:
                if os.path.exists(frame_file):
                    try:
                        with open(frame_file, "rb") as f:
                            data = f.read()
                        yield (
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n'
                        )
                    except Exception:
                        time.sleep(0.05)
                        continue
                else:
                    time.sleep(0.05)
            except GeneratorExit:
                break
            except Exception:
                time.sleep(0.05)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# --------------------- #
# START FLASK APP
# --------------------- #
if __name__ == "__main__":
    print("Starting backend on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
