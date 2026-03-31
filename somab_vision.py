#!/usr/bin/env python3
# somab_vision.py — Webcam vision module for Somab
# Runs as a standalone process. Writes gaze/face/emotion state to flat IPC files.
#
# Requires: face_landmarker.task in MODEL_PATH
# Download:
#   wget -O /home/amosh/face_landmarker.task \
#     https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
#
# IPC outputs:
#   somab_gaze.txt  →  six lines:
#                       looking:<0|1>
#                       face_detected:<0|1>
#                       face_name:<name|unknown>
#                       face_confidence:<0.0–1.0>
#                       emotion:<neutral|happy|stressed|tired>
#                       emotion_confidence:<0.0–1.0>
#
# IPC inputs:
#   somab_vision_cmd.txt  →  enroll:<n> | forget:<n> | list
#
# Architecture:
#   Main loop  — runs at TARGET_FPS, handles gaze + emotion (fast, landmark-based)
#   RecognitionThread — runs CNN face recognition in background, never blocks main loop
#                       CNN takes 8-12s on CPU; results trickle in asynchronously

import os
import time
import json
import logging
import threading
import queue
import cv2
import mediapipe as mp
import numpy as np
import face_recognition
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX            = 0
FRAME_WIDTH             = 640
FRAME_HEIGHT            = 480
TARGET_FPS              = 10

MODEL_PATH              = "/home/amosh/face_landmarker.task"
GAZE_FILE               = "/home/amosh/somab_gaze.txt"
VISION_CMD_FILE         = "/home/amosh/somab_vision_cmd.txt"
VISION_RESULT_FILE      = "/home/amosh/somab_vision_result.txt"
FACE_ROSTER_FILE        = "/home/amosh/somab_face_roster.json"
STATE_FILE_VISION = "/home/amosh/somab_state.txt"


# Gaze thresholds
YAW_THRESHOLD           = 25.0
IRIS_THRESHOLD          = 0.12
SMOOTHING_FRAMES        = 3

# Face recognition — CNN runs in background thread, no interval needed
RECOGNITION_THRESHOLD   = 0.55
ENROLLMENTS_PER_PERSON  = 5

# Emotion detection
EMOTION_WINDOW          = 6
HAPPY_MOUTH_NEUTRAL     = -0.010  # calibrated: neutral sits at ~-0.030
STRESS_BROW_NEUTRAL     = 0.213
STRESS_BROW_DROP        = 0.010
TIRED_EYE_THRESHOLD     = 0.025

# Landmark indices
LEFT_IRIS_IDX   = 468
RIGHT_IRIS_IDX  = 473
NOSE_TIP_IDX    = 1
LEFT_CHEEK_IDX  = 234
RIGHT_CHEEK_IDX = 454
L_EYE_OUTER     = 33
L_EYE_INNER     = 133
R_EYE_INNER     = 362
R_EYE_OUTER     = 263

# Emotion landmarks
MOUTH_L         = 61
MOUTH_R         = 291
MOUTH_TOP       = 13
MOUTH_BOT       = 14
BROW_L_INNER    = 107
BROW_R_INNER    = 336
L_EYE_TOP       = 159
L_EYE_BOT       = 145
R_EYE_TOP       = 386
R_EYE_BOT       = 374
FOREHEAD        = 10
CHIN            = 152

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [vision] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vision")

# ── IPC ───────────────────────────────────────────────────────────────────────
def write_gaze(looking: bool, face_detected: bool,
               face_name: str = "unknown", face_confidence: float = 0.0,
               emotion: str = "neutral", emotion_confidence: float = 0.0):
    try:
        with open(GAZE_FILE, "w") as f:
            f.write(f"looking:{1 if looking else 0}\n")
            f.write(f"face_detected:{1 if face_detected else 0}\n")
            f.write(f"face_name:{face_name}\n")
            f.write(f"face_confidence:{face_confidence:.3f}\n")
            f.write(f"emotion:{emotion}\n")
            f.write(f"emotion_confidence:{emotion_confidence:.3f}\n")
    except Exception as e:
        log.warning(f"Failed to write gaze file: {e}")

def read_cmd() -> str | None:
    try:
        if not os.path.exists(VISION_CMD_FILE):
            return None
        with open(VISION_CMD_FILE, "r") as f:
            cmd = f.read().strip()
        os.remove(VISION_CMD_FILE)
        return cmd if cmd else None
    except Exception:
        return None

def write_result(text: str):
    try:
        with open(VISION_RESULT_FILE, "w") as f:
            f.write(text)
    except Exception as e:
        log.warning(f"Failed to write result file: {e}")

# ── Gaze math ─────────────────────────────────────────────────────────────────
def estimate_yaw(lm):
    nose_x = lm[NOSE_TIP_IDX].x
    l_x    = lm[LEFT_CHEEK_IDX].x
    r_x    = lm[RIGHT_CHEEK_IDX].x
    face_w = r_x - l_x
    if face_w < 0.001:
        return 0.0
    mid    = (l_x + r_x) / 2.0
    offset = (nose_x - mid) / (face_w / 2.0)
    return offset * 45.0

def estimate_iris_deviation(lm):
    def dev(iris_idx, outer_idx, inner_idx):
        eye_w = abs(lm[inner_idx].x - lm[outer_idx].x)
        if eye_w < 0.001:
            return 0.0
        mid = (lm[outer_idx].x + lm[inner_idx].x) / 2.0
        return abs(lm[iris_idx].x - mid) / eye_w
    left  = dev(LEFT_IRIS_IDX,  L_EYE_OUTER, L_EYE_INNER)
    right = dev(RIGHT_IRIS_IDX, R_EYE_INNER, R_EYE_OUTER)
    return (left + right) / 2.0

# ── Emotion math ──────────────────────────────────────────────────────────────
def estimate_emotion_scores(lm) -> dict[str, float]:
    face_h = abs(lm[CHIN].y - lm[FOREHEAD].y)
    face_w = abs(lm[RIGHT_CHEEK_IDX].x - lm[LEFT_CHEEK_IDX].x)
    if face_h < 0.001 or face_w < 0.001:
        return {"neutral": 1.0, "happy": 0.0, "stressed": 0.0, "tired": 0.0}

    # Happy: mouth corners above calibrated neutral baseline
    mouth_centre_y = (lm[MOUTH_TOP].y + lm[MOUTH_BOT].y) / 2.0
    smile_score    = ((mouth_centre_y - lm[MOUTH_L].y) + (mouth_centre_y - lm[MOUTH_R].y)) / 2.0 / face_h
    happy_conf     = max(0.0, min(1.0, (smile_score - HAPPY_MOUTH_NEUTRAL) / 0.015))

    # Stressed: brow gap drops below calibrated neutral
    brow_gap    = abs(lm[BROW_R_INNER].x - lm[BROW_L_INNER].x) / face_w
    brow_drop   = STRESS_BROW_NEUTRAL - brow_gap
    stress_conf = max(0.0, min(1.0, brow_drop / STRESS_BROW_DROP))

    # Tired: eye openness drops below threshold
    l_eye_open   = abs(lm[L_EYE_TOP].y - lm[L_EYE_BOT].y) / face_h
    r_eye_open   = abs(lm[R_EYE_TOP].y - lm[R_EYE_BOT].y) / face_h
    avg_eye_open = (l_eye_open + r_eye_open) / 2.0
    tired_conf   = max(0.0, min(1.0, 1.0 - (avg_eye_open / TIRED_EYE_THRESHOLD)))

    neutral_conf = max(0.0, 1.0 - max(happy_conf, stress_conf, tired_conf))

    return {
        "happy":    happy_conf,
        "stressed": stress_conf,
        "tired":    tired_conf,
        "neutral":  neutral_conf,
    }

# ── Smoothing ─────────────────────────────────────────────────────────────────
class GazeSmoothing:
    def __init__(self, window=SMOOTHING_FRAMES):
        self.window  = window
        self.history = []

    def update(self, raw: bool) -> bool:
        self.history.append(raw)
        if len(self.history) > self.window:
            self.history.pop(0)
        return self.history.count(True) > self.window // 2


class EmotionSmoothing:
    def __init__(self, window=EMOTION_WINDOW):
        self.window  = window
        self.history = deque(maxlen=window)

    def update(self, scores: dict[str, float]) -> tuple[str, float]:
        self.history.append(scores)
        if not self.history:
            return "neutral", 0.0
        avg = {}
        for emotion in ("happy", "stressed", "tired", "neutral"):
            avg[emotion] = sum(s.get(emotion, 0.0) for s in self.history) / len(self.history)
        best_emotion = max(avg, key=avg.get)
        return best_emotion, avg[best_emotion]

# ── Face roster ───────────────────────────────────────────────────────────────
class FaceRoster:
    def __init__(self):
        self.roster: dict[str, list[list[float]]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.exists(FACE_ROSTER_FILE):
            try:
                with open(FACE_ROSTER_FILE, "r") as f:
                    self.roster = json.load(f)
                log.info(f"Face roster loaded: {list(self.roster.keys())}")
            except Exception as e:
                log.warning(f"Could not load face roster: {e}")
                self.roster = {}
        else:
            log.info("No face roster found — starting fresh")

    def _save(self):
        try:
            with open(FACE_ROSTER_FILE, "w") as f:
                json.dump(self.roster, f)
        except Exception as e:
            log.warning(f"Could not save face roster: {e}")

    def enroll(self, name: str, frame_rgb: np.ndarray) -> str:
        locations = face_recognition.face_locations(frame_rgb, model="cnn")
        if not locations:
            scale     = 2
            big       = cv2.resize(frame_rgb, (frame_rgb.shape[1] * scale, frame_rgb.shape[0] * scale))
            locations = face_recognition.face_locations(big, model="hog", number_of_times_to_upsample=2)
            locations = [(t//scale, r//scale, b//scale, l//scale) for t, r, b, l in locations]
        if not locations:
            log.warning(f"Enroll failed for '{name}' — no face detected")
            return "error:no_face_detected"
        encodings = face_recognition.face_encodings(frame_rgb, locations)
        if not encodings:
            return "error:encoding_failed"
        embedding = encodings[0].tolist()
        with self._lock:
            if name not in self.roster:
                self.roster[name] = []
            self.roster[name].append(embedding)
            if len(self.roster[name]) > ENROLLMENTS_PER_PERSON:
                self.roster[name] = self.roster[name][-ENROLLMENTS_PER_PERSON:]
            self._save()
            count = len(self.roster[name])
        log.info(f"Enrolled '{name}' — now has {count} embedding(s)")
        return f"ok:{name}:{count}"

    def forget(self, name: str) -> str:
        with self._lock:
            if name in self.roster:
                del self.roster[name]
                self._save()
                log.info(f"Forgot '{name}'")
                return f"ok:forgot:{name}"
        return f"error:not_found:{name}"

    def list_names(self) -> str:
        with self._lock:
            if not self.roster:
                return "empty"
            return ",".join(f"{n}({len(e)})" for n, e in self.roster.items())

    def identify(self, frame_rgb: np.ndarray) -> tuple[str, float]:
        with self._lock:
            if not self.roster:
                return "unknown", 0.0
            roster_copy = {n: list(e) for n, e in self.roster.items()}

        locations = face_recognition.face_locations(frame_rgb, model="cnn")
        if not locations:
            scale     = 2
            big       = cv2.resize(frame_rgb, (frame_rgb.shape[1] * scale, frame_rgb.shape[0] * scale))
            locations = face_recognition.face_locations(big, model="hog", number_of_times_to_upsample=2)
            locations = [(t//scale, r//scale, b//scale, l//scale) for t, r, b, l in locations]
        if not locations:
            return "unknown", 0.0

        encodings = face_recognition.face_encodings(frame_rgb, locations)
        if not encodings:
            return "unknown", 0.0

        query     = np.array(encodings[0])
        best_name = "unknown"
        best_dist = 1.0

        for name, embeddings in roster_copy.items():
            for stored in embeddings:
                stored_vec = np.array(stored)
                norm = np.linalg.norm(query) * np.linalg.norm(stored_vec)
                if norm < 1e-8:
                    continue
                cos_sim  = np.dot(query, stored_vec) / norm
                distance = 1.0 - cos_sim
                if distance < best_dist:
                    best_dist = distance
                    best_name = name

        confidence = max(0.0, min(1.0, 1.0 - best_dist))
        if confidence < RECOGNITION_THRESHOLD:
            return "unknown", confidence
        return best_name, confidence

# ── Recognition thread ────────────────────────────────────────────────────────
class RecognitionThread:
    """
    Runs CNN face recognition in a background thread so it never blocks
    the main gaze/emotion loop. The main loop drops frames into a queue;
    this thread processes them one at a time and writes results back via
    a lock-protected result tuple.
    """
    def __init__(self, roster: FaceRoster):
        self.roster      = roster
        self._queue      = queue.Queue(maxsize=1)   # only keep latest frame
        self._result     = ("unknown", 0.0)
        self._result_lock = threading.Lock()
        self._enroll_queue = queue.Queue()
        self._thread     = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("Recognition thread started")

    def submit_frame(self, frame_rgb: np.ndarray):
        """Drop a frame for recognition. If queue is full, discard oldest."""
        try:
            self._queue.put_nowait(frame_rgb.copy())
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame_rgb.copy())
            except queue.Full:
                pass

    def submit_enroll(self, name: str, frame_rgb: np.ndarray):
        self._enroll_queue.put((name, frame_rgb.copy()))

    def get_result(self) -> tuple[str, float]:
        with self._result_lock:
            return self._result

    def _run(self):
        # Deprioritize this thread so CNN doesn't starve gaze/emotion loop
        os.nice(15)

        while True:
            # Handle enroll requests first (higher priority)
            try:
                name, frame_rgb = self._enroll_queue.get_nowait()
                result = self.roster.enroll(name, frame_rgb)
                write_result(result)
                log.info(f"Enroll complete: {result}")
                continue
            except queue.Empty:
                pass

            # Process recognition frame
            try:
                frame_rgb = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                name, conf = self.roster.identify(frame_rgb)
                with self._result_lock:
                    self._result = (name, conf)
                if name != "unknown":
                    log.debug(f"Recognised: {name} ({conf:.2f})")
            except Exception as e:
                log.warning(f"Recognition error: {e}")
            # Yield to main loop between passes
            time.sleep(0.1)

# ── Main loop ─────────────────────────────────────────────────────────────────
def run():
    log.info("Starting vision module")

    base_opts  = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options    = mp_vision.FaceLandmarkerOptions(
        base_options                          = base_opts,
        running_mode                          = mp_vision.RunningMode.IMAGE,
        num_faces                             = 1,
        min_face_detection_confidence         = 0.5,
        min_face_presence_confidence          = 0.5,
        min_tracking_confidence               = 0.5,
        output_face_blendshapes               = False,
        output_facial_transformation_matrixes = False,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    log.info("FaceLandmarker loaded")

    roster          = FaceRoster()
    recognizer      = RecognitionThread(roster)
    gaze_smoother   = GazeSmoothing()
    emotion_smoother = EmotionSmoothing()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    if not cap.isOpened():
        log.error(f"Could not open camera {CAMERA_INDEX}")
        write_gaze(False, False)
        return

    log.info(f"Camera {CAMERA_INDEX} opened — {FRAME_WIDTH}x{FRAME_HEIGHT} @ {TARGET_FPS}fps")

    frame_interval       = 1.0 / TARGET_FPS
    last_frame_t         = 0.0
    pending_enroll_name  = None

    write_gaze(False, False)

    try:
        while True:
            now = time.time()
            if now - last_frame_t < frame_interval:
                time.sleep(0.005)
                continue
            last_frame_t = now

            # ── Command check ─────────────────────────────────────────────────
            cmd = read_cmd()
            if cmd:
                if cmd.startswith("enroll:"):
                    pending_enroll_name = cmd.split(":", 1)[1].strip()
                    log.info(f"Enroll pending for '{pending_enroll_name}' — waiting for frame")
                elif cmd.startswith("forget:"):
                    write_result(roster.forget(cmd.split(":", 1)[1].strip()))
                elif cmd == "list":
                    write_result(roster.list_names())

            # ── Sleep mode: skip all processing ──────────────────────────────
            try:
                with open(STATE_FILE_VISION, "r") as f:
                    current_state = f.read().strip()
            except Exception:
                current_state = ""

            if current_state == "sleeping":
                # Drain the camera buffer so we don't get a stale frame on wake
                cap.grab()
                time.sleep(0.5)
                continue

            # ── Frame capture ─────────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                log.warning("Frame read failed — retrying")
                time.sleep(0.1)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Enroll: send to recognition thread ────────────────────────────
            if pending_enroll_name:
                recognizer.submit_enroll(pending_enroll_name, rgb)
                pending_enroll_name = None

            # ── Submit frame for background recognition ───────────────────────
            recognizer.submit_frame(rgb)

            # ── Gaze detection (runs every frame, fast) ───────────────────────
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mp_result = landmarker.detect(mp_image)

            if not mp_result.face_landmarks:
                smoothed = gaze_smoother.update(False)
                current_name, current_conf = recognizer.get_result()
                write_gaze(smoothed, False, "unknown", 0.0, "neutral", 0.0)
                continue

            lm       = mp_result.face_landmarks[0]
            yaw      = estimate_yaw(lm)
            iris_dev = estimate_iris_deviation(lm)
            smoothed = gaze_smoother.update(abs(yaw) < YAW_THRESHOLD and iris_dev < IRIS_THRESHOLD)

            # ── Emotion (every frame, landmark-based = zero cost) ─────────────
            raw_scores                         = estimate_emotion_scores(lm)
            current_emotion, current_emotion_conf = emotion_smoother.update(raw_scores)

            # ── Get latest recognition result (non-blocking) ──────────────────
            current_name, current_conf = recognizer.get_result()

            write_gaze(smoothed, True, current_name, current_conf,
                       current_emotion, current_emotion_conf)
            log.debug(
                f"yaw={yaw:+.1f}° iris={iris_dev:.3f} looking={smoothed} "
                f"face={current_name}({current_conf:.2f}) "
                f"emotion={current_emotion}({current_emotion_conf:.2f})"
            )

    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        cap.release()
        landmarker.close()
        write_gaze(False, False)
        log.info("Camera released")


if __name__ == "__main__":
    run()