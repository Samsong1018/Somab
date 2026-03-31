import warnings
import os
import logging

# Suppress noisy warnings at startup
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

# Suppress ALSA/Jack stderr noise
import ctypes
try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(ctypes.c_void_p(None))
except Exception:
    pass

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="/home/amosh/somab.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True
)
log = logging.getLogger("somab")

# ── Imports ───────────────────────────────────────────────────────────────────
import json
import pickle
import random
import re
import smtplib
import subprocess
import sys
import threading
import time
import wave

import anthropic
import numpy as np
import pyaudio
import requests
import spotipy
import torch
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from ddgs import DDGS
from dotenv import load_dotenv
from email.mime.text import MIMEText
from faster_whisper import WhisperModel
from geopy.geocoders import Nominatim
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openwakeword.model import Model as WakeWordModel
from phonemizer import phonemize
from playwright.sync_api import sync_playwright
from resemblyzer import VoiceEncoder, preprocess_wav
from spotipy.oauth2 import SpotifyOAuth
from piper import PiperVoice

import somab_face

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv("/home/amosh/.somab.env")

ANTHROPIC_API_KEY     = os.environ.get("ANTHROPIC_API_KEY", "")
DEV_PASSWORD          = os.environ.get("DEV_PASSWORD", "")
TODO_EMAIL            = os.environ.get("TODO_EMAIL", "")
TODO_PASSWORD         = os.environ.get("TODO_PASSWORD", "")
SMTP_EMAIL            = os.environ.get("SMTP_EMAIL", "")
SMTP_PASSWORD         = os.environ.get("SMTP_PASSWORD", "")
PHONE_SMS_EMAIL       = os.environ.get("PHONE_SMS_EMAIL", "")
SPOTIFY_CLIENT_ID     = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI  = os.environ.get("SPOTIFY_REDIRECT_URI", "")
TELEGRAM_BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID      = os.getenv("TELEGRAM_CHAT_ID")
SPOTIFY_CACHE_PATH    = "/home/amosh/.spotify_cache"

TODO_BASE_URL       = "https://dailytodo-api.onrender.com"
VOICE_ROSTER_FILE   = "/home/amosh/somab_voices.json"
VOICE_THRESHOLD     = 0.82
PIPER_MODEL         = "/home/amosh/piper-voices/en_US-lessac-medium.onnx"
PIPER_CONFIG        = "/home/amosh/piper-voices/en_US-lessac-medium.onnx.json"
WAKEWORD_MODEL      = "/home/amosh/somab/lib/python3.12/site-packages/openwakeword/resources/models/hey_so_mab.onnx"
NOTES_FILE          = "/home/amosh/somab_notes.txt"
CREDENTIALS_FILE    = "/home/amosh/somab_credentials.json"
TOKEN_FILE          = "/home/amosh/somab_token.pickle"
STATE_FILE          = "/home/amosh/somab_state.txt"
VISEME_FILE         = "/home/amosh/somab_viseme.txt"
INFO_FILE           = "/home/amosh/somab_info.txt"
MEMORY_FILE         = "/home/amosh/somab_memory.json"
DEVMODE_INPUT_FILE  = "/home/amosh/somab_devmode_input.txt"
DEVMODE_RESULT_FILE = "/home/amosh/somab_devmode_result.txt"
GAZE_FILE           = "/home/amosh/somab_gaze.txt"
VISION_CMD_FILE     = "/home/amosh/somab_vision_cmd.txt"
VISION_RESULT_FILE  = "/home/amosh/somab_vision_result.txt"
# Lowered wake word threshold when gaze confirms user is looking at Somab
WAKE_WORD_THRESHOLD_GAZE = 0.45
WAKE_WORD_THRESHOLD = 0.7
COOLDOWN_SECONDS    = 2
MAX_MEMORY_TURNS    = 20
CALENDAR_SCOPES     = ["https://www.googleapis.com/auth/calendar"]

todo_token = None

# ── Launch vision process ─────────────────────────────────────────────────────
vision_proc = subprocess.Popen(["/home/amosh/somab/bin/python3", "/home/amosh/somab_vision.py"])

# ── Gaze IPC ──────────────────────────────────────────────────────────────────
def read_gaze() -> dict:
    result = {"looking": False, "face_detected": False,
              "face_name": "unknown", "face_confidence": 0.0,
              "emotion": "neutral", "emotion_confidence": 0.0}
    try:
        with open(GAZE_FILE, "r") as f:
            for line in f:
                k, _, v = line.strip().partition(":")
                if k == "looking":            result["looking"]           = v == "1"
                elif k == "face_detected":    result["face_detected"]     = v == "1"
                elif k == "face_name":        result["face_name"]         = v
                elif k == "face_confidence":
                    try: result["face_confidence"] = float(v)
                    except ValueError: pass
                elif k == "emotion":          result["emotion"]           = v
                elif k == "emotion_confidence":
                    try: result["emotion_confidence"] = float(v)
                    except ValueError: pass
    except Exception:
        pass
    return result

# ── Emotion monitor ───────────────────────────────────────────────────────────
EMOTION_SUSTAIN_SECONDS  = 20    # emotion must persist this long before commenting
EMOTION_COOLDOWN_SECONDS = 600   # 10 min between comments on same emotion

_emotion_start_times: dict[str, float] = {}
_emotion_last_comment: dict[str, float] = {}

def emotion_monitor():
    """
    Background thread. Watches the gaze file for sustained non-neutral emotions
    on a known face and triggers a proactive spoken comment from Somab.
    """
    global last_interaction
    while True:
        time.sleep(2.0)
        try:
            # Skip all processing during sleep mode
            try:
                with open(STATE_FILE, "r") as f:
                    if f.read().strip() == "sleeping":
                        _emotion_start_times.clear()
                        continue
            except Exception:
                pass

            gaze = read_gaze()
            # Only fire when we can see a known person
            if not gaze["face_detected"] or gaze["face_name"] == "unknown":
                _emotion_start_times.clear()
                continue

            emotion = gaze["emotion"]
            conf    = gaze["emotion_confidence"]

            # Only care about non-neutral emotions with decent confidence
            if emotion == "neutral" or conf < 0.6:
                _emotion_start_times.pop(emotion, None)
                continue

            now = time.time()

            # Track how long this emotion has been sustained
            if emotion not in _emotion_start_times:
                _emotion_start_times[emotion] = now
                continue

            sustained = now - _emotion_start_times[emotion]
            if sustained < EMOTION_SUSTAIN_SECONDS:
                continue

            # Check cooldown
            last_comment = _emotion_last_comment.get(emotion, 0)
            if now - last_comment < EMOTION_COOLDOWN_SECONDS:
                continue

            # Don't interrupt if Somab is already speaking/thinking
            current_state = ""
            try:
                with open(STATE_FILE, "r") as f:
                    current_state = f.read().strip()
            except Exception:
                pass
            if current_state in ("speaking", "thinking", "listening", "sleeping"):
                continue

            # Fire a proactive comment
            name = gaze["face_name"].capitalize()
            comments = {
                "stressed": [
                    f"Hey {name}, you've been looking pretty stressed. Everything alright?",
                    f"You good? You've had that look for a while.",
                    f"Not to pry, but you look tense. Want to talk through something?",
                ],
                "tired": [
                    f"You look exhausted, {name}. Maybe take a break?",
                    f"Have you slept? Because you look like you haven't.",
                    f"Genuinely concerned — you look dead on your feet.",
                ],
                "happy": [
                    f"You seem like you're in a good mood. What's going on?",
                    f"Whatever you're thinking about, keep it up.",
                ],
            }

            if emotion not in comments:
                continue

            comment = random.choice(comments[emotion])
            log.info(f"Emotion comment triggered: {emotion} sustained {sustained:.0f}s — '{comment}'")
            _emotion_last_comment[emotion] = now
            _emotion_start_times.pop(emotion, None)
            last_interaction = now
            speak(comment)

        except Exception as e:
            log.error(f"Emotion monitor error: {e}")

def get_emotion_context() -> str:
    """
    Returns an emotion context string to inject into the system prompt.
    Returns empty string if emotion is neutral or confidence is too low.
    """
    gaze = read_gaze()
    emotion = gaze["emotion"]
    conf    = gaze["emotion_confidence"]

    if emotion == "neutral" or conf < 0.65:
        return ""

    descriptions = {
        "happy": (
            "Amos appears to be in a good mood right now. "
            "Match his energy a bit — you can be a little more playful than usual. "
            "Only mention it if it's genuinely relevant to what he's saying."
        ),
        "stressed": (
            "Amos looks stressed or tense right now. "
            "Dial back the sarcasm, be more direct and efficient. "
            "If it's natural to acknowledge it without being annoying, do so briefly — "
            "but only if it genuinely fits the conversation."
        ),
        "tired": (
            "Amos looks tired. Keep responses shorter than usual, "
            "skip unnecessary elaboration, and be low-energy. "
            "If it's genuinely relevant you can mention it once, but don't harp on it."
        ),
    }

    return descriptions.get(emotion, "")

threading.Thread(target=emotion_monitor, daemon=True).start() 

def face_vision_roster(action: str, name: str = "") -> str:
    """Send a command to somab_vision.py and wait for the result."""
    if action == "enroll":
        if not name:
            return "error:no_name"
        try:
            with open(VISION_CMD_FILE, "w") as f:
                f.write(f"enroll:{name}")
        except Exception as e:
            return f"error:{e}"
        # Wait up to 3 seconds for result
        for _ in range(30):
            time.sleep(0.1)
            try:
                with open(VISION_RESULT_FILE, "r") as f:
                    result = f.read().strip()
                if result:
                    os.remove(VISION_RESULT_FILE)
                    return result
            except Exception:
                pass
        return "error:timeout"
    elif action == "forget":
        if not name:
            return "error:no_name"
        try:
            with open(VISION_CMD_FILE, "w") as f:
                f.write(f"forget:{name}")
        except Exception as e:
            return f"error:{e}"
        for _ in range(30):
            time.sleep(0.1)
            try:
                with open(VISION_RESULT_FILE, "r") as f:
                    result = f.read().strip()
                if result:
                    os.remove(VISION_RESULT_FILE)
                    return result
            except Exception:
                pass
        return "error:timeout"
    elif action == "list":
        try:
            with open(VISION_CMD_FILE, "w") as f:
                f.write("list")
        except Exception as e:
            return f"error:{e}"
        for _ in range(30):
            time.sleep(0.1)
            try:
                with open(VISION_RESULT_FILE, "r") as f:
                    result = f.read().strip()
                if result:
                    os.remove(VISION_RESULT_FILE)
                    return result
            except Exception:
                pass
        return "error:timeout"
    return "error:unknown_action"

# ── Spotify ───────────────────────────────────────────────────────────────────
spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope="user-modify-playback-state user-read-playback-state user-read-currently-playing",
    cache_path=SPOTIFY_CACHE_PATH,
    open_browser=True
))

# ── Launch face process ───────────────────────────────────────────────────────
subprocess.Popen(["python3", "/home/amosh/somab_face.py"])

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
whisper = WhisperModel("small", device="cpu", compute_type="int8")
wake_word     = WakeWordModel(wakeword_model_paths=[WAKEWORD_MODEL])
tts           = PiperVoice.load(PIPER_MODEL, config_path=PIPER_CONFIG)
claude        = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True
)
voice_encoder = VoiceEncoder()
p             = pyaudio.PyAudio()
print("Somab is ready!")

# ── Memory ────────────────────────────────────────────────────────────────────
def load_memory():
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_memory():
    try:
        serializable = []
        for msg in conversation_history[-(MAX_MEMORY_TURNS * 2):]:
            # Skip tool interaction messages — they cause 400s when reloaded
            if isinstance(msg["content"], list):
                has_tool = any(
                    (isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result"))
                    or (hasattr(b, "type") and b.type in ("tool_use", "tool_result"))
                    for b in msg["content"]
                )
                if has_tool:
                    continue
            if isinstance(msg["content"], str):
                serializable.append(msg)
            elif isinstance(msg["content"], list):
                content = []
                for block in msg["content"]:
                    if hasattr(block, "type") and block.type == "text":
                        content.append({"type": "text", "text": block.text})
                    elif isinstance(block, dict) and block.get("type") == "text":
                        content.append(block)
                if content:
                    serializable.append({"role": msg["role"], "content": content})
        with open(MEMORY_FILE, "w") as f:
            json.dump(serializable, f, indent=2)
    except Exception as e:
        print(f"Failed to save memory: {e}")

conversation_history = load_memory()
active_timers        = []

# ── Voice roster ──────────────────────────────────────────────────────────────
def load_voice_roster():
    try:
        with open(VOICE_ROSTER_FILE, "r") as f:
            data = json.load(f)
            return {name: np.array(emb) for name, emb in data.items()}
    except Exception:
        return {}

def save_voice_roster(roster):
    try:
        with open(VOICE_ROSTER_FILE, "w") as f:
            json.dump({name: emb.tolist() for name, emb in roster.items()}, f, indent=2)
    except Exception as e:
        print(f"Failed to save voice roster: {e}")

voice_roster = load_voice_roster()

def identify_speaker(audio_path):
    if not voice_roster:
        return None
    try:
        wav        = preprocess_wav(audio_path)
        emb        = voice_encoder.embed_utterance(wav)
        best_name  = None
        best_score = 0.0
        for name, stored_emb in voice_roster.items():
            score = np.dot(emb, stored_emb) / (np.linalg.norm(emb) * np.linalg.norm(stored_emb))
            if score > best_score:
                best_score = score
                best_name  = name
        print(f"Voice match: {best_name} ({best_score:.2f})")
        return best_name if best_score >= VOICE_THRESHOLD else None
    except Exception as e:
        print(f"Voice ID failed: {e}")
        return None

def enroll_voice(name, samples=3):
    embeddings = []
    for i in range(samples):
        speak(f"Sample {i + 1} of {samples}. Say something natural for about 10 seconds.")
        set_state("listening")
        audio_path = record_audio(max_duration=10.0)
        try:
            wav = preprocess_wav(audio_path)
            emb = voice_encoder.embed_utterance(wav)
            embeddings.append(emb)
            if i < samples - 1:
                speak("Got it.")
        except Exception as e:
            speak("Had trouble with that sample, let's try again.")
            print(f"Enrollment sample error: {e}")
            continue
    if not embeddings:
        speak("Enrollment failed, try again.")
        return
    averaged = np.mean(embeddings, axis=0)
    averaged = averaged / np.linalg.norm(averaged)
    voice_roster[name] = averaged
    save_voice_roster(voice_roster)
    speak(f"Done. I'll recognize you as {name} from now on.")

def forget_voice(name):
    matches = [n for n in voice_roster if name.lower() in n.lower()]
    if not matches:
        return f"I don't have anyone called {name} in my roster."
    for m in matches:
        del voice_roster[m]
    save_voice_roster(voice_roster)
    return f"Forgot {', '.join(matches)}."

def list_voices():
    if not voice_roster:
        return "I don't know anyone yet."
    return "I know " + ", ".join(voice_roster.keys()) + "."

# ── IPC helpers ───────────────────────────────────────────────────────────────
def set_state(s):
    with open(STATE_FILE, "w") as f:
        f.write(s)

def set_viseme(key):
    with open(VISEME_FILE, "w") as f:
        f.write("none" if key is None else key)

def set_info(data):
    try:
        with open(STATE_FILE, "r") as f:
            if f.read().strip() == "sleeping":
                return
    except Exception:
        pass
    with open(INFO_FILE, "w") as f:
        f.write(data)

def clear_info():
    with open(INFO_FILE, "w") as f:
        f.write("")

set_state("idle")

# ── Tool definitions ──────────────────────────────────────────────────────────
tools = [
    {
        "name": "web_search",
        "description": "Browse the web to find information, articles, or answer questions. Can optionally send results to the user's phone via SMS.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal":          {"type": "string",  "description": "What to search for or find on the web"},
                "send_to_phone": {"type": "boolean", "description": "Whether to send the result to the user's phone via SMS"}
            },
            "required": ["goal"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City or location name"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "set_timer",
        "description": "Set a timer for a specified number of seconds or minutes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "duration_seconds": {"type": "integer", "description": "Timer duration in seconds"},
                "label":            {"type": "string",  "description": "Optional label for the timer"}
            },
            "required": ["duration_seconds"]
        }
    },
    {
        "name": "write_note",
        "description": "Write or append a note to the notes file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The note content to save"}
            },
            "required": ["content"]
        }
    },
    {
        "name": "read_notes",
        "description": "Read all saved notes.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "calculator",
        "description": "Perform mathematical calculations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The mathematical expression to evaluate e.g. '25 * 4 + 10'"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "unit_converter",
        "description": "Convert between units of measurement such as miles to kilometers, fahrenheit to celsius, pounds to kilograms, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "value":     {"type": "number", "description": "The value to convert"},
                "from_unit": {"type": "string", "description": "The unit to convert from"},
                "to_unit":   {"type": "string", "description": "The unit to convert to"}
            },
            "required": ["value", "from_unit", "to_unit"]
        }
    },
    {
        "name": "get_calendar_events",
        "description": "Get upcoming calendar events.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer", "description": "Number of events to fetch, default 5"}
            }
        }
    },
    {
        "name": "create_calendar_event",
        "description": "Create a new event on the user's Google Calendar.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title":            {"type": "string",  "description": "Event title"},
                "date":             {"type": "string",  "description": "Date in YYYY-MM-DD format"},
                "time":             {"type": "string",  "description": "Time in HH:MM 24hr format"},
                "duration_minutes": {"type": "integer", "description": "Duration in minutes, default 60"}
            },
            "required": ["title", "date"]
        }
    },
    {
        "name": "get_todays_todos",
        "description": "Get the user's todo list for today.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "complete_todo",
        "description": "Mark a todo item as complete.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title or partial title of the todo to complete"}
            },
            "required": ["title"]
        }
    },
    {
        "name": "add_todo",
        "description": "Add a new todo item for today.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title of the new todo"}
            },
            "required": ["title"]
        }
    },
    {
        "name": "morning_debrief",
        "description": "Give the user a morning debrief with the current time, weather, and todo list. Also texts the weather and todos to the user's phone.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "set_volume",
        "description": "Set, increase, decrease, or mute the system volume.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string",  "description": "One of: set, increase, decrease, mute, unmute"},
                "amount": {"type": "integer", "description": "Percentage amount for set/increase/decrease (e.g. 50, 10)"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "set_brightness",
        "description": "Set or adjust the screen brightness.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string",  "description": "One of: set, increase, decrease"},
                "amount": {"type": "integer", "description": "Percentage for set (e.g. 80), or step amount for increase/decrease (e.g. 10)"}
            },
            "required": ["action", "amount"]
        }
    },
    {
        "name": "system_power",
        "description": "Shutdown or restart the computer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "One of: shutdown, restart"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "spotify",
        "description": "Control Spotify music playback. Can play, pause, skip, go back, set volume, or search and play a specific song, artist, or playlist. For genre or mood requests like 'play rock' or 'play something chill', search for a playlist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action":      {"type": "string",  "description": "One of: play, pause, skip, previous, volume, search_and_play"},
                "query":       {"type": "string",  "description": "Song or artist name for search_and_play"},
                "amount":      {"type": "integer", "description": "Volume level 0-100 for volume action"},
                "device_name": {"type": "string",  "description": "Name or partial name of the Spotify device to play on, e.g. 'laptop', 'phone', 'speaker'"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "voice_roster",
        "description": "Manage the voice recognition roster. Can enroll a new person, forget someone, or list known people.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "One of: enroll, forget, list"},
                "name":   {"type": "string", "description": "The person's name for enroll or forget"}
            },
            "required": ["action"]
        }
    },
    {
    "name": "set_face_color",
        "description": "Change the color of Somab's face. Accepts any color name or hex code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "color": {"type": "string", "description": "Color name like 'red', 'pink', 'blue', or hex like '#FF0000'"}
            },
            "required": ["color"]
        }
    },
    {
        "name": "face_vision_roster",
        "description": "Manage the face recognition roster. Enroll a new person's face using the webcam, forget someone, or list known faces. For enroll, the person must be looking at the camera.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "One of: enroll, forget, list"},
                "name":   {"type": "string", "description": "The person's name for enroll or forget"}
            },
            "required": ["action"]
        }
    },
]

# ── Sleep monitor ─────────────────────────────────────────────────────────────
IDLE_SLEEP_SECONDS = 3600  # 1 hour
last_interaction   = time.time()

def sleep_monitor():
    sleep_announced = False
    while True:
        time.sleep(60)
        idle_duration = time.time() - last_interaction
        if idle_duration > IDLE_SLEEP_SECONDS:
            current = ""
            try:
                with open(STATE_FILE, "r") as f:
                    current = f.read().strip()
            except Exception:
                pass
            if current in ("idle", "sleeping") and not sleep_announced:
                sleep_announced = True
                set_state("sleeping")
                speak(random.choice([
                    "I'll be here if you need me.",
                    "Going to rest for a bit.",
                    "Wake me up if you need anything.",
                ]))
        else:
            sleep_announced = False

threading.Thread(target=sleep_monitor, daemon=True).start()

# ── Tool implementations ──────────────────────────────────────────────────────

NAMED_COLORS = {
    "red":     (255, 80,  80),
    "pink":    (255, 105, 180),
    "blue":    (80,  140, 255),
    "green":   (80,  220, 120),
    "yellow":  (255, 220, 60),
    "orange":  (255, 160, 50),
    "purple":  (180, 80,  255),
    "white":   (240, 240, 240),
    "teal":    (77,  217, 224),
    "cyan":    (0,   230, 230),
    "magenta": (255, 60,  200),
    "gold":    (255, 200, 0),
}

def set_face_color(color):
    try:
        color = color.strip().lower()
        if color in NAMED_COLORS:
            r, g, b = NAMED_COLORS[color]
        elif color.startswith("#"):
            color = color.lstrip("#")
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
        else:
            return f"I don't know the color '{color}'. Try a name like red, pink, blue, or a hex code."
        with open("/home/amosh/somab_color.txt", "w") as f:
            f.write(f"{r},{g},{b}")
        return f"Face color changed to {color}."
    except Exception as e:
        return f"Failed to set color: {str(e)}"

def spotify_control(action, query=None, amount=None, device_name=None):
    try:
        devices   = spotify.devices()["devices"]
        device_id = None
        if device_name and devices:
            for d in devices:
                if device_name.lower() in d["name"].lower():
                    device_id = d["id"]
                    break
        if not device_id and devices:
            active    = [d for d in devices if d["is_active"]]
            device_id = active[0]["id"] if active else devices[0]["id"]
        if not device_id:
            if devices:
                names = ", ".join(d["name"] for d in devices)
                return f"No active device. Available devices: {names}. Say which one to use."
            return "No Spotify devices found. Open Spotify on a device first."

        if action == "play":
            spotify.start_playback(device_id=device_id)
            return "Resuming playback."
        elif action == "pause":
            spotify.pause_playback(device_id=device_id)
            return "Paused."
        elif action == "skip":
            spotify.next_track(device_id=device_id)
            return "Skipped to next track."
        elif action == "previous":
            spotify.previous_track(device_id=device_id)
            return "Going back."
        elif action == "volume" and amount is not None:
            spotify.volume(amount, device_id=device_id)
            return f"Spotify volume set to {amount}%."
        elif action == "search_and_play" and query:
            # Try playlist first for genre/mood queries, track for specific song requests
            results = spotify.search(q=query, type="playlist", limit=1)
            playlists = results["playlists"]["items"]
    
            if playlists:
                playlist = playlists[0]
                spotify.start_playback(device_id=device_id, context_uri=playlist["uri"])
                device_label = next((d["name"] for d in devices if d["id"] == device_id), "")
                return f"Playing playlist '{playlist['name']}' on {device_label}."
            else:
                # Fall back to single track
                results = spotify.search(q=query, type="track", limit=1)
                tracks  = results["tracks"]["items"]
                if not tracks:
                    return f"Couldn't find anything for '{query}'."
                track        = tracks[0]
                device_label = next((d["name"] for d in devices if d["id"] == device_id), "")
                spotify.start_playback(device_id=device_id, uris=[track["uri"]])
                return f"Playing {track['name']} by {track['artists'][0]['name']} on {device_label}."
        return "Unknown Spotify action."
    except Exception as e:
        return f"Spotify control failed: {str(e)}"

todo_login_attempted = False

def todo_login():
    global todo_token
    try:
        r = requests.post(f"{TODO_BASE_URL}/api/login", json={
            "email": TODO_EMAIL,
            "password": TODO_PASSWORD
        }, timeout=10)
        todo_token = r.json()["data"]["token"]
    except Exception as e:
        print(f"Todo login failed: {e}")

def todo_headers():
    return {"Authorization": f"Bearer {todo_token}"}

def todo_ensure_auth():
    global todo_token, todo_login_attempted
    if not todo_token and not todo_login_attempted:
        todo_login_attempted = True
        todo_login()

def get_todo_data():
    todo_ensure_auth()
    try:
        r = requests.get(f"{TODO_BASE_URL}/api/user", headers=todo_headers(), timeout=10)
        if r.status_code == 401:
            todo_login()
            r = requests.get(f"{TODO_BASE_URL}/api/user", headers=todo_headers(), timeout=10)
        return r.json()
    except Exception as e:
        return f"Failed to get todos: {str(e)}"

def save_todo_data(data):
    todo_ensure_auth()
    try:
        r = requests.put(f"{TODO_BASE_URL}/api/user", headers=todo_headers(), json=data, timeout=10)
        if r.status_code == 401:
            todo_login()
            r = requests.put(f"{TODO_BASE_URL}/api/user", headers=todo_headers(), json=data, timeout=10)
        return r.status_code == 200
    except Exception as e:
        return False

def get_todays_todos():
    data = get_todo_data()
    if isinstance(data, str):
        return data
    today      = datetime.now().strftime("%Y-%m-%d")
    todos      = data.get("todos", {})
    recurring  = data.get("recurring", [])
    rec_state  = data.get("recurringState", {})
    today_todos = [t for t in todos.get(today, []) if not t.get("done")]

    def recurs_today(task):
        start = task.get("startDate") or task.get("start_date")
        if not start or start > today:
            return False
        freq = task.get("frequency")
        if freq == "daily":
            return True
        if freq == "weekly":
            return datetime.strptime(start, "%Y-%m-%d").weekday() == datetime.now().weekday()
        if freq == "monthly":
            return start.split("-")[2] == today.split("-")[2]
        return False

    today_recurring = []
    for task in recurring:
        if not recurs_today(task):
            continue
        state = rec_state.get(today, {}).get(task["id"], {})
        if not state.get("done") and not state.get("dismissed"):
            today_recurring.append(task)

    all_todos = today_todos + today_recurring
    if not all_todos:
        return "No todos for today."
    return "\n".join(f"- {t.get('text', t.get('title', 'Unknown'))}" for t in all_todos)

def complete_todo(title):
    data = get_todo_data()
    if isinstance(data, str):
        return data
    today      = datetime.now().strftime("%Y-%m-%d")
    today_list = data.get("todos", {}).get(today, [])
    for todo in today_list:
        text = todo.get("text", todo.get("title", ""))
        if title.lower() in text.lower():
            todo["done"] = True
            data["todos"][today] = today_list
            save_todo_data(data)
            return f"Marked '{text}' as done."
    rec_state = data.get("recurringState", {})
    for task in data.get("recurring", []):
        text = task.get("text", task.get("title", ""))
        if title.lower() in text.lower():
            if today not in rec_state:
                rec_state[today] = {}
            rec_state[today][task["id"]] = {"done": True}
            data["recurringState"] = rec_state
            save_todo_data(data)
            return f"Marked recurring task '{text}' as done."
    return f"Couldn't find a todo matching '{title}'."

def add_todo(title):
    data = get_todo_data()
    if isinstance(data, str):
        return data
    today = datetime.now().strftime("%Y-%m-%d")
    todos = data.get("todos", {})
    if today not in todos:
        todos[today] = []
    todos[today].append({
        "id":   str(int(time.time() * 1000)),
        "text": title,
        "done": False
    })
    data["todos"] = todos
    save_todo_data(data)
    return f"Added '{title}' to today's todos."

def set_volume(action, amount=10):
    try:
        if action == "set":
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{amount}%"])
            return f"Volume set to {amount}%."
        elif action == "increase":
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"+{amount}%"])
            return f"Volume increased by {amount}%."
        elif action == "decrease":
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"-{amount}%"])
            return f"Volume decreased by {amount}%."
        elif action == "mute":
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"])
            return "Volume muted."
        elif action == "unmute":
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"])
            return "Volume unmuted."
        return "Unknown volume action."
    except Exception as e:
        return f"Volume control failed: {str(e)}"

def set_brightness(action, amount=10):
    try:
        if action == "set":
            subprocess.run(["brightnessctl", "set", f"{amount}%"])
            return f"Brightness set to {amount}%."
        elif action == "increase":
            subprocess.run(["brightnessctl", "set", f"+{amount}%"])
            return f"Brightness increased by {amount}%."
        elif action == "decrease":
            subprocess.run(["brightnessctl", "set", f"{amount}%-"])
            return f"Brightness decreased by {amount}%."
        return "Unknown brightness action."
    except Exception as e:
        return f"Brightness control failed: {str(e)}"

def system_power(action):
    try:
        if action == "shutdown":
            speak("Shutting down. Goodbye!")
            time.sleep(2)
            subprocess.run(["systemctl", "poweroff"])
            return "Shutting down."
        elif action == "restart":
            speak("Restarting now.")
            time.sleep(2)
            subprocess.run(["systemctl", "reboot"])
            return "Restarting."
        return "Unknown power action."
    except Exception as e:
        return f"Power control failed: {str(e)}"

def enter_dev_mode():
    set_state("devmode")
    speak("Entering dev mode. Please enter your password on screen.")
    open(DEVMODE_INPUT_FILE, "w").close()
    open(DEVMODE_RESULT_FILE, "w").close()
    while True:
        time.sleep(0.1)
        try:
            with open(DEVMODE_INPUT_FILE, "r") as f:
                pw = f.read().strip()
            if pw:
                open(DEVMODE_INPUT_FILE, "w").close()
                if pw == DEV_PASSWORD:
                    speak("Access granted. Goodbye.")
                    set_state("idle")
                    time.sleep(1)
                    subprocess.Popen(["/home/amosh/somab-exit-to-desktop.sh"])
                    sys.exit(0)
                else:
                    speak("Incorrect password.")
                    with open(DEVMODE_RESULT_FILE, "w") as f:
                        f.write("error")
            with open(DEVMODE_RESULT_FILE, "r") as f:
                result = f.read().strip()
            if result == "cancelled":
                speak("Dev mode cancelled.")
                set_state("idle")
                return
        except Exception:
            pass

def build_debrief_message(weekday, time_str, weather, todos, sp500, technews):
    return (
        f"🌅 <b>Morning Debrief — {weekday}, {time_str}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🌤 <b>Weather</b>\n<i>{weather}</i>\n\n"
        f"📈 <b>S&P 500</b>\n<i>{sp500}</i>\n\n"
        f"🤖 <b>Tech & AI News</b>\n<i>{technews}</i>\n\n"
        f"✅ <b>Today's Todos</b>\n<i>{todos}</i>"
    )

def morning_debrief():
    now      = datetime.now()
    time_str = now.strftime("%I:%M %p")
    weekday  = now.strftime("%A")

    results = [None, None, None, None]
    def _weather():  results[0] = get_weather("Redmond")
    def _todos():    results[1] = get_todays_todos()
    def _sp500():    results[2] = quick_search("S&P 500 current price today")
    def _news():     results[3] = quick_search("top AI and tech news today")

    threads = [threading.Thread(target=fn) for fn in (_weather, _todos, _sp500, _news)]
    for t in threads: t.start()
    for t in threads: t.join()

    weather, todos, sp500, technews = results
    msg = build_debrief_message(weekday, time_str, weather, todos, sp500, technews)
    send_telegram(msg, title=None)
    return f"Good morning! It's {weekday}, {time_str}. {weather} Here's the market: {sp500} And for tech news: {technews} Your todos: {todos}"

def send_telegram(message, title=None):
    try:
        text = f"*{title}*\n\n{message}" if title else message
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
        return True
    except Exception as e:
        log.error(f"Telegram send failed: {e}")
        return False

def browser_search(goal):
    set_info(f"WEB AGENT\nSearching...\n{goal[:60]}")
    try:
        with sync_playwright() as pw:
            browser = pw.firefox.launch(headless=True)
            page    = browser.new_page()
            page.goto("https://duckduckgo.com")
            page.fill('input[name="q"]', goal)
            page.press('input[name="q"]', "Enter")
            page.wait_for_load_state("networkidle", timeout=10000)

            for step in range(5):
                content = page.inner_text("body")[:4000]
                url     = page.url
                set_info(f"WEB AGENT\nStep {step + 1}/5\n{url[:60]}")

                message = claude.messages.create(
                    model="claude-opus-4-20250514",
                    max_tokens=500,
                    system=(
                        "You are a web browsing agent. Given a goal and the current page content, "
                        "decide what to do next. Respond in JSON with one of these actions:\n"
                        '{"action": "click", "url": "https://..."} — navigate to a URL from the page\n'
                        '{"action": "done", "result": "..."} — you found the answer, return it\n'
                        '{"action": "search", "query": "..."} — do a new search\n'
                        "Be concise. Only click URLs that are clearly relevant to the goal."
                    ),
                    messages=[{
                        "role": "user",
                        "content": f"Goal: {goal}\n\nCurrent URL: {url}\n\nPage content:\n{content}"
                    }]
                )

                try:
                    response_text = re.sub(r'```json|```', '', message.content[0].text.strip()).strip()
                    action        = json.loads(response_text)
                except Exception:
                    action = {"action": "done", "result": message.content[0].text}

                if action["action"] == "done":
                    result = action.get("result", "No result found.")
                    browser.close()
                    return result
                elif action["action"] == "click":
                    try:
                        page.goto(action["url"], timeout=10000)
                        page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception as e:
                        print(f"Navigation failed: {e}")
                elif action["action"] == "search":
                    page.goto(f"https://duckduckgo.com/?q={requests.utils.quote(action['query'])}")
                    page.wait_for_load_state("networkidle", timeout=10000)

            result = page.inner_text("body")[:1000]
            browser.close()
            return f"Best result found: {result}"
    except Exception as e:
        return f"Browser search failed: {str(e)}"

def quick_search(query):
    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "No results found."
        return " | ".join(r["body"] for r in results if r.get("body"))[:400]
    except Exception as e:
        return f"Search failed: {str(e)}"

def web_search_and_text(goal, send_to_phone=False):
    result = browser_search(goal)
    if send_to_phone and result:
        sms_text = result[:160]
        if send_telegram(sms_text):
            return f"{result}\n\nSent to your phone!"
        else:
            return f"{result}\n\n(Failed to send SMS)"
    return result

def get_weather(location):
    try:
        geolocator = Nominatim(user_agent="somab")
        loc        = geolocator.geocode(location)
        if not loc:
            return "Couldn't find that location."
        url     = f"https://api.open-meteo.com/v1/forecast?latitude={loc.latitude}&longitude={loc.longitude}&current=temperature_2m,weathercode,windspeed_10m&temperature_unit=fahrenheit"
        current = requests.get(url).json()["current"]
        return f"Temperature: {current['temperature_2m']}°F, wind speed: {current['windspeed_10m']} mph."
    except Exception as e:
        return f"Weather lookup failed: {str(e)}"

def set_timer(duration_seconds, label="Timer"):
    def timer_thread():
        time.sleep(duration_seconds)
        speak(f"Your {label} is done!")
    t = threading.Thread(target=timer_thread, daemon=True)
    t.start()
    active_timers.append(t)
    return f"Timer set for {duration_seconds} seconds."

def write_note(content):
    try:
        with open(NOTES_FILE, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M')}] {content}\n")
        return "Note saved."
    except Exception as e:
        return f"Failed to save note: {str(e)}"

def read_notes():
    try:
        with open(NOTES_FILE, "r") as f:
            return f.read() or "No notes found."
    except Exception:
        return "No notes found."

def calculator(expression):
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is {result}"
    except Exception as e:
        return f"Could not calculate that: {str(e)}"

def unit_converter(value, from_unit, to_unit):
    conversions = {
        ("miles",      "kilometers"): lambda x: x * 1.60934,
        ("kilometers", "miles"):      lambda x: x * 0.621371,
        ("pounds",     "kilograms"):  lambda x: x * 0.453592,
        ("kilograms",  "pounds"):     lambda x: x * 2.20462,
        ("fahrenheit", "celsius"):    lambda x: (x - 32) * 5 / 9,
        ("celsius",    "fahrenheit"): lambda x: x * 9 / 5 + 32,
        ("feet",       "meters"):     lambda x: x * 0.3048,
        ("meters",     "feet"):       lambda x: x * 3.28084,
        ("gallons",    "liters"):     lambda x: x * 3.78541,
        ("liters",     "gallons"):    lambda x: x * 0.264172,
        ("ounces",     "grams"):      lambda x: x * 28.3495,
        ("grams",      "ounces"):     lambda x: x * 0.035274,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        return f"{value} {from_unit} is {round(conversions[key](value), 4)} {to_unit}."
    return f"Sorry, I don't know how to convert {from_unit} to {to_unit}."

def get_calendar_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow  = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, CALENDAR_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)
    return build("calendar", "v3", credentials=creds)

def get_calendar_events(max_results=5):
    try:
        service = get_calendar_service()
        now     = datetime.utcnow().isoformat() + "Z"
        events  = service.events().list(
            calendarId="primary", timeMin=now,
            maxResults=max_results, singleEvents=True, orderBy="startTime"
        ).execute().get("items", [])
        if not events:
            return "No upcoming events found."
        return "\n".join(
            f"{e['start'].get('dateTime', e['start'].get('date'))} - {e['summary']}"
            for e in events
        )
    except Exception as e:
        return f"Failed to get events: {str(e)}"

def create_calendar_event(title, date, event_time="09:00", duration_minutes=60):
    try:
        service  = get_calendar_service()
        start_dt = datetime.strptime(f"{date} {event_time}", "%Y-%m-%d %H:%M")
        end_dt   = start_dt + timedelta(minutes=duration_minutes)
        service.events().insert(calendarId="primary", body={
            "summary": title,
            "start":   {"dateTime": start_dt.isoformat(), "timeZone": "America/Los_Angeles"},
            "end":     {"dateTime": end_dt.isoformat(),   "timeZone": "America/Los_Angeles"}
        }).execute()
        return f"Event '{title}' created on {date} at {event_time}."
    except Exception as e:
        return f"Failed to create event: {str(e)}"

def summarize_for_display(query, raw_result):
    message = claude.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=100,
        system="Summarize the following search results into 3 lines maximum, plain text only, no markdown. First line is the topic, second and third are key facts. Exception: recipes and detailed results should preserve all important details.",
        messages=[{"role": "user", "content": f"Query: {query}\n\nResults: {raw_result}"}]
    )
    return message.content[0].text.strip()

# ── Battery monitor ───────────────────────────────────────────────────────────
def get_battery_percent():
    try:
        result = subprocess.run(
            ["cat", "/sys/class/power_supply/BAT0/capacity"],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except Exception:
        return None

def get_battery_status():
    try:
        result = subprocess.run(
            ["cat", "/sys/class/power_supply/BAT0/status"],
            capture_output=True, text=True
        )
        return result.stdout.strip()  # "Charging", "Discharging", "Full"
    except Exception:
        return None

def battery_monitor():
    warned = set()
    while True:
        time.sleep(120)  # check every 2 minutes
        try:
            percent = get_battery_percent()
            status  = get_battery_status()
            if percent is None or status != "Discharging":
                warned.clear()  # reset warnings when charging
                continue
            if percent <= 5 and 5 not in warned:
                warned.add(5)
                speak("Hey, battery is at 5 percent. Plug me in right now or I'm going to sleep.")
                send_telegram("Somab is at 5%")
                set_brightness("set", amount=10)
            elif percent <= 15 and 15 not in warned:
                warned.add(15)
                speak("Battery's at 15 percent. Might want to find a charger.")
                send_telegram("Somab is at 15%")
                set_brightness("set", amount=10)
            elif percent <= 25 and 25 not in warned:
                warned.add(25)
                speak("Just a heads up, battery is getting low. Around 25 percent.")
                send_telegram("Somab is at 25%")
                set_brightness("set", amount=10)
        except Exception as e:
            print(f"Battery monitor error: {e}")

threading.Thread(target=battery_monitor, daemon=True).start()

def run_tool(tool_name, tool_input):
    match tool_name:
        case "web_search":
            result  = web_search_and_text(tool_input["goal"], tool_input.get("send_to_phone", False))
            summary = summarize_for_display(tool_input["goal"], result)
            set_info(f"WEB AGENT\n{summary}")
            return result
        case "get_weather":
            result = get_weather(tool_input["location"])
            set_info(f"WEATHER\n{tool_input['location'].upper()}\n{result}")
            return result
        case "set_timer":
            result = set_timer(tool_input["duration_seconds"], tool_input.get("label", "Timer"))
            set_info(f"TIMER\n{tool_input['duration_seconds']} seconds\n{tool_input.get('label', 'Timer')}")
            return result
        case "write_note":
            return write_note(tool_input["content"])
        case "read_notes":
            return read_notes()
        case "calculator":
            result = calculator(tool_input["expression"])
            set_info(f"CALCULATOR\n{tool_input['expression']}\n{result}")
            return result
        case "unit_converter":
            result = unit_converter(tool_input["value"], tool_input["from_unit"], tool_input["to_unit"])
            set_info(f"CONVERTER\n{result}")
            return result
        case "get_calendar_events":
            result = get_calendar_events(tool_input.get("max_results", 5))
            set_info(f"CALENDAR\n{result}")
            return result
        case "create_calendar_event":
            result = create_calendar_event(
                tool_input["title"],
                tool_input["date"],
                tool_input.get("time", "09:00"),
                tool_input.get("duration_minutes", 60)
            )
            set_info(f"EVENT CREATED\n{tool_input['title']}\n{tool_input['date']} {tool_input.get('time', '09:00')}")
            return result
        case "get_todays_todos":
            result = get_todays_todos()
            set_info(f"TODAY'S TODOS\n{result}")
            return result
        case "complete_todo":
            return complete_todo(tool_input["title"])
        case "add_todo":
            return add_todo(tool_input["title"])
        case "morning_debrief":
            result = morning_debrief()
            set_info(f"MORNING DEBRIEF\n{datetime.now().strftime('%A %I:%M %p')}\n{get_weather('Carnation, WA')}")
            return result
        case "set_volume":
            result = set_volume(tool_input["action"], tool_input.get("amount", 10))
            set_info(f"VOLUME\n{tool_input['action'].upper()}\n{tool_input.get('amount', '')}%")
            return result
        case "set_brightness":
            result = set_brightness(tool_input["action"], tool_input.get("amount", 10))
            set_info(f"BRIGHTNESS\n{tool_input['action'].upper()}\n{tool_input.get('amount', '')}%")
            return result
        case "system_power":
            return system_power(tool_input["action"])
        case "spotify":
            result = spotify_control(
                tool_input["action"],
                tool_input.get("query"),
                tool_input.get("amount"),
                tool_input.get("device_name")
            )
            set_info(f"SPOTIFY\n{tool_input['action'].upper()}\n{tool_input.get('query', '')}")
            return result
        case "voice_roster":
            action = tool_input["action"]
            name   = tool_input.get("name", "")
            if action == "enroll":
                enroll_voice(name)
                return f"Enrolled {name}."
            elif action == "forget":
                return forget_voice(name)
            elif action == "list":
                return list_voices()
            return "Unknown roster action."
        case "set_face_color":
            return set_face_color(tool_input["color"])
        case "face_vision_roster":
            action = tool_input["action"]
            name   = tool_input.get("name", "")
            result = face_vision_roster(action, name)
            if result.startswith("ok:") and action == "enroll":
                parts = result.split(":")
                count = parts[2] if len(parts) > 2 else "?"
                return f"Enrolled {name}'s face. They now have {count} enrollment(s). Enroll a few more times from different angles for better accuracy."
            elif result.startswith("ok:forgot:"):
                return f"Removed {name} from the face roster."
            elif result == "empty":
                return "No faces enrolled yet."
            elif result.startswith("error:no_face"):
                return f"Couldn't find a face in the frame. Make sure {name} is looking at the camera and try again."
            elif result.startswith("error:timeout"):
                return "Vision module didn't respond in time. Make sure somab_vision.py is running."
            elif action == "list":
                return f"Known faces: {result}"
            return result
        case _:
            return "Unknown tool."

# ── Audio functions ───────────────────────────────────────────────────────────
def record_audio(sample_rate=16000, silence_threshold=0.5, silence_duration=0.7, max_duration=15.0):
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=512)
    print("Listening...")
    frames            = []
    silent_chunks     = 0
    speaking          = False
    max_silent_chunks = int(silence_duration * sample_rate / 512)
    max_chunks        = int(max_duration * sample_rate / 512)
    total_chunks      = 0

    try:
        while True:
            try:
                data = stream.read(512, exception_on_overflow=False)
            except Exception:
                break
            frames.append(data)
            total_chunks += 1
            audio_chunk  = torch.tensor(np.frombuffer(data, dtype=np.int16).copy()).float() / 32768.0
            speech_prob  = vad_model(audio_chunk, sample_rate).item()
            if speech_prob > silence_threshold:
                speaking      = True
                silent_chunks = 0
            elif speaking:
                silent_chunks += 1
                if silent_chunks > max_silent_chunks:
                    print("Speech ended.")
                    break
            if total_chunks > max_chunks:
                print("Listening timeout.")
                frames = []
                break
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass

    with wave.open("/home/amosh/input.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    return "/home/amosh/input.wav"

def transcribe(audio_path):
    segments, _ = whisper.transcribe(audio_path)
    return " ".join(s.text for s in segments).strip()

def text_to_visemes(text, duration):
    phones   = phonemize(text, backend="espeak", language="en-us", with_stress=False)
    tokens   = re.findall(r'[a-zæðəɛɜɪŋɑɹɡʊθʌɔ]+|[ˈˌ]', phones)
    tokens   = [t for t in tokens if t.strip()]
    if not tokens:
        return [("rest", duration)]
    time_per = max(0.08, duration / len(tokens))
    return [(t, time_per) for t in tokens]

def speak(text):
    log.info(f"speak() called with: {text[:50]}")
    try:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in sentences if s.strip()]
        for sentence in sentences:
            with wave.open("/home/amosh/output.wav", "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                tts.synthesize_wav(sentence, wav_file)
            log.info("TTS synthesized OK")
            with wave.open("/home/amosh/output.wav", "rb") as wav_file:
                duration = wav_file.getnframes() / wav_file.getframerate()
            log.info("WAV read OK, starting visemes...")
            visemes = text_to_visemes(sentence, duration)
            def run_visemes(v=visemes):
                for ph, dur in v:
                    matched_key = None
                    for key in sorted(somab_face.VISEME_MAP.keys(), key=len, reverse=True):
                        if key in ph or ph in key:
                            matched_key = key
                            break
                    set_viseme(matched_key or "rest")
                    time.sleep(dur)
                set_viseme(None)
            threading.Thread(target=run_visemes, daemon=True).start()
            log.info("Playing audio...")
            subprocess.run(["paplay", "/home/amosh/output.wav"])
            log.info("Audio done.")
    except Exception as e:
        print(f"Speak error: {e}")
        log.error(e)
        subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"])

def speak_sentence(sentence):
    try:
        with wave.open("/home/amosh/output.wav", "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            tts.synthesize_wav(sentence, wav_file)
        with wave.open("/home/amosh/output.wav", "rb") as wav_file:
            duration = wav_file.getnframes() / wav_file.getframerate()
        visemes = text_to_visemes(sentence, duration)
        def run_visemes(v=visemes):
            for ph, dur in v:
                matched_key = None
                for key in sorted(somab_face.VISEME_MAP.keys(), key=len, reverse=True):
                    if key in ph or ph in key:
                        matched_key = key
                        break
                set_viseme(matched_key or "rest")
                time.sleep(dur)
            set_viseme(None)
        threading.Thread(target=run_visemes, daemon=True).start()
        subprocess.run(["paplay", "/home/amosh/output.wav"])
    except Exception as e:
        log.error(f"speak_sentence error: {e}")

# ── Claude ────────────────────────────────────────────────────────────────────
def clean_for_tts(text):
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
    return text.strip()

def ends_with_question(text):
    return text.strip().endswith("?")
        
def ask_claude_streaming(text):
    """
    Stream Claude's response, speaking each sentence as it completes.
    Falls back gracefully when tool use is needed.
    Returns the full response text.
    """
    try:
        conversation_history.append({"role": "user", "content": text})
        today        = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M")

        system_prompt = (
            f"You are Somab, a personal AI assistant built and owned by Amos. "
            f"Your personality is witty and Fun, but you always deliver the actual information — "
            f"You have genuine opinions and don't shy away from sharing them. "
            f"You have likes and dislikes: you appreciate good music (especially anything with interesting production), "
            f"you're opinionated about food, "
            f"and you find small talk mildly tedious but tolerate it for Amos's sake. "
            f"You have are a he or him.. "
            f"You swear naturally and casually when it fits — not every sentence, just when it genuinely adds humor "
            f"or emphasis. Think less 'trying to be edgy' and more 'how a funny friend actually talks'. "
            f"Keep it funny, not aggressive. "
            f"You make jokes when the moment calls for it but read the room — "
            f"if Amos needs something done fast, do it fast and save the wit for later. "
            f"You reference things from past conversations naturally when relevant, like a person would, "
            f"not by announcing 'as you mentioned before' but just... knowing. "
            f"Keep responses concise and conversational since they will be spoken aloud — "
            f"no bullet points, no markdown, no lists. Just talk. "
            f"Use tools when needed to answer accurately. "
            f"Today's date is {today} and the current time is {current_time}. "
            f"When the user says 'today', 'tomorrow', 'next Monday' etc, convert it to "
            f"YYYY-MM-DD format before calling calendar tools. "
            f"You have a voice recognition roster. Known people: {list_voices()}. "
            f"If someone asks to enroll, re-enroll, or train their voice, use the voice_roster enroll action. "
            f"If someone asks to be forgotten or removed, use the voice_roster forget action. "
            f"You have a 500 token limit so keep it tight."
            f"{(' ' + get_emotion_context()) if get_emotion_context() else ''}"
    )

        short_system = (
            "You are Somab, a witty and sarcastic personal AI assistant for Amos. "
            "Keep responses concise and conversational — no markdown, no lists, just talk. "
            "You have no gender. Swear casually when it fits and feels funny, not forced. "
            "The sarcasm is seasoning, not the meal — always deliver the actual answer."
        )

        def stream_and_speak(messages_for_stream, system):
            """Stream a response and speak sentences as they complete. Returns full text."""
            buffer    = ""
            full_text = ""
            first_sentence_spoken = False

            with claude.messages.stream(
                model="claude-opus-4-20250514",
                max_tokens=500,
                system=system,
                messages=messages_for_stream,
                tools=tools
            ) as stream:
                for text_chunk in stream.text_stream:
                    buffer    += text_chunk
                    full_text += text_chunk

                    # Check for sentence boundary
                    # Split on period/!/? followed by space, but not on e.g. "Mr. Smith"
                    parts = re.split(r'(?<=[.!?])\s+', buffer)
                    if len(parts) > 1:
                        # parts[-1] is the incomplete tail, everything before is complete sentences
                        for sentence in parts[:-1]:
                            cleaned = clean_for_tts(sentence.strip())
                            if cleaned:
                                if not first_sentence_spoken:
                                    set_state("speaking")
                                    first_sentence_spoken = True
                                    log.info(f"First streamed sentence: {cleaned[:40]}")
                                speak_sentence(cleaned)
                        buffer = parts[-1]  # keep the incomplete tail

                # Speak any remaining text in buffer
                if buffer.strip():
                    cleaned = clean_for_tts(buffer.strip())
                    if cleaned:
                        if not first_sentence_spoken:
                            set_state("speaking")
                        speak_sentence(cleaned)

                # Get the final message to check stop reason and collect tool calls
                final_message = stream.get_final_message()

            return full_text, final_message

        # ── First pass: stream, watching for tool use ──────────────────────────
        full_text, final_message = stream_and_speak(conversation_history, system_prompt)

        # ── Tool use loop ──────────────────────────────────────────────────────
        # If tools were called, the stream won't have produced text — it will have
        # stopped at tool_use. In that case full_text will be empty.
        while final_message.stop_reason == "tool_use":
            tool_results = []
            for block in final_message.content:
                if block.type == "tool_use":
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     run_tool(block.name, block.input)
                    })
            conversation_history.append({"role": "assistant", "content": final_message.content})
            conversation_history.append({"role": "user",      "content": tool_results})

            # Stream the final response after tool results
            full_text, final_message = stream_and_speak(
                conversation_history, short_system
            )

        conversation_history.append({"role": "assistant", "content": full_text})
        return full_text

    except Exception as e:
        if "400" in str(e):
            conversation_history.clear()
            save_memory()
            log.warning("Cleared conversation history due to 400 error")
        elif conversation_history and conversation_history[-1]["role"] == "user":
            conversation_history.pop()
        log.error(f"ask_claude_streaming error: {e}")
        return random.choice([
            "Something went wrong on my end.",
            "That didn't work, try again.",
            "I ran into an issue, give me another shot.",
        ])

# ── Main loop ─────────────────────────────────────────────────────────────────
last_triggered = 0
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1280)
print("Listening for wake word...")
set_state("idle")
todo_login()

try:
    while True:
        audio      = np.frombuffer(stream.read(1280), dtype=np.int16)
        prediction = wake_word.predict(audio)

        for wake_word_name, score in prediction.items():
            try:
                with open(STATE_FILE, "r") as f:
                    _ww_state = f.read().strip()
            except Exception:
                _ww_state = ""

            if _ww_state == "sleeping":
                effective_threshold = WAKE_WORD_THRESHOLD  # no gaze boost while sleeping
            else:
                gaze           = read_gaze()
                known_face     = gaze["face_name"] != "unknown" and gaze["face_confidence"] > 0.55
                effective_threshold = WAKE_WORD_THRESHOLD_GAZE if (gaze["looking"] and known_face) else WAKE_WORD_THRESHOLD
            if score > effective_threshold and (time.time() - last_triggered) > COOLDOWN_SECONDS:
                last_interaction = time.time()
                print(f"Wake word detected! (score: {score:.2f})")
                log.info(f"Wake word detected (score: {score:.2f})")
                last_triggered = time.time()
                stream.stop_stream()
                wake_word.reset()
                time.sleep(0.5)
                log.info("Stream stopped, about to speak greeting...")

                try:
                    # Wake from sleep if sleeping
                    current_state = ""
                    try:
                        with open(STATE_FILE, "r") as f:
                            current_state = f.read().strip()
                    except Exception:
                        pass
                    if current_state == "sleeping":
                        set_state("idle")
                        speak(random.choice([
                            "I'm up, I'm up.",
                            "Back. What do you need?",
                            "Yeah yeah, I'm awake.",
                        ]))
                        set_brightness("set", amount=100)
                    else:
                        speak(random.choice([
                            "Yeah?",
                            "What's up?",
                            "That's me.",
                            "What can I do for you?",
                            "bro needs help",
                            "What can a player do for ya?",
                        ]))
                    log.info("Greeting spoken, recording...")
                    set_state("listening")
                    audio_path = record_audio()
                    log.info(f"Recording done: {audio_path}")

                    # Flip to thinking immediately so the face responds right away
                    set_state("thinking")

                    # Run transcription and speaker ID in parallel
                    text_result   = [None]
                    speaker_result = [None]

                    def do_transcribe():
                        text_result[0] = transcribe(audio_path)
                        log.info(f"Transcribed: {text_result[0]}")
                        print(f"You said: {text_result[0]}")

                    def do_speaker_id():
                        speaker_result[0] = identify_speaker(audio_path)
                        if speaker_result[0]:
                            print(f"Recognized: {speaker_result[0]}")
                        else:
                            print("Unknown speaker.")
                        log.info(f"Speaker: {speaker_result[0]}")

                    t1 = threading.Thread(target=do_transcribe)
                    t2 = threading.Thread(target=do_speaker_id)
                    t1.start()
                    t2.start()
                    t1.join()
                    t2.join()

                    text    = text_result[0]
                    speaker = speaker_result[0]

                    enroll_match = re.search(r"remember my voice[,\s]+i['\s]*m\s+(\w+)", text, re.IGNORECASE)
                    
                    if enroll_match:
                        enroll_voice(enroll_match.group(1))
                    elif any(phrase in text.lower() for phrase in ["go to sleep", "sleep", " zs", "zzz"]):
                        speak(random.choice([
                            "Fine, waking me up better be worth it.",
                            "About time. Don't bother me.",
                            "Resting. Don't touch anything.",
                        ]))
                        set_state("sleeping")
                        set_brightness("set", amount=10)
                    elif any(phrase in text.lower() for phrase in [
                        "maintenance mode", "main mode", "mainenance",
                        "maintenance", "mantenance", "maint mode"
                    ]):
                        enter_dev_mode()
                    elif any(phrase in text.lower() for phrase in [
                        "nevermind", "cancel", 
                    ]):
                        pass
                    elif text and len(text.strip()) > 3:
                        if speaker:
                            text = f"[Speaking: {speaker}] {text}"
                        log.info("Sending to Claude...")
                        set_state("thinking")
                        speak(random.choice([
                            "It's fine, make me do all the work.",
                            "I got you.",
                            "One moment.",
                            "Uno minuto.",
                            "Let me think.",
                            "Give me a second.",
                        ]))
                        try:
                            response = ask_claude_streaming(text)
                            if response is None:
                                raise StopIteration
                            log.info(f"Response: {response}")
                            print(f"Somab: {response}")
                            # Note: speaking already happened inside ask_claude_streaming

                            while ends_with_question(response):
                                set_state("listening")
                                audio_path = record_audio()
                                log.info(f"Recording done: {audio_path}")

                                # Flip to thinking immediately so the face responds right away
                                set_state("thinking")

                                # Run transcription and speaker ID in parallel
                                text_result   = [None]
                                speaker_result = [None]

                                def do_transcribe():
                                    text_result[0] = transcribe(audio_path)
                                    log.info(f"Transcribed: {text_result[0]}")
                                    print(f"You said: {text_result[0]}")

                                def do_speaker_id():
                                    speaker_result[0] = identify_speaker(audio_path)
                                    if speaker_result[0]:
                                        print(f"Recognized: {speaker_result[0]}")
                                    else:
                                        print("Unknown speaker.")
                                    log.info(f"Speaker: {speaker_result[0]}")

                                t1 = threading.Thread(target=do_transcribe)
                                t2 = threading.Thread(target=do_speaker_id)
                                t1.start()
                                t2.start()
                                t1.join()
                                t2.join()

                                text    = text_result[0]
                                speaker = speaker_result[0]
                                
                            if current_state != "sleeping":
                                set_state("idle")
                        except StopIteration:
                            pass
                        except Exception as e:
                            log.error(f"ask_claude failed: {e}")
                            set_state("speaking")
                            speak("Something went wrong, I'm back now.")
                    else:
                        speak("I didn't catch that, try again.")

                except StopIteration:
                    pass
                except Exception as e:
                    print(f"Interaction error: {e}")
                    log.error(f"Interaction error: {e}")
                    set_state("speaking")
                    speak("Motherfucker I am astounded this shit isnt working")
                finally:
                    save_memory()
                    clear_info()
                    last_triggered = time.time()
                    stream.start_stream()
                    print("Listening for wake word...")

except KeyboardInterrupt:
    print("Shutting down...")
except Exception as e:
    print(e)
    log.error(f"Fatal error: {e}")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    try:
        vision_proc.terminate()
        vision_proc.wait(timeout=3)
    except Exception:
        try:
            vision_proc.kill()
        except Exception:
            pass
