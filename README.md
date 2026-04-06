# Somab

A fully custom, personality-driven AI voice assistant — built from scratch with a matching custom hardware platform. Somab runs as a kiosk-style device with a face, a voice, a webcam, and a personality that isn't here to impress anyone.

**Version:** v0.12 · **Last Updated:** April 2026 · **Status:** Active development

---

## What It Is

Somab is a standalone voice assistant with an animated face, real-time speech recognition, streaming AI responses, webcam-based vision, and long-term memory. It runs on a dedicated ThinkPad X1 Carbon Gen 9 (Pop!_OS 24) as a software prototype, with a custom Jetson Orin NX carrier PCB in design as the eventual deployment target.

It has a defined personality: witty, sarcastic, no assigned gender, casual swearing enabled. It knows your name.

---

## Architecture

Somab is split across three cooperating Python processes that communicate via flat IPC files in `/home/amosh/`:

| Process | Role |
|---|---|
| `somab.py` | Main brain — STT, LLM, TTS, tools, conversation state |
| `somab_face.py` | pygame face renderer — animated expressions, lip sync |
| `somab_vision.py` | Webcam pipeline — gaze detection, face recognition, emotion |

IPC files: `somab_state.txt`, `somab_viseme.txt`, `somab_gaze.txt`, `somab_info.txt`, `somab_vision_cmd.txt`, `somab_vision_result.txt`

---

## Software Stack

| Layer | Technology |
|---|---|
| **Wake word** | OpenWakeWord (custom `hey_so_mab.onnx` model) |
| **Speech-to-text** | Faster-Whisper (`small` for speed, `medium` for accuracy) |
| **LLM** | Claude API — Sonnet (main), Haiku (memory extraction, summarization) |
| **Text-to-speech** | Piper TTS (`en_US-lessac-medium`, custom voice in training) |
| **Face renderer** | pygame (current) → Rive in Chromium kiosk (planned) |
| **Vision** | MediaPipe FaceLandmarker (tasks API 0.10+), VIDEO mode, 1280×720 @ 15fps |
| **Face recognition** | `face_recognition` library, CNN model |
| **Speaker ID** | Resemblyzer |
| **Long-term memory** | ChromaDB + `all-MiniLM-L6-v2` embeddings, cosine dedup (threshold 0.15) |
| **Conversation log** | SQLite |
| **Music** | Spotipy (Spotify Web API) |
| **Search** | DuckDuckGo (`ddgs`) |
| **Web scraping** | Playwright |
| **Location** | geopy / Nominatim |
| **Calendar** | Google Calendar API |
| **Notifications** | Telegram Bot API |
| **Audio** | PipeWire / pactl |
| **Display brightness** | brightnessctl |

---

## Conversation Flow

1. Wake word detected (`hey somab`) → optional gaze confirmation lowers threshold
2. VAD records utterance via PyAudio
3. Faster-Whisper transcribes in parallel with Resemblyzer speaker ID
4. Face flips to `thinking` immediately while inference runs
5. Claude API streams response sentence-by-sentence
6. Each sentence is synthesized and spoken via Piper as it arrives
7. `[WAIT]` / `[DONE]` continuation tags control whether Somab keeps listening after speaking
8. 20-second silence timeout ends the engagement

---

## Vision Pipeline

- **1280×720 MJPEG** capture (YUYV tops out at lower resolution on this webcam)
- **15fps target** — benchmarked as the stable operating point with Whisper inference spikes (~450% CPU at 30fps)
- **Gaze detection** via yaw estimation + iris deviation, 3-frame majority vote smoother
- **Face recognition** using CNN model with background `RecognitionThread` (deprioritized via `os.nice(15)`)
- **Emotion detection** from MediaPipe's 52 blendshape coefficients — 8 emotions (happy, sad, angry, surprised, disgusted, fearful, contempt, tired) + neutral
- Sustained emotion (20s) on a known face triggers a proactive comment; 10-minute cooldown per emotion type
- Vision process idles automatically during sleep mode

---

## Voice Tools

Somab can be instructed to:

- Set volume / brightness
- Play music via Spotify (genre, playlist, artist)
- Search the web (DuckDuckGo)
- Scrape a URL (Playwright)
- Check weather
- Read / create Google Calendar events
- Add/read notes
- Add todos (external API)
- Send SMS / email
- Manage long-term memory (`manage_memory`)
- Enter maintenance mode (drop to COSMIC desktop + open editor)
- Sleep / wake
- Shutdown / restart the system

---

## Memory

- **Short-term:** 20-turn sliding conversation window with prompt caching
- **Ambient buffer:** 5-minute rolling transcription context (~800 chars), injected into system prompt
- **Long-term:** ChromaDB vector store, Haiku-based extraction, cosine deduplication
- **Morning debrief:** Daily digest (weather, S&P 500, AI/tech news) delivered via Telegram (HTML parse mode)

---

## Face & Expressions

**Current:** pygame renderer with bezier control-point morphing. Features independent eyebrow control, circular eyes, `> <` chevron X-squint for thinking state, idle micro-expressions, sleeping state with floating Z animations, battery warnings, and dynamic face color changeable by voice command.

**Planned:** Migration to [Rive](https://rive.app) running in a Chromium kiosk window. Python ↔ Rive via WebSocket on `localhost:7734`. Multi-layer state machine: facial states layer (idle/listening/thinking/speaking/sleeping) + mouth/viseme layer composing independently. 80ms blend times between states.

---

## Hardware

### Prototype
- **ThinkPad X1 Carbon Gen 9** — dedicated machine, Pop!_OS 24, COSMIC desktop
- 8GB RAM (uses swap at idle with Somab running)
- Systemd user service, auto-launch via greetd on login

### Target Platform (PCB Rev A — schematic complete, layout not started)
- **NVIDIA Jetson Orin NX 16GB** on custom KiCad 9 carrier board
- Connector: TE Connectivity 2309413-1 (single 260-pin SODIMM)
- HDMI display output
- HUSB238 USB PD power delivery
- I2S audio, I2C peripherals, USB hub, status LED, boot control

### PCB Rev B (planned)
- M.2 Key M slot — Hailo-8L vision accelerator
- M.2 Key E slot — Intel AX200 (WiFi 6 + Bluetooth 5.2)
- Board size bump to ~120×80mm

### Phase 1 Peripherals (planned purchases)
- ReSpeaker USB Mic Array v2.0
- Google Coral USB Accelerator
- Anker USB 3.0 hub
- IMX219 CSI camera

---

## Project Structure

```
somab.py              # Main process — brain
somab_face.py         # Face renderer (pygame)
somab_vision.py       # Vision pipeline (MediaPipe)
somab_hardware_overview.html  # PCB/hardware reference doc
SOMAB_PROJECT_LOG.md  # Living project changelog
/home/amosh/
├── .somab.env        # API keys and credentials
├── somab_voices.json # Speaker voice roster
├── somab_notes.txt   # Persistent notes store
├── somab_memory.json # Long-term memory backing
├── somab.log         # Runtime log
└── somab_*.txt       # IPC files between processes
```

---

## Known Issues (v0.12)

- **Speaker ID broken** — `identify_speaker()` receives transcript text instead of audio file path
- **Sleep monitor never updates** — `emotion_monitor` missing `global last_interaction`
- **PCB layout** — schematic finalized, layout not yet started
- **Piper custom voice** — training blocked on Kaggle Cell 1 (Python 3.12 Cython fix applied, pending confirmation)

---

## Roadmap

- [ ] Rive face design and Chromium kiosk integration
- [ ] Fix speaker ID argument bug
- [ ] PCB Rev A layout and routing in KiCad
- [ ] Confirm Piper custom voice training on Kaggle (P100)
- [ ] Order Phase 1 USB peripherals
- [ ] Pre-synthesized TTS filler sound cache
- [ ] Interruption detection
- [ ] Persistent reminders
- [ ] Ollama local inference on Jetson (with Claude API fallback)
- [ ] XTTS v2 voice cloning
- [ ] QLoRA fine-tune for baked-in personality (Unsloth)
- [ ] Watchdog / crash recovery
- [ ] MediaPipe Pose/Holistic gesture detection (Step 5 vision)

---

## Non-Commercial

Somab is a personal project. Not a product. Not for sale.
