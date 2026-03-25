# Somab — Personal Voice AI Assistant

Somab is a fully local, always-on voice AI assistant built to run as the primary interface on a dedicated laptop. It features an animated face, wake word detection, natural conversation, and a growing set of tools — all controlled entirely by voice.

---

## Overview

Somab boots automatically when the laptop starts, fills the screen with an animated face, and listens for its wake word. When it hears you, it responds with synthesized speech, lip-synced to an expressive animated face. It uses the Claude API as its brain and runs all audio processing locally.

The design philosophy is simple: it should feel like talking to a character, not issuing commands to a device.

---

## Hardware

- **Machine:** Dedicated laptop (ThinkPad X1 Carbon Gen 9)
- **OS:** Pop!_OS 24 with COSMIC desktop, full disk encryption (LUKS)
- **Audio:** PipeWire via `paplay`
- **Display:** Fullscreen pygame face via systemd user service

---

## Tech Stack

| Component | Technology |
|---|---|
| Wake word | OpenWakeWord (custom "Hey Somab" model) |
| Speech to text | Faster Whisper (medium, CPU, int8) |
| AI brain | Anthropic Claude API (claude-opus-4) |
| Text to speech | Piper TTS (en_US-lessac-medium) |
| Voice activity detection | Silero VAD |
| Face rendering | Pygame (bezier morphing system) |
| Voice recognition | Resemblyzer |
| Browser automation | Playwright (Firefox headless) |

---

## Features

### Core
- Wake word detection — custom trained "Hey Somab" model
- Voice activity detection — auto-stops recording when you finish speaking
- Lip-synced animated face — bezier-based morphing system with true shape interpolation
- Conversation memory — persists last 20 exchanges across restarts
- Voice recognition roster — learns and recognizes enrolled speakers

### Tools
- **Web search** — agentic browser search using Playwright + Claude
- **Weather** — real-time via Open-Meteo API (no key required)
- **Timers** — background threads, supports multiple concurrent timers
- **Notes** — append/read a persistent notes file
- **Calculator** — safe expression evaluation
- **Unit converter** — common unit conversions
- **Google Calendar** — read upcoming events, create new events (OAuth2)
- **Todo list** — integrated with dailytodo REST API
- **Morning debrief** — weather + todos spoken aloud + texted to phone
- **SMS** — send messages to phone via Gmail SMTP gateway
- **Spotify** — play/pause/skip/search tracks and playlists, multi-device support
- **System volume** — increase/decrease/mute/set via pactl
- **Screen brightness** — increase/decrease/set via brightnessctl
- **System power** — shutdown/restart
- **Face color** — change face color by name or hex code at runtime
- **Voice roster** — enroll/forget/list recognized speakers

### Face States
| State | Description |
|---|---|
| Idle | Circular open eyes, smile, random micro-expressions |
| Listening | Open eyes, smile, right eyebrow raised |
| Thinking | `> <` X-squint eyes, flat mouth |
| Speaking | Squint eyes, open mouth with live lip sync |
| Sleeping | Half-closed squint eyes, smile, floating Z animations |
| Dev mode | Password overlay for maintenance access |

### Behaviors
- **Idle micro-expressions** — random subtle expressions (curious squint, raised brow, amused smirk, deep blink) during idle state
- **Sleeping** — goes to sleep after 1 hour of no interaction, or on command ("go to sleep")
- **Wake from sleep** — wake word wakes Somab back up with a groggy response
- **Follow-up conversation** — continues listening if Somab's response ends with a question
- **Battery warnings** — spoken alerts at 25%, 15%, and 5% with SMS notification
- **Face color change** — say "turn pink" or "go blue" to change face color live
- **Maintenance mode** — voice-triggered, password-protected escape to desktop

---

## File Structure

```
/home/USER/
├── somab.py                  # Main brain — wake word, STT, Claude, TTS, tools
├── somab_face.py             # Pygame face renderer
├── somab-session.sh          # Systemd session launch script
├── somab-exit-to-desktop.sh  # Maintenance mode exit script
├── .somab.env                # Secret keys (never commit this)
├── somab_memory.json         # Conversation history
├── somab_notes.txt           # Voice notes
├── somab_voices.json         # Voice recognition embeddings
├── somab_color.txt           # Current face color (R,G,B)
├── somab_state.txt           # IPC: current face state
├── somab_viseme.txt          # IPC: current viseme/phoneme
├── somab_info.txt            # IPC: info panel content
├── somab_devmode_input.txt   # IPC: dev mode password input
├── somab_devmode_result.txt  # IPC: dev mode result
├── somab.log                 # Runtime log
├── input.wav                 # Temporary STT audio
├── output.wav                # Temporary TTS audio
├── piper-voices/             # Piper TTS voice models
└── somab/                    # Python virtual environment
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo>
cd somab
python3 -m venv somab
source somab/bin/activate
```

### 2. Install dependencies

```bash
pip install anthropic faster-whisper openwakeword piper-tts silero-vad \
    pygame phonemizer resemblyzer spotipy ddgs playwright \
    python-dotenv geopy google-auth google-auth-oauthlib \
    google-api-python-client requests beautifulsoup4 pyaudio torch

playwright install firefox
sudo apt install brightnessctl unclutter portaudio19-dev espeak-ng
```

### 3. Configure environment variables

Create `~/.somab.env`:

```env
ANTHROPIC_API_KEY=your_key_here
DEV_PASSWORD=your_password_here
TODO_EMAIL=your_email_here
TODO_PASSWORD=your_password_here
SMTP_EMAIL=your_gmail_here
SMTP_PASSWORD=your_app_password_here
PHONE_SMS_EMAIL=your_number@carrier.com
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

> **Never commit `.somab.env` to version control.**

### 4. Download Piper voice model

```bash
mkdir -p ~/piper-voices
# Download en_US-lessac-medium.onnx and .onnx.json from
# https://github.com/rhasspy/piper/releases
```

### 5. Google Calendar setup

- Create a project at https://console.cloud.google.com
- Enable the Google Calendar API
- Download OAuth2 credentials as `~/somab_credentials.json`
- Run Somab once manually — it will open a browser to authorize

### 6. Spotify setup

- Create an app at https://developer.spotify.com/dashboard
- Enable Web API and Web Playback SDK
- Set redirect URI to `http://127.0.0.1:8888/callback`
- Add Client ID and Secret to `.somab.env`

### 7. Wake word

The default model is a custom "Hey Somab" `.onnx` model trained with OpenWakeWord. To use a different wake word, replace the model path in `WAKEWORD_MODEL` and retrain or use a built-in model.

### 8. Initialize color file

```bash
echo "77,217,224" > ~/somab_color.txt
```

### 9. Audio setup (PipeWire)

Somab uses `paplay` for audio output. Make sure your default sink is set to speakers:

```bash
pactl set-default-sink alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__Speaker__sink
```

If `fluidsynth` is holding the audio device on startup, mask it:

```bash
systemctl --user mask fluidsynth
```

---

## Running as a Service

### Create the service file

`~/.config/systemd/user/somab.service`:

```ini
[Unit]
Description=Somab Voice Assistant
After=graphical-session.target pipewire.service pipewire-pulse.service
Wants=graphical-session.target

[Service]
Type=simple
ExecStart=/home/USER/somab-session.sh
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0
Environment=XDG_RUNTIME_DIR=/run/user/1000
Environment=PULSE_RUNTIME_PATH=/run/user/1000/pulse
Environment=DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus

[Install]
WantedBy=graphical-session.target
```

### Enable and start

```bash
systemctl --user enable somab.service
systemctl --user start somab.service
```

### Session script (`somab-session.sh`)

```bash
#!/bin/bash
pkill fluidsynth 2>/dev/null
pactl set-default-sink alsa_output...Speaker__sink
xset s off
xset -dpms
xset s noblank
unclutter -idle 1 &
export XDG_RUNTIME_DIR=/run/user/1000
export PULSE_RUNTIME_PATH=/run/user/1000/pulse
cd /home/USER
source /home/USER/somab/bin/activate
python3 /home/USER/somab.py
```

---

## Auto-login (greetd / COSMIC)

Edit `/etc/greetd/cosmic-greeter.toml` to add auto-login:

```toml
[initial_session]
command = "start-cosmic"
user = "USER"
```

---

## Maintenance Mode

Say **"maintenance mode"** to Somab. It will prompt for a password on screen. Enter your `DEV_PASSWORD` and it will exit to the desktop and open your code editor and terminal automatically.

---

## Customization

### Change personality
Edit the system prompt in `ask_claude()` inside `somab.py`.

### Add tools
1. Add a tool definition to the `tools` list
2. Implement the function
3. Add a `case` to `run_tool()`

### Add face states
1. Add a state to `STATES` in `somab_face.py`
2. Call `set_state("your_state")` from `somab.py`

### Add face colors
Colors can be changed at runtime by voice ("turn red", "go purple") or by writing directly to `~/somab_color.txt` as `R,G,B`.

### Add idle expressions
Add entries to the `expressions` list in `IdleExpression._trigger()` in `somab_face.py`. Each expression is a dict with `left_eye`, `right_eye`, `mouth`, `brow_l`, `brow_r`, and `duration`.

---

## Known Issues

- Conversation history can occasionally produce 400 errors from the Claude API if tool result blocks don't serialize cleanly — auto-cleared when detected
- Whisper medium model takes 3-5 seconds to transcribe on CPU — a GPU would significantly speed this up
- `paplay` can take 20-30 seconds if the PipeWire socket isn't found — ensure `XDG_RUNTIME_DIR` is set in the service environment

---

## Roadmap

- Custom voice cloning with Coqui XTTS v2
- Email reading via Gmail API
- Reminders (persists across restarts)
- News briefing via RSS
- App launcher
- Face recognition (InsightFace)
- Move to custom CM4 PCB hardware

---

## License

Personal project — not licensed for redistribution.
