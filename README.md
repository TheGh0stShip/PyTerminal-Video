# Terminal Video Renderer (ANSI Truecolor + Optional Audio)

Plays a video inside a terminal by converting each frame into ANSI-colored character cells. Grayscale output is the default. Full RGB output and audio playback are optional.

This repository is intended to run either:
- **Locally** (Windows/macOS/Linux), or
- **Remotely over SSH** on a **Linux x64 host** (e.g., an Azure VM), streaming the rendered terminal output to your client (Windows Terminal, etc.)

**Usage** python3 'PyTerminal-Video.py' 'video.mp4' --audio --audio-warmup 250

**v1**
<img width="1293" height="511" alt="image" src="https://github.com/user-attachments/assets/2c876b48-428c-45a2-bfcc-9ce6151b28a1" />

**v2**
<img width="1278" height="524" alt="image" src="https://github.com/user-attachments/assets/fca4e871-c6fd-46ec-b68b-317869a80b43" />
Includes new features! Progress bar, navigation buttons, subtitles, greyscale, and PAUSE!

---

## What you need

- **Python 3.9+**
- Python packages:
  - `numpy`
  - `opencv-python` (or `opencv-python-headless` on servers)
- Optional audio:
  - `ffplay` (provided by `ffmpeg`)
- A terminal that supports ANSI escape sequences and a font that can display **Braille block characters** (used for texture)

Audio behavior:
- `--audio` plays on the **machine running the script**.
- If you run the script on a remote host over SSH, audio plays on the **remote host** (usually not useful). For local audio, run the script locally.

## Install

### Debian / Ubuntu / Kali
`sudo apt-get update \
  && sudo apt-get install -y python3 python3-pip ffmpeg \
  && python3 -m pip install --upgrade pip \
  && python3 -m pip install --upgrade numpy opencv-python-headless`

### MacOS (Homebrew)
`brew install python ffmpeg \
  && python3 -m pip install --upgrade pip \
  && python3 -m pip install --upgrade numpy opencv-python`

### Windows PowerShell + Winget
`winget install -e --id Python.Python.3.12
winget install -e --id Gyan.FFmpeg
py -m pip install --upgrade pip
py -m pip install --upgrade numpy opencv-python`
