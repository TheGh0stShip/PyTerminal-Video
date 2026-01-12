# Terminal Video Renderer (ANSI Truecolor + Optional Audio)

Plays a video inside a terminal by converting each frame into ANSI-colored character cells. Grayscale output is the default. Full RGB output and audio playback are optional.

This repository is intended to run either:
- **Locally** (Windows/macOS/Linux), or
- **Remotely over SSH** on a **Linux x64 host** (e.g., an Azure VM), streaming the rendered terminal output to your client (Windows Terminal, etc.)

**Usage** python3 'PyTerminal-Video' 'video.mp4' --color --audio

<img width="1293" height="511" alt="image" src="https://github.com/user-attachments/assets/2c876b48-428c-45a2-bfcc-9ce6151b28a1" />

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
