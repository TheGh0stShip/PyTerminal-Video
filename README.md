# Terminal Video Renderer (ANSI Truecolor + Optional Audio)

Plays a video inside a terminal by converting each frame into ANSI-colored character cells. Grayscale output is the default. Full RGB output and audio playback are optional.

This repository is intended to run either:
- **Locally** (Windows/macOS/Linux), or
- **Remotely over SSH** on a **Kali Linux x64 host** (e.g., an Azure VM), streaming the rendered terminal output to your client (Windows Terminal, etc.)

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
- If you run the script on a remote Kali VM over SSH, audio plays on the **remote VM** (usually not useful). For local audio, run the script locally.
