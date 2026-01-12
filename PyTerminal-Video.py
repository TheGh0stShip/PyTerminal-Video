import os
import sys
import time
import ctypes
import cv2
import numpy as np
import argparse
import subprocess

### Usage python 'PyTerminal-Video.py' 'video-name.mp4' --color --audio ###

WIDTH = 140
ASPECT_FIX = 0.42
FRAME_SKIP = 1

EDGE_THRESHOLD = 18
GAMMA = 1.0

TEXTURE_CHARS = " ⡀⡄⡆⡇⡏⡟⡿⣿"


def enable_vt():
    if os.name != "nt":
        return
    k32 = ctypes.windll.kernel32
    h = k32.GetStdHandle(-11)
    mode = ctypes.c_uint32()
    if k32.GetConsoleMode(h, ctypes.byref(mode)):
        k32.SetConsoleMode(h, mode.value | 0x0004)


def bg_rgb(r, g, b):
    return f"\033[48;2;{r};{g};{b}m"


def fg_rgb(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"


def apply_gamma(gray):
    if GAMMA == 1.0:
        return gray
    inv = 1.0 / GAMMA
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)


def frame_to_cells(frame, use_color):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = apply_gamma(gray)

    h, w = gray.shape
    new_h = max(1, int(WIDTH * (h / w) * ASPECT_FIX))

    gray = cv2.resize(gray, (WIDTH, new_h), interpolation=cv2.INTER_AREA)
    color = cv2.resize(frame, (WIDTH, new_h), interpolation=cv2.INTER_AREA)

    edges = cv2.Laplacian(gray, cv2.CV_16S)
    edges = cv2.convertScaleAbs(edges)

    lines = []
    for y in range(new_h):
        row = []
        for x in range(WIDTH):
            b, g, r = color[y, x]

            if use_color:
                bg = bg_rgb(r, g, b)
            else:
                v = gray[y, x]
                bg = bg_rgb(v, v, v)

            if edges[y, x] > EDGE_THRESHOLD:
                idx = min(
                    len(TEXTURE_CHARS) - 1,
                    edges[y, x] * len(TEXTURE_CHARS) // 255
                )
                ch = TEXTURE_CHARS[idx]

                if use_color:
                    fg = fg_rgb(255 - r, 255 - g, 255 - b)
                else:
                    fg = fg_rgb(255, 255, 255)

                row.append(f"{bg}{fg}{ch}")
            else:
                row.append(f"{bg} ")

        lines.append("".join(row))

    return "\n".join(lines)


def play(video, use_color=False, use_audio=False):
    enable_vt()

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("[-] Failed to open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_dt = 1.0 / fps

    audio_proc = None
    if use_audio:
        try:
            audio_proc = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", video],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            print("[!] ffplay not found - running without audio")

    sys.stdout.write("\033[2J\033[?25l")
    sys.stdout.flush()

    start = time.perf_counter()
    shown = 0
    idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            idx += 1
            if FRAME_SKIP > 1 and idx % FRAME_SKIP:
                continue

            ascii_frame = frame_to_cells(frame, use_color)

            sys.stdout.write("\033[H")
            sys.stdout.write(ascii_frame)
            sys.stdout.write("\033[0m")
            sys.stdout.flush()

            deadline = start + shown * frame_dt
            sleep = deadline - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

            shown += 1
    finally:
        cap.release()
        if audio_proc:
            audio_proc.terminate()
        sys.stdout.write("\033[?25h\033[0m\n")
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Terminal video renderer (color + audio optional)"
    )
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--color", action="store_true", help="Enable full RGB color output")
    parser.add_argument("--audio", action="store_true", help="Enable audio playback (via ffplay)")
    args = parser.parse_args()

    play(args.video, use_color=args.color, use_audio=args.audio)
