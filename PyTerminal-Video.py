import os
import sys
import time
import ctypes
import argparse
import subprocess
import threading
import textwrap
from bisect import bisect_right
from queue import Queue, Empty

import cv2
import numpy as np

np.seterr(over="ignore")

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
    return f"\033[48;2;{int(r)};{int(g)};{int(b)}m"


def fg_rgb(r, g, b):
    return f"\033[38;2;{int(r)};{int(g)};{int(b)}m"


def apply_gamma(gray):
    if GAMMA == 1.0:
        return gray
    inv = 1.0 / GAMMA
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)


def fmt_time(sec):
    if sec < 0:
        sec = 0
    total = int(sec + 0.5)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def parse_srt_time(ts):
    ts = ts.strip().replace(".", ",")
    parts = ts.split(":")
    if len(parts) != 3:
        return 0.0
    hh = int(parts[0])
    mm = int(parts[1])
    ss_ms = parts[2].split(",")
    ss = int(ss_ms[0])
    ms = int(ss_ms[1]) if len(ss_ms) > 1 else 0
    return hh * 3600 + mm * 60 + ss + (ms / 1000.0)


def load_srt(path):
    items = []
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            lines = [ln.rstrip("\n") for ln in f]
    except OSError:
        return items

    i = 0
    n = len(lines)
    while i < n:
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break

        if lines[i].strip().isdigit():
            i += 1
            if i >= n:
                break

        if i >= n or "-->" not in lines[i]:
            i += 1
            continue

        time_line = lines[i]
        i += 1
        try:
            a, b = time_line.split("-->")
            start = parse_srt_time(a.strip())
            end = parse_srt_time(b.strip().split()[0])
        except Exception:
            continue

        text_lines = []
        while i < n and lines[i].strip():
            text_lines.append(lines[i])
            i += 1

        text = " ".join(t.strip() for t in text_lines if t.strip())
        if text:
            items.append((start, end, text))

    items.sort(key=lambda x: x[0])
    return items


def find_sub_index(starts, t):
    return bisect_right(starts, t) - 1


class AbortRender(Exception):
    pass


def make_bw_frame(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 128, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def frame_to_cells_v2(frame, use_color, abort_event=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = apply_gamma(gray)

    h, w = gray.shape
    new_h = max(1, int(WIDTH * (h / w) * ASPECT_FIX))

    gray_r = cv2.resize(gray, (WIDTH, new_h), interpolation=cv2.INTER_AREA)
    color_r = cv2.resize(frame, (WIDTH, new_h), interpolation=cv2.INTER_AREA)

    edges = cv2.Laplacian(gray_r, cv2.CV_16S)
    edges = cv2.convertScaleAbs(edges)

    lines = []
    for y in range(new_h):
        if abort_event is not None and abort_event.is_set():
            raise AbortRender()
        row = []
        for x in range(WIDTH):
            b, g, r = color_r[y, x]
            if use_color:
                bg = bg_rgb(r, g, b)
            else:
                v = gray_r[y, x]
                bg = bg_rgb(v, v, v)

            e = edges[y, x]
            if e > EDGE_THRESHOLD:
                with np.errstate(over="ignore"):
                    idx = min(len(TEXTURE_CHARS) - 1, (e * len(TEXTURE_CHARS)) // 255)
                ch = TEXTURE_CHARS[int(idx)]
                if use_color:
                    fg = fg_rgb(255 - r, 255 - g, 255 - b)
                else:
                    fg = fg_rgb(255, 255, 255)
                row.append(f"{bg}{fg}{ch}")
            else:
                row.append(f"{bg} ")
        lines.append("".join(row))

    return "\n".join(lines), new_h


class KeyReader:
    def __init__(self):
        self.is_windows = os.name == "nt"
        self._old_termios = None

    def __enter__(self):
        if not self.is_windows:
            import termios
            import tty
            fd = sys.stdin.fileno()
            self._old_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.is_windows and self._old_termios is not None:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)

    def poll(self):
        if self.is_windows:
            import msvcrt
            if not msvcrt.kbhit():
                return None
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                code = msvcrt.getwch()
                if code == "K":
                    return ("ARROW", "LEFT")
                if code == "M":
                    return ("ARROW", "RIGHT")
                return None
            if ch == "\t":
                return ("CMD", "TAB")
            if ch == " ":
                return ("CMD", "SPACE")
            if ch in ("m", "M"):
                return ("CMD", "MUTE")
            if ch in ("c", "C"):
                return ("CMD", "COLOR")
            if ch in ("q", "Q", "\x03"):
                return ("CMD", "QUIT")
            if ch.isdigit():
                return ("NUM", ch)
            return None

        import select
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if not r:
            return None
        s = sys.stdin.read(1)

        if s == "\x1b":
            r2, _, _ = select.select([sys.stdin], [], [], 0)
            if not r2:
                return None
            s2 = sys.stdin.read(1)
            if s2 != "[":
                return None
            r3, _, _ = select.select([sys.stdin], [], [], 0)
            if not r3:
                return None
            s3 = sys.stdin.read(1)
            if s3 == "D":
                return ("ARROW", "LEFT")
            if s3 == "C":
                return ("ARROW", "RIGHT")
            return None

        if s == "\t":
            return ("CMD", "TAB")
        if s == " ":
            return ("CMD", "SPACE")
        if s in ("m", "M"):
            return ("CMD", "MUTE")
        if s in ("c", "C"):
            return ("CMD", "COLOR")
        if s in ("q", "Q", "\x03"):
            return ("CMD", "QUIT")
        if s.isdigit():
            return ("NUM", s)
        return None


def _spawn_ffplay(video, start_sec, volume):
    args = [
        "ffplay",
        "-nodisp",
        "-autoexit",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-sync",
        "audio",
        "-volume",
        str(int(clamp(volume, 0, 100))),
    ]
    if start_sec > 0:
        args += ["-ss", f"{start_sec:.3f}"]
    args += [video]

    p = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    time.sleep(0.12)
    if p.poll() is None:
        return p, ""
    err = ""
    try:
        if p.stderr:
            err = (p.stderr.read() or "").strip()
    except Exception:
        err = ""
    return None, (err or "ffplay exited immediately")


class InputPump(threading.Thread):
    def __init__(self, kr, out_q, abort_event, stop_event):
        super().__init__(daemon=True)
        self.kr = kr
        self.out_q = out_q
        self.abort_event = abort_event
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            ev = self.kr.poll()
            if ev:
                self.out_q.put(ev)
                self.abort_event.set()
            else:
                time.sleep(0.002)


def play(video, mode=0, use_audio=False, audio_warmup_ms=300, subtitles_path=None, osd_time=2.0):
    enable_vt()

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("[-] Failed to open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration_sec = (frame_count / fps) if (frame_count > 0 and fps > 0) else 0.0

    subs = []
    if subtitles_path:
        subs = load_srt(subtitles_path)
    else:
        base, _ = os.path.splitext(video)
        auto = base + ".srt"
        if os.path.exists(auto):
            subs = load_srt(auto)

    subs_starts = [s for s, _, _ in subs]
    subs_ends = [e for _, e, _ in subs]
    subs_texts = [t for _, _, t in subs]
    subs_idx = -1

    audio_proc = None
    muted = False
    paused = False

    warmup_s = max(0.0, audio_warmup_ms / 1000.0)

    osd_pinned = False
    osd_until = 0.0

    wall_base = time.perf_counter()

    last_frame_text = ""
    last_frame_h = 1
    last_t = 0.0

    def stop_audio():
        nonlocal audio_proc
        if audio_proc:
            try:
                audio_proc.terminate()
            except Exception:
                pass
            try:
                audio_proc.wait(timeout=1.0)
            except Exception:
                pass
            audio_proc = None

    def start_audio(at_sec):
        nonlocal audio_proc
        stop_audio()
        if not use_audio:
            return
        vol = 0 if muted else 100
        try:
            audio_proc, err = _spawn_ffplay(video, at_sec, vol)
            if audio_proc is None and err:
                print("[!] Audio disabled:", err)
        except FileNotFoundError:
            print("[!] ffplay not found - running without audio")
            audio_proc = None

    def anchor_clock(at_sec):
        nonlocal wall_base
        wall_base = time.perf_counter() - at_sec

    def restart_audio_and_anchor(at_sec):
        if not use_audio:
            anchor_clock(at_sec)
            return
        if paused:
            stop_audio()
            anchor_clock(at_sec)
            return
        start_audio(at_sec)
        if warmup_s > 0:
            time.sleep(warmup_s)
        anchor_clock(at_sec)

    def show_osd_now():
        nonlocal osd_until
        osd_until = time.perf_counter() + max(0.0, osd_time)

    def current_time_sec():
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_ms and pos_ms >= 0:
            return float(pos_ms) / 1000.0
        if fps > 0:
            pos_frames = cap.get(cv2.CAP_PROP_POS_FRAMES) or 0.0
            cur_frame = max(0, int(pos_frames) - 1)
            return cur_frame / fps
        return last_t

    def seek_to_time(target_sec):
        if target_sec < 0:
            target_sec = 0.0
        if duration_sec > 0 and target_sec > duration_sec:
            target_sec = duration_sec

        ok = cap.set(cv2.CAP_PROP_POS_MSEC, float(target_sec * 1000.0))
        if not ok and fps > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(int(target_sec * fps)))

        if fps > 0:
            target_frame = int(target_sec * fps)
            for _ in range(600):
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES) or 0.0
                if int(pos) >= target_frame:
                    break
                ok2, _ = cap.read()
                if not ok2:
                    break
            actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or target_frame)
            actual_sec = actual_frame / fps
        else:
            actual_sec = target_sec

        restart_audio_and_anchor(actual_sec)
        show_osd_now()
        return actual_sec

    def get_subtitle(t):
        nonlocal subs_idx
        if not subs:
            return ""
        if subs_idx < 0:
            subs_idx = find_sub_index(subs_starts, t)

        while subs_idx + 1 < len(subs) and subs_starts[subs_idx + 1] <= t:
            subs_idx += 1
        while subs_idx >= 0 and subs_starts[subs_idx] > t:
            subs_idx -= 1

        if subs_idx >= 0 and subs_starts[subs_idx] <= t <= subs_ends[subs_idx]:
            return subs_texts[subs_idx]
        return ""

    def build_osd_lines(t):
        subtitle = get_subtitle(t)
        sub_lines = []
        if subtitle:
            wrapped = textwrap.wrap(subtitle, width=max(10, WIDTH - 2))
            sub_lines = wrapped[:2]

        if duration_sec > 0:
            pct = clamp(t / duration_sec, 0.0, 1.0)
            pct_txt = f"{int(pct * 100):3d}%"
            tail = f" {fmt_time(t)}/{fmt_time(duration_sec)} {pct_txt}"
        else:
            tail = f" {fmt_time(t)}"

        flags = []
        flags.append("PAUSED" if paused else "PLAY")
        flags.append("MUTE" if (use_audio and muted) else ("AUD" if use_audio else "NOAUD"))
        flags.append("COLOR" if mode == 1 else ("B/W" if mode == 2 else "GRAY"))
        flag_txt = " | " + " ".join(flags)

        bar_total = WIDTH - (len(tail) + len(flag_txt) + 3)
        bar_total = max(10, bar_total)
        filled = int(bar_total * (clamp(t / duration_sec, 0.0, 1.0) if duration_sec > 0 else 0.0))
        bar = "[" + ("=" * filled) + ("-" * (bar_total - filled)) + "]"

        line = f"{bar}{tail}{flag_txt}"
        if len(line) < WIDTH:
            line += " " * (WIDTH - len(line))
        else:
            line = line[:WIDTH]

        sty = f"{bg_rgb(0,0,0)}{fg_rgb(255,255,255)}"
        out = []
        for s in sub_lines:
            s = " " + s
            if len(s) < WIDTH:
                s += " " * (WIDTH - len(s))
            else:
                s = s[:WIDTH]
            out.append(sty + s + "\033[0m")
        out.append(sty + line + "\033[0m")
        return out

    def draw(frame_text, frame_h, t):
        sys.stdout.write("\033[H")
        sys.stdout.write(frame_text)
        sys.stdout.write("\033[0m")

        now = time.perf_counter()
        want_osd = osd_pinned or (now < osd_until)
        if want_osd:
            osd_lines = build_osd_lines(t)
            osd_h = len(osd_lines)
            start_row = max(1, frame_h - osd_h + 1)
            sys.stdout.write(f"\033[{start_row};1H")
            sys.stdout.write("\n".join(osd_lines))

        sys.stdout.write("\033[0m")
        sys.stdout.flush()

    sys.stdout.write("\033[2J\033[?25l")
    sys.stdout.flush()

    last_t = seek_to_time(0.0)

    q = Queue()
    abort_event = threading.Event()
    stop_event = threading.Event()

    try:
        with KeyReader() as kr:
            pump = InputPump(kr, q, abort_event, stop_event)
            pump.start()

            idx = 0

            while True:
                abort_event.clear()

                handled = False
                while True:
                    try:
                        et, val = q.get_nowait()
                    except Empty:
                        break

                    handled = True

                    if et == "CMD" and val == "QUIT":
                        return

                    if et == "CMD" and val == "TAB":
                        osd_pinned = not osd_pinned
                        show_osd_now()
                        draw(last_frame_text, last_frame_h, last_t)
                        continue

                    if et == "CMD" and val == "SPACE":
                        paused = not paused
                        if paused:
                            stop_audio()
                        else:
                            restart_audio_and_anchor(last_t)
                        show_osd_now()
                        draw(last_frame_text, last_frame_h, last_t)
                        continue

                    if et == "CMD" and val == "MUTE":
                        muted = not muted
                        if paused:
                            stop_audio()
                            anchor_clock(last_t)
                        else:
                            restart_audio_and_anchor(last_t)
                        show_osd_now()
                        draw(last_frame_text, last_frame_h, last_t)
                        continue

                    if et == "CMD" and val == "COLOR":
                        mode = (mode + 1) % 3
                        show_osd_now()
                        draw(last_frame_text, last_frame_h, last_t)
                        continue

                    if et == "ARROW":
                        tcur = current_time_sec()
                        step = (0.10 * duration_sec) if duration_sec > 0 else 5.0
                        if val == "LEFT":
                            last_t = seek_to_time(tcur - step)
                        elif val == "RIGHT":
                            last_t = seek_to_time(tcur + step)
                        continue

                    if et == "NUM":
                        d = int(val)
                        if d == 0:
                            last_t = seek_to_time(0.0)
                        else:
                            if duration_sec > 0:
                                last_t = seek_to_time(duration_sec * (d / 10.0))
                        continue

                if paused:
                    if not handled:
                        time.sleep(0.01)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                idx += 1
                if FRAME_SKIP > 1 and (idx % FRAME_SKIP):
                    continue

                t = current_time_sec()
                last_t = t

                target_wall = wall_base + t

                while True:
                    remaining = target_wall - time.perf_counter()
                    if remaining <= 0:
                        break
                    if abort_event.is_set():
                        break
                    time.sleep(min(0.005, remaining))

                if abort_event.is_set():
                    continue

                render_frame = frame
                use_color = (mode == 1)
                if mode == 2:
                    render_frame = make_bw_frame(frame)
                    use_color = False

                try:
                    frame_text, frame_h = frame_to_cells_v2(render_frame, use_color, abort_event=abort_event)
                except AbortRender:
                    continue

                last_frame_text = frame_text
                last_frame_h = frame_h
                draw(frame_text, frame_h, t)

    finally:
        stop_event.set()
        cap.release()
        try:
            if audio_proc:
                audio_proc.terminate()
        except Exception:
            pass
        sys.stdout.write("\033[?25h\033[0m\n")
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Keys: Left/Right seek +/-10%, 1-9 jump, 0 start, TAB OSD, SPACE pause, M mute, C mode, Q quit."
    )
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--audio", action="store_true", help="Enable audio playback via ffplay")
    parser.add_argument("--audio-warmup-ms", type=int, default=300, help="Anchor sync after N ms when (re)starting audio")
    parser.add_argument("--subtitles", default=None, help="Path to .srt file (defaults to <video_basename>.srt if present)")
    parser.add_argument("--osd-time", type=float, default=2.0, help="Seconds to show OSD after seeking (TAB pins it)")
    parser.add_argument("--mode", choices=["gray", "color", "bw"], default="gray", help="Initial display mode")
    args = parser.parse_args()

    initial_mode = 0 if args.mode == "gray" else 1 if args.mode == "color" else 2

    play(
        args.video,
        mode=initial_mode,
        use_audio=args.audio,
        audio_warmup_ms=args.audio_warmup_ms,
        subtitles_path=args.subtitles,
        osd_time=args.osd_time,
    )
