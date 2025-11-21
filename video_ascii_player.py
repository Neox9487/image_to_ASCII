import cv2
import os
import sys
import time
import subprocess
from PIL import Image

ASCII_CODE = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", " "]
WEIGHTS = [0.42, 0.11, 0.47]


def rgb_to_weighted_luminance(r, g, b, r_w=0.33, g_w=0.33, b_w=0.33):
    return r * r_w + g * g_w + b * b_w


def frame_to_ascii(frame, width=80, scale=0.55, invert=False,
                   r_w=0.33, g_w=0.33, b_w=0.33):

    img = Image.fromarray(frame).convert("RGB")
    orig_w, orig_h = img.size
    new_w = int(width)
    new_h = max(1, int((orig_h / orig_w) * new_w * scale))
    img = img.resize((new_w, new_h))

    pixels = img.load()
    h = img.height
    w = img.width
    buckets = len(ASCII_CODE) - 1

    lines = []
    for y in range(h):
        line_chars = []
        for x in range(w):
            r, g, b = pixels[x, y]
            lum = rgb_to_weighted_luminance(r, g, b, r_w, g_w, b_w)
            if invert:
                lum = 255 - lum
            idx = int(lum * buckets / 255)
            line_chars.append(ASCII_CODE[idx])
        lines.append("".join(line_chars))

    return "\n".join(lines)


def start_audio_process(audio_path):
    try:
        proc = subprocess.Popen(
            [
                "ffplay",
                "-nodisp",     
                "-autoexit", 
                "-loglevel", "quiet",
                audio_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return proc
    except FileNotFoundError:
        print("can't find available ffplay `sudo apt install ffmpeg`")
        return None
    except Exception as e:
        print("start audio player failed", e)
        return None


def play_video_ascii(video_path, width=80, invert=False):
    if not os.path.exists(video_path):
        print("can't find video: ", video_path)
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("can't open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback
    frame_delay = 1 / fps

    audio_tmp = "_tmp_audio.wav"
    cmd = f'ffmpeg -y -i "{video_path}" -vn -ac 2 -ar 44100 -f wav "{audio_tmp}" > /dev/null 2>&1'
    os.system(cmd)

    audio_proc = None
    if os.path.exists(audio_tmp):
        audio_proc = start_audio_process(audio_tmp)

    os.system("clear")
    print("press Ctrl-C stop...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ascii_frame = frame_to_ascii(
                frame,
                width=width,
                invert=invert,
                r_w=WEIGHTS[0],
                g_w=WEIGHTS[1],
                b_w=WEIGHTS[2]
            )

            os.system("clear")
            print(ascii_frame)

            time.sleep(frame_delay)

    except KeyboardInterrupt:
        print("\nplayer stopped")

    cap.release()

    if audio_proc is not None:
        try:
            audio_proc.terminate()
        except Exception:
            pass

    if os.path.exists(audio_tmp):
        try:
            os.remove(audio_tmp)
        except Exception:
            pass

def main():
    if len(sys.argv) < 2:
        print("python video_ascii_player.py <video_path> [width] [--invert]")
        return

    video_path = sys.argv[1]
    width = 80
    invert = False

    args = sys.argv[2:]
    for a in args:
        if a.isdigit():
            width = int(a)
        elif a == "--invert":
            invert = True

    play_video_ascii(video_path, width=width, invert=invert)


if __name__ == "__main__":
    main()
