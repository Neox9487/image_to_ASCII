import cv2
import os
import sys
import time
import subprocess
from PIL import Image

ASCII_CODE = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", " "]
WEIGHTS = [0.33, 0.33, 0.33] # r, g, b weights

def rgb_to_weighted_luminance(r, g, b, r_w, g_w, b_w):
    return r * r_w + g * g_w + b * b_w

def frame_to_ascii(frame, width, scale, invert, r_w, g_w, b_w):
    img = Image.fromarray(frame).convert("RGB")
    ow, oh = img.size
    nw = width
    nh = max(1, int((oh / ow) * nw * scale))
    img = img.resize((nw, nh))
    px = img.load()
    h = img.height
    w = img.width
    buckets = len(ASCII_CODE) - 1
    lines = []
    for y in range(h):
        row = []
        for x in range(w):
            r, g, b = px[x, y]
            lum = rgb_to_weighted_luminance(r, g, b, r_w, g_w, b_w)
            if invert:
                lum = 255 - lum
            row.append(ASCII_CODE[int(lum * buckets / 255)])
        lines.append("".join(row))
    return "\n".join(lines)

def extract_audio(video_path):
    audio_tmp = "_tmp_audio.wav"
    os.system(f'ffmpeg -y -i "{video_path}" -vn -ac 2 -ar 44100 "{audio_tmp}" > /dev/null 2>&1')
    return audio_tmp

def play_video_ascii(video_path, width=80, invert=False):
    audio = extract_audio(video_path)
    if not os.path.exists(audio):
        print("audio extract failed, please check if video: "+ video_path+ "realy existed?")
        return

    audio_proc = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio]
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_time = 1.0 / fps
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    t0 = time.time()

    try:
        for i, f in enumerate(frames):
            t_target = i * frame_time
            while (time.time() - t0) < t_target:
                pass
            ascii_f = frame_to_ascii(
                f, width, 0.55, invert,
                WEIGHTS[0], WEIGHTS[1], WEIGHTS[2]
            )
            os.system("clear")
            print(ascii_f)
    except KeyboardInterrupt:
        pass

    audio_proc.terminate()
    if os.path.exists(audio):
        os.remove(audio)

def main():
    if len(sys.argv) < 2:
        print("usage: python video_ascii_player.py <video.mp4> [width] [--invert]")
        return

    video_path = sys.argv[1]
    width = 80
    invert = False

    for a in sys.argv[2:]:
        if a.isdigit():
            width = int(a)
        elif a == "--invert":
            invert = True

    play_video_ascii(video_path, width, invert)

if __name__ == "__main__":
    main()
