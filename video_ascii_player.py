import cv2
import os
import sys
import time
import pygame
from PIL import Image

ASCII_CODE = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", " "]
WEIGHTS = [0.33, 0.33, 0.33]

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

def extract_audio(video_path):
    audio_tmp = "_tmp_audio.wav"
    cmd = f'ffmpeg -y -i "{video_path}" -vn -ac 2 -ar 44100 "{audio_tmp}" > /dev/null 2>&1'
    os.system(cmd)
    return audio_tmp

def play_video_ascii(video_path, width=80, invert=False):
    if not os.path.exists(video_path):
        print("Video not found:", video_path)
        return

    audio_file = extract_audio(video_path)
    if not os.path.exists(audio_file):
        print("Audio extraction failed.")
        return

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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    try:
        while True:
            audio_pos = pygame.mixer.music.get_pos() / 1000.0
            frame_index = int(audio_pos / frame_time)
            if frame_index >= len(frames):
                break
            ascii_frame = frame_to_ascii(
                frames[frame_index],
                width=width,
                invert=invert,
                r_w=WEIGHTS[0],
                g_w=WEIGHTS[1],
                b_w=WEIGHTS[2]
            )
            os.system("clear")
            print(ascii_frame)
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass

    pygame.mixer.music.stop()
    if os.path.exists("_tmp_audio.wav"):
        os.remove("_tmp_audio.wav")

def main():
    if len(sys.argv) < 2:
        print("Usage: python video_ascii_sync.py <video.mp4> [width] [--invert]")
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
