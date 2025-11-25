import cv2
import os
import sys
import numpy as np
import subprocess
from PIL import ImageFont, Image, ImageDraw
from tqdm import tqdm 

# === CUDA ===
CUDA_AVAILABLE = False
try:
    import cupy as cp
    if cp.is_available():
        CUDA_AVAILABLE = True
except:
    CUDA_AVAILABLE = False
print("CUDA available:", CUDA_AVAILABLE)

# === ASCII code ===
# ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", " "]

ASCII = np.frombuffer(b"@#S%?*+;:, ", dtype=np.uint8)
WEIGHTS = np.array([0.33, 0.33, 0.33])  # r, g, b weights

def find_mono_font():
    """find available fonts"""
    fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
    ]
    for f in fonts:
        if os.path.exists(f):
            return f
    print("No monospace font found.")
    sys.exit(1)

FONT_PATH = find_mono_font()
FONT = ImageFont.truetype(FONT_PATH, 14)
ascent, descent = FONT.getmetrics()
CH = ascent + descent
CW = int(FONT.getlength("A"))

CHAR_CACHE = {}

for ch in ASCII:
    c = chr(ch) 
    img = Image.new("L", (CW, CH), 0)
    d = ImageDraw.Draw(img)
    d.text((0, 0), c, font=FONT, fill=255)
    CHAR_CACHE[ch] = np.array(img, dtype=np.uint8)

# === transitions ===
def frame_to_ascii(frame, ascii_width, invert):
    H, W, _ = frame.shape
    aspect = H / W
    ascii_height = int(ascii_width * aspect * (CW / CH))

    small = cv2.resize(frame, (ascii_width, ascii_height))

    if CUDA_AVAILABLE:
        small_gpu = cp.asarray(small, dtype=cp.float32)
        lum = (small_gpu * cp.array(WEIGHTS)).sum(axis=2)
        if invert:
            lum = 255 - lum

        idx = (lum * (len(ASCII) - 1) / 255).astype(cp.uint8)
        ascii_img = ASCII[idx.get()] 
        return ascii_img
    else:
        lum = (small * WEIGHTS).sum(axis=2)

        if invert:
            lum = 255 - lum

        idx = (lum * (len(ASCII) - 1) / 255).astype(np.uint8)
        return ASCII[idx]

def ascii_to_image(ascii_img):
    """transfer ASCII to image"""
    h, w = ascii_img.shape

    out_w = w * CW
    out_h = h * CH

    out = np.zeros((out_h, out_w), dtype=np.uint8)

    for i in range(h):
        row = ascii_img[i]
        y0 = i * CH
        for j in range(w):
            char = row[j]
            x0 = j * CW
            out[y0:y0+CH, x0:x0+CW] = CHAR_CACHE[char]

    rgb = np.stack([out, out, out], axis=2)

    if rgb.shape[1] % 2 == 1:
        rgb = rgb[:, :-1]
    if rgb.shape[0] % 2 == 1:
        rgb = rgb[:-1, :]

    return rgb

def ascii_video_to_mp4(video_path, output_path, ascii_width=100, invert=False):
    """transfer video to ASCII mp4"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Total frames = {total}")
    print(f"FPS = {fps}")
    print("Preparing for converting...")

    ret, frame = cap.read()
    if not ret:
        print("Failed to read first frame.")
        cap.release()
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ascii_img = frame_to_ascii(frame, ascii_width, invert)
    out_img = ascii_to_image(ascii_img)
    out_h, out_w, _ = out_img.shape

    print(f"ASCII frame size = {out_w}x{out_h}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}",
        "-r", str(fps),
        "-i", "-",
        "-i", video_path,
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-shortest",
        output_path,
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    pbar = tqdm(total=total or None, desc="Rendering ASCII", dynamic_ncols=True)
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ascii_img = frame_to_ascii(frame, ascii_width, invert)
        out_img = ascii_to_image(ascii_img)

        if out_img.shape[0] != out_h or out_img.shape[1] != out_w:
            out_img = cv2.resize(out_img, (out_w, out_h))

        proc.stdin.write(out_img.tobytes())

        processed += 1
        if total > 0:
            pbar.update(1)

    pbar.close()
    cap.release()

    proc.stdin.close()
    proc.wait()

    print("Done:", output_path)

# === main ===
def main():
    global CUDA_AVAILABLE

    if len(sys.argv) < 3:
        print("Usage: python ascii_to_mp4.py input.mp4 output.mp4 [width] [--invert] [--use-cuda]")
        return

    video = sys.argv[1]
    output = sys.argv[2]

    ascii_width = 100
    invert = False
    want_cuda = False

    for a in sys.argv[3:]:
        if a.isdigit():
            ascii_width = int(a)
        elif a == "--invert":
            invert = True
        elif a == "--use-cuda":
            want_cuda = True

    if want_cuda:
        if CUDA_AVAILABLE:
            print("Using CUDA for ASCII conversion.")
        else:
            print("CUDA requested but not available. Use CPU conversion instead...")
    else:
        CUDA_AVAILABLE = False

    ascii_video_to_mp4(video, output, ascii_width, invert)


if __name__ == "__main__":
    main()