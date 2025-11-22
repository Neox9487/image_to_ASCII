import cv2
import os
import sys
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import subprocess

ASCII_CHARS = np.array(list("@#S%?*+;:, "))
WEIGHTS = np.array([0.33, 0.33, 0.33])

def find_best_font():
    preferred = [
        "DejaVuSansMono.ttf",
        "LiberationMono-Regular.ttf",
        "NotoMono-Regular.ttf",
    ]

    try:
        result = subprocess.check_output(
            ["fc-list", ":spacing=mono", "file"],
            universal_newlines=True
        )
        fonts_raw = [line.strip() for line in result.split("\n") if line.strip()]
    except:
        fonts_raw = []

    fonts = []
    for f in fonts_raw:
        real_path = f.split(":")[0]
        if os.path.isfile(real_path):
            fonts.append(real_path)

    for p in preferred:
        for f in fonts:
            if p in f:
                print("Using font:", f)
                return f

    if fonts:
        print("Using fallback font:", fonts[0])
        return fonts[0]

    print("No monospace font found!")
    print("Install a monospace font:")
    print("sudo apt install fonts-dejavu-core")
    exit(1)

FONT_PATH = find_best_font()
FONT_SIZE = 14
FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)

BBOX = FONT.getbbox("A")
TEXT_W = BBOX[2] - BBOX[0]
TEXT_H = BBOX[3] - BBOX[1]
CHAR_RATIO = TEXT_H / TEXT_W


def frame_to_ascii(frame, width, invert):
    h, w, _ = frame.shape
    new_h = int(h * (width / w) / CHAR_RATIO)

    small = cv2.resize(frame, (width, new_h))
    lum = (small * WEIGHTS).sum(axis=2)

    if invert:
        lum = 255 - lum

    idx = (lum * (len(ASCII_CHARS) - 1) / 255).astype(np.uint8)

    return ASCII_CHARS[idx]


def ascii_to_image(ascii_img):
    h, w = ascii_img.shape
    img_w = w * TEXT_W
    img_h = h * TEXT_H

    img = Image.new("RGB", (img_w, img_h), "black")
    draw = ImageDraw.Draw(img)

    for i, row in enumerate(ascii_img):
        draw.text((0, i * TEXT_H), "".join(row), font=FONT, fill="white")

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def ascii_video_to_mp4(video_path, output_path, width=100, invert=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Converting video to ASCII frames...")

    tmp_dir = "_ascii_frames"
    os.makedirs(tmp_dir, exist_ok=True)

    start = time.time()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ascii_img = frame_to_ascii(frame, width, invert)
        img = ascii_to_image(ascii_img)

        cv2.imwrite(
            f"{tmp_dir}/{frame_id:05d}.jpg",
            img,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        frame_id += 1

        percent = frame_id / total
        bar = "#" * int(percent * 30)
        bar = bar.ljust(30, "-")

        elapsed = time.time() - start
        eta = (elapsed / percent - elapsed) if percent > 0 else 0

        print(
            f"\r[{bar}] {percent*100:5.1f}%  "
            f"{frame_id}/{total}  "
            f"Elapsed: {elapsed:5.1f}s  ETA: {eta:5.1f}s",
            end=""
        )

    cap.release()
    print("\nEncoding MP4...")

    os.system(
        f'ffmpeg -y -framerate {fps} -i "{tmp_dir}/%05d.jpg" '
        f'-pix_fmt yuv420p "{output_path}"'
    )

    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)

    print("Done:", output_path)


def main():
    if len(sys.argv) < 3:
        print("Usage: python ascii_to_mp4.py input.mp4 output.mp4 [width] [--invert]")
        return

    video = sys.argv[1]
    output = sys.argv[2]

    width = 100
    invert = False

    for a in sys.argv[3:]:
        if a.isdigit(): width = int(a)
        elif a == "--invert": invert = True

    ascii_video_to_mp4(video, output, width, invert)


if __name__ == "__main__":
    main()
