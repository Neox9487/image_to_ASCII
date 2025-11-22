import cv2
import os
import sys
import time
import numpy as np
from PIL import ImageFont, Image, ImageDraw

# !

ASCII = np.array(list("@#S%?*+;:, "))
WEIGHTS = np.array([0.33, 0.33, 0.33])

def find_mono_font():
    fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
    ]
    for f in fonts:
        if os.path.exists(f):
            return f
    print("Can't find monospace fonts")
    sys.exit(1)

FONT_PATH = find_mono_font()
FONT_SIZE = 14
FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)

bbox = FONT.getbbox("A")
CW = bbox[2] - bbox[0] 
CH = bbox[3] - bbox[1]  

def frame_to_ascii(frame, ascii_width, invert):
    H, W, _ = frame.shape

    ascii_height = int(H / W * ascii_width * (CW / CH))

    small = cv2.resize(frame, (ascii_width, ascii_height))
    lum = (small * WEIGHTS).sum(axis=2)

    if invert:
        lum = 255 - lum

    idx = (lum * (len(ASCII)-1) / 255).astype(np.uint8)
    return ASCII[idx]

def ascii_to_image(ascii_img):
    h, w = ascii_img.shape

    out_w = w * CW
    out_h = h * CH

    img = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    for i, row in enumerate(ascii_img):
        draw.text((0, i * CH), "".join(row), font=FONT, fill=(255, 255, 255))

    return np.array(img)

def ascii_video_to_mp4(video_path, output_path, ascii_width=100, invert=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Converting video to ASCII frames...")

    tmp = "_ascii_frames"
    os.makedirs(tmp, exist_ok=True)

    start = time.time()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ascii_img = frame_to_ascii(frame, ascii_width, invert)
        out_img = ascii_to_image(ascii_img)

        cv2.imwrite(f"{tmp}/{frame_id:05d}.jpg", out_img, [cv2.IMWRITE_JPEG_QUALITY, 70])

        frame_id += 1

        percent = frame_id / total
        bar = "#" * int(percent * 30)
        bar = bar.ljust(30, "-")

        elapsed = time.time() - start
        eta = (elapsed / percent - elapsed) if percent > 0 else 0

        print(f"\r[{bar}] {percent*100:5.1f}%  {frame_id}/{total}  "
              f"Elapsed: {elapsed:5.1f}s  ETA: {eta:5.1f}s",
              end="")

    cap.release()

    print("\nEncoding MP4...")
    os.system(
        f'ffmpeg -y -framerate {fps} -i "{tmp}/%05d.jpg" '
        f'-pix_fmt yuv420p "{output_path}"'
    )

    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)

    print("Done:", output_path)

def main():
    if len(sys.argv) < 3:
        print("Usage: python ascii_to_mp4.py input.mp4 output.mp4 [width] [--invert]")
        return

    video = sys.argv[1]
    output = sys.argv[2]

    ascii_width = 100
    invert = False

    for a in sys.argv[3:]:
        if a.isdigit():
            ascii_width = int(a)
        elif a == "--invert":
            invert = True

    ascii_video_to_mp4(video, output, ascii_width, invert)


if __name__ == "__main__":
    main() 
