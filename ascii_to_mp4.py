import cv2
import os
import sys
import time
import numpy as np
from PIL import ImageFont, Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm 

# ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", " "]

ASCII = np.array(list("@#S%?*+;:, "))
WEIGHTS = np.array([0.33, 0.33, 0.33])  # r, g, b weights

def find_mono_font():
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

def frame_to_ascii(frame, ascii_width, invert):
    H, W, _ = frame.shape
    aspect = H / W
    ascii_height = int(ascii_width * aspect * (CW / CH))

    small = cv2.resize(frame, (ascii_width, ascii_height))
    lum = (small * WEIGHTS).sum(axis=2)

    if invert:
        lum = 255 - lum

    idx = (lum * (len(ASCII) - 1) / 255).astype(np.uint8)
    return ASCII[idx]

def ascii_to_image(ascii_img):
    h, w = ascii_img.shape

    out_w = w * CW
    out_h = h * CH

    img = Image.new("RGB", (out_w, out_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    for i, row in enumerate(ascii_img):
        draw.text((0, i * CH), "".join(row), font=FONT, fill=(255, 255, 255))

    np_img = np.array(img)

    H, W, _ = np_img.shape
    if W % 2 == 1:
        np_img = np_img[:, :-1]
    if H % 2 == 1:
        np_img = np_img[:-1, :]

    return np_img

def process_one_frame(args):
    frame_id, frame, ascii_width, invert, tmp = args

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ascii_img = frame_to_ascii(frame, ascii_width, invert)
    out_img = ascii_to_image(ascii_img)

    out_path = f"{tmp}/{frame_id:05d}.png"
    cv2.imwrite(out_path, out_img)

    return frame_id

def ascii_video_to_mp4(video_path, output_path, ascii_width=100, invert=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Total frames = {total}")
    print(f"FPS = {fps}")
    print("Preparing for converting...")

    frames = []
    for i in tqdm(range(total), desc="Loading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((i, frame))

    cap.release()

    print("Processing frames...")
    tmp = "_ascii_frames"
    os.makedirs(tmp, exist_ok=True)

    job_args = [(i, frame, ascii_width, invert, tmp) for i, frame in frames]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        results = list(tqdm(
            exe.map(process_one_frame, job_args),
            total=len(job_args),
            desc="Rendering ASCII"
        ))

    print("Encoding MP4...")

    os.system(
        f'ffmpeg -y -thread_queue_size 1024 -framerate {fps} -i "{tmp}/%05d.png" '
        f'-thread_queue_size 1024 -i "{video_path}" '
        f'-map 0:v -map 1:a? '
        f'-vcodec libx264 -pix_fmt yuv420p -shortest "{output_path}"'
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
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()