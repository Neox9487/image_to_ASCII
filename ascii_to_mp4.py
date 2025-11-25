import cv2
import os
import sys
import numpy as np
import subprocess
from PIL import ImageFont, Image, ImageDraw
from tqdm import tqdm

# === CUDA ===
CUDA_AVAILABLE = False
cp = None
try:
    import cupy as _cp
    if _cp.is_available():
        cp = _cp
        CUDA_AVAILABLE = True
except:
    CUDA_AVAILABLE = False

print("CUDA available:", CUDA_AVAILABLE)

# === ASCII codes setup ===
ASCII = np.frombuffer(b"@#S%?*+;:, ", dtype=np.uint8)
NCH = len(ASCII)
WEIGHTS = np.array([0.33, 0.33, 0.33], dtype=np.float32)

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
asc, desc = FONT.getmetrics()
CH = asc + desc
CW = int(FONT.getlength("A"))

# === Pre-render glyphs ===
CHAR_CACHE = np.zeros((NCH, CH, CW), dtype=np.uint8)
for i, ch in enumerate(ASCII):
    img = Image.new("L", (CW, CH), 0)
    d = ImageDraw.Draw(img)
    d.text((0, 0), chr(ch), font=FONT, fill=255)
    CHAR_CACHE[i] = np.array(img, dtype=np.uint8)

CHAR_CACHE_GPU = None
WEIGHTS_GPU = None

def ensure_gpu():
    global CHAR_CACHE_GPU, WEIGHTS_GPU
    if CHAR_CACHE_GPU is None:
        CHAR_CACHE_GPU = cp.asarray(CHAR_CACHE)
    if WEIGHTS_GPU is None:
        WEIGHTS_GPU = cp.asarray(WEIGHTS)

# === GPU-only resize ===
resize_kernel = cp.ElementwiseKernel(
    "raw uint8 src, int32 tw, int32 th, float32 fw, float32 fh",
    "uint8 dst",
    """
    // idk what are these
    int x = i % tw;
    int y = i / tw;

    float gx = x * fw;
    float gy = y * fh;

    int ix = (int)gx;
    int iy = (int)gy;

    if (ix >= tw) ix = tw - 1;
    if (iy >= th) iy = th - 1;

    dst = src[iy * tw + ix];
    """,
    "gpu_resize"
)

def gpu_resize_img(img, tw, th):
    h, w = img.shape[:2]
    fw = cp.float32(w / tw)
    fh = cp.float32(h / th)

    out = cp.empty((th, tw, 3), dtype=cp.uint8)

    for c in range(3):
        resize_kernel(
            img[:, :, c].ravel(),
            tw,
            th,
            fw,
            fh,
            out[:, :, c].ravel()
        )
    return out

# === Conversion ===
def frame_to_ascii(frame, ascii_width, invert):
    H, W, _ = frame.shape
    aspect = H / W
    ascii_height = int(ascii_width * aspect * (CW / CH))

    ensure_gpu()
    g = cp.asarray(frame, dtype=cp.float32)

    r = gpu_resize_img(g.astype(cp.uint8), ascii_width, ascii_height).astype(cp.float32)

    lum = (r * WEIGHTS_GPU).sum(axis=2)

    if invert:
        lum = 255 - lum

    idx = (lum * (NCH - 1) / 255).astype(cp.int32)
    return idx

def ascii_to_image(idx):
    ensure_gpu()
    h, w = idx.shape
    tiles = CHAR_CACHE_GPU[idx]
    tiles = tiles.transpose(0, 2, 1, 3)
    gray = tiles.reshape(h * CH, w * CW)
    rgb = cp.stack([gray, gray, gray], axis=2)
    if rgb.shape[0] % 2 == 1:
        rgb = rgb[:-1]
    if rgb.shape[1] % 2 == 1:
        rgb = rgb[:, :-1]
    return rgb.astype(cp.uint8)

def ascii_video_to_mp4(video_path, output_path, ascii_width=100, invert=False):
    if not CUDA_AVAILABLE:
        print("CUDA not available. Cannot run GPU ASCII conversion.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Total frames =", total)
    print("FPS =", fps)
    print("Preparing for converting...")

    ret, frame = cap.read()
    if not ret:
        print("Failed to read first frame.")
        cap.release()
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    idx0 = frame_to_ascii(frame, ascii_width, invert)
    out0_gpu = ascii_to_image(idx0)
    out0 = cp.asnumpy(out0_gpu)
    oh, ow, _ = out0.shape

    print(f"ASCII frame size = {ow}x{oh}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{ow}x{oh}",
        "-r", str(fps),
        "-i", "-",
        "-i", video_path,
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-b:v", "5M",
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx = frame_to_ascii(frame, ascii_width, invert)
        img_gpu = ascii_to_image(idx)
        img = cp.asnumpy(img_gpu)

        if img.shape[0] != oh or img.shape[1] != ow:
            img = cv2.resize(img, (ow, oh))

        proc.stdin.write(img.tobytes())
        pbar.update(1)

    pbar.close()
    cap.release()
    proc.stdin.close()
    proc.wait()

    print("Done:", output_path)

# === Main ===
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
