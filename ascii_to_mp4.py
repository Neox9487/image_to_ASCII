import cv2
import os
import sys
import numpy as np
import subprocess
import platform
from PIL import ImageFont, Image, ImageDraw
from tqdm import tqdm

# === Detect OS ===
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

# === CUDA ===
CUDA_AVAILABLE = False
CUDA_LOGGING = ""
cp = None
FORCE_CPU = False

try:
    import cupy as _cp
    from cupyx.scipy.ndimage import zoom
except Exception as e:
    CUDA_LOGGING = f"cupy import failed: {e}"
    CUDA_AVAILABLE = False
else:
    try:
        if _cp.is_available():
            cp = _cp
            CUDA_AVAILABLE = True
        else:
            CUDA_LOGGING = "cupy imported but CUDA backend not available"
            CUDA_AVAILABLE = False
    except Exception as e:
        CUDA_LOGGING = f"cupy CUDA check failed: {e}"
        CUDA_AVAILABLE = False

print("CUDA available:", CUDA_AVAILABLE)
if not CUDA_AVAILABLE:
    print("Fallback to CPU:", CUDA_LOGGING)


# === Monospace font search ===
def find_mono_font():
    linux_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
    ]
    win_fonts = [
        "C:/Windows/Fonts/consola.ttf", 
        "C:/Windows/Fonts/lucon.ttf",  
        "C:/Windows/Fonts/cour.ttf",
    ]

    if IS_LINUX:
        for f in linux_fonts:
            if os.path.exists(f):
                return f

    if IS_WINDOWS:
        for f in win_fonts:
            if os.path.exists(f):
                return f

    print("No monospace font found.")
    sys.exit(1)

FONT_PATH = find_mono_font()
FONT = ImageFont.truetype(FONT_PATH, 14)
asc, desc = FONT.getmetrics()
CH = asc + desc
CW = int(FONT.getlength("A"))

# === ASCII codes ===
ASCII = np.frombuffer(b"@#S%?*+;:, ", dtype=np.uint8)
NCH = len(ASCII)
WEIGHTS = np.array([0.33, 0.33, 0.33], dtype=np.float32)

# === glyph cache ===
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

# === GPU resize kernel ===
if CUDA_AVAILABLE:
    resize_kernel = cp.ElementwiseKernel(
        "raw uint8 src, int32 sw, int32 sh, int32 tw, int32 th",
        "uint8 dst",
        r"""
        // cpp
        int x = i % tw;  
        int y = i / tw;   

        float gx = (float)x * (float)sw / (float)tw;
        float gy = (float)y * (float)sh / (float)th;

        int ix = (int)gx;
        int iy = (int)gy;

        if (ix < 0) ix = 0;
        if (iy < 0) iy = 0;
        if (ix >= sw) ix = sw - 1;
        if (iy >= sh) iy = sh - 1;

        dst = src[iy * sw + ix];
        """,
        "gpu_resize"
    )

def gpu_resize_img(img, tw, th):
    h, w = img.shape[:2]
    g = cp.asarray(img, dtype=cp.uint8)
    zx = th / h
    zy = tw / w
    out = zoom(g, (zx, zy, 1), order=1)
    return out.astype(cp.uint8)


def frame_to_ascii_cpu(frame, ascii_width, invert):
    """convert frame to ASCII via CPU"""
    H, W, _ = frame.shape
    aspect = H / W
    ascii_height = int(ascii_width * aspect * (CW / CH))

    
    small = cv2.resize(frame, (ascii_width, ascii_height))

    lum = (small.astype(np.float32) * WEIGHTS).sum(axis=2)
    if invert:
        lum = 255 - lum
    idx = (lum * (NCH - 1) / 255).astype(np.int32)
    return idx

def frame_to_ascii_gpu(frame, ascii_width, invert):
    """convert frame to ASCII via GPU"""
    H, W, _ = frame.shape
    aspect = H / W
    ascii_height = int(ascii_width * aspect * (CW / CH))

    ensure_gpu()
    g = cp.asarray(frame, dtype=cp.uint8)
    r = gpu_resize_img(g, ascii_width, ascii_height).astype(cp.float32)

    gray = 0.299 * r[:, :, 0] + 0.587 * r[:, :, 1] + 0.114 * r[:, :, 2]

    if invert:
        gray = 255 - gray

    idx = (gray * (NCH - 1) / 255.0).astype(cp.int32)
    idx = cp.clip(idx, 0, NCH - 1)

    return idx

def ascii_to_image_cpu(idx):
    """convert ASCII to image via CPU"""
    h, w = idx.shape
    tiles = CHAR_CACHE[idx]
    tiles = tiles.transpose(0, 2, 1, 3)
    gray = tiles.reshape(h * CH, w * CW)
    rgb = np.stack([gray, gray, gray], axis=2).astype(np.uint8)
    return rgb

def ascii_to_image_gpu(idx):
    """convert ascii to image via GPU"""
    ensure_gpu()
    h, w = idx.shape
    tiles = CHAR_CACHE_GPU[idx]
    tiles = tiles.transpose(0, 2, 1, 3)
    gray = tiles.reshape(h * CH, w * CW)
    rgb = cp.stack([gray, gray, gray], axis=2).astype(cp.uint8)
    return rgb

def get_video_info(path):
    """video info"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 30.0, 0, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total, w, h

def open_ffmpeg_decode(video_path):
    cmd = ["ffmpeg"]
    if CUDA_AVAILABLE:
        cmd += ["-hwaccel", "cuda"]
    else:
        cmd += ["-hwaccel", "auto"]
    cmd += [
        "-i", video_path,
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-"
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8
    )

def open_ffmpeg_encode(video_path, output_path, ow, oh, fps, use_gpu):
    if use_gpu:
        vcodec = "h264_nvenc" 
    else:
        vcodec = "libx264"

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-thread_queue_size", "4096",
        "-s", f"{ow}x{oh}",
        "-r", str(fps),
        "-i", "-",
        "-i", video_path,
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", vcodec,
        "-pix_fmt", "yuv420p",
        "-shortest",
        output_path,
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        bufsize=10**8
    )

def ascii_video_to_mp4(video_path, output_path, ascii_width=100, invert=False):
    """main pipeline"""
    global FORCE_CPU

    fps, total, vw, vh = get_video_info(video_path)
    if vw == 0 or vh == 0:
        print("Failed to read video info:", video_path)
        return

    use_gpu = CUDA_AVAILABLE and not FORCE_CPU

    print("Mode:", "GPU" if use_gpu else "CPU")
    print("Video size =", vw, "x", vh)
    print("FPS =", fps, "Frames =", total)

    if use_gpu:
        proc_in = open_ffmpeg_decode(video_path)
        raw = proc_in.stdout.read(vw * vh * 3)
        if len(raw) < vw * vh * 3:
            print("Failed first frame.")
            proc_in.terminate()
            return
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((vh, vw, 3))
        idx0 = frame_to_ascii_gpu(frame, ascii_width, invert)
        out0 = ascii_to_image_gpu(idx0).get()
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video:", video_path)
            return
        ret, frame = cap.read()
        if not ret:
            print("Failed first frame.")
            cap.release()
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx0 = frame_to_ascii_cpu(frame, ascii_width, invert)
        out0 = ascii_to_image_cpu(idx0)
        cap.release()

    oh, ow, _ = out0.shape
    print("ASCII frame size =", ow, "x", oh)

    def fix_even(x):
        return x if x % 2 == 0 else x + 1

    ow2 = fix_even(ow)
    oh2 = fix_even(oh)

    if ow2 != ow or oh2 != oh:
        out0 = cv2.resize(out0, (ow2, oh2), interpolation=cv2.INTER_NEAREST)
        print(f"Adjusted ASCII frame size to {ow2} x {oh2}")

    proc_out = open_ffmpeg_encode(video_path, output_path, ow2, oh2, fps, use_gpu)
    proc_out.stdin.write(out0.tobytes())

    pbar = tqdm(total=total or None, desc="Rendering ASCII")

    if use_gpu:
        pbar.update(1)
        while True:
            raw = proc_in.stdout.read(vw * vh * 3)
            if len(raw) < vw * vh * 3:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((vh, vw, 3))
            idx = frame_to_ascii_gpu(frame, ascii_width, invert)
            out_img = ascii_to_image_gpu(idx).get()
            out_img = cv2.resize(out_img, (ow2, oh2), interpolation=cv2.INTER_NEAREST)
            proc_out.stdin.write(out_img.tobytes())
            pbar.update(1)
        proc_in.stdout.close()
        proc_in.wait()
    else:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        pbar.update(1)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx = frame_to_ascii_cpu(frame, ascii_width, invert)
            out_img = ascii_to_image_cpu(idx)
            out_img = cv2.resize(out_img, (ow2, oh2), interpolation=cv2.INTER_NEAREST)
            proc_out.stdin.write(out_img.tobytes())
            pbar.update(1)
        cap.release()

    pbar.close()
    proc_out.stdin.close()
    proc_out.wait()
    print("Done:", output_path)

# === Main ===
def main():
    global FORCE_CPU

    if len(sys.argv) < 3:
        print("Usage: python ascii_to_mp4.py input.mp4 output.mp4 [width] [--invert] [--no-cuda]")
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
        elif a == "--no-cuda":
            FORCE_CPU = True

    ascii_video_to_mp4(video, output, ascii_width, invert)

if __name__ == "__main__":
    main()