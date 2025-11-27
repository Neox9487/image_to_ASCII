import cv2
import os
import sys
import numpy as np
import subprocess
import platform
from PIL import ImageFont, Image, ImageDraw
from tqdm import tqdm
import queue
import threading

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    print("Numba available: True")
except Exception:
    NUMBA_AVAILABLE = False
    print("Numba available: False")

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

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

ASCII = np.frombuffer(b" .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$", dtype=np.uint8)
NCH = len(ASCII)
LUT_LUM_TO_ASCII = ((np.arange(256) / 255) * (NCH - 1)).astype(np.uint8)

CHAR_CACHE = np.zeros((NCH, CH, CW), dtype=np.uint8)
for i, ch in enumerate(ASCII):
    img = Image.new("L", (CW, CH), 0)
    d = ImageDraw.Draw(img)
    d.text((0, 0), chr(ch), font=FONT, fill=255)
    CHAR_CACHE[i] = np.array(img, dtype=np.uint8)

if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def ascii_to_color_numba(idx, small_rgb, char_cache):
        h, w = idx.shape
        ch = char_cache.shape[1]
        cw = char_cache.shape[2]

        out_h = h * ch
        out_w = w * cw
        out = np.empty((out_h, out_w, 3), np.uint8)

        for i in prange(h):
            base_y = i * ch
            for j in range(w):
                base_x = j * cw
                char_idx = idx[i, j]
                tile = char_cache[char_idx]
                r = small_rgb[i, j, 0]
                g = small_rgb[i, j, 1]
                b = small_rgb[i, j, 2]

                for yy in range(ch):
                    for xx in range(cw):
                        v = tile[yy, xx]
                        out[base_y+yy, base_x+xx, 0] = (v * r) >> 8
                        out[base_y+yy, base_x+xx, 1] = (v * g) >> 8
                        out[base_y+yy, base_x+xx, 2] = (v * b) >> 8

        return out

    @njit(parallel=True, fastmath=True)
    def rgb_to_ascii_idx_numba(small, invert, lut):
        h, w, _ = small.shape
        out = np.empty((h, w), np.uint8)
        for i in prange(h):
            for j in range(w):
                r = small[i, j, 0]
                g = small[i, j, 1]
                b = small[i, j, 2]
                lum = (r * 77 + g * 150 + b * 29) >> 8
                if invert:
                    lum = 255 - lum
                out[i, j] = lut[lum]
        return out

    @njit(parallel=True, fastmath=True)
    def ascii_to_gray_numba(idx, char_cache):
        h, w = idx.shape
        ch = char_cache.shape[1]
        cw = char_cache.shape[2]
        out_h = h * ch
        out_w = w * cw
        out = np.empty((out_h, out_w), np.uint8)

        ch4 = ch >> 2
        cw4 = cw >> 2
        ch_rem = ch & 3
        cw_rem = cw & 3

        for i in prange(h):
            base_y = i * ch
            for j in range(w):
                tile = char_cache[idx[i, j]]
                base_x = j * cw

                for yy in range(ch4):
                    y0 = yy * 4
                    dy0 = base_y + y0
                    dy1 = dy0 + 1
                    dy2 = dy0 + 2
                    dy3 = dy0 + 3

                    for xx in range(cw4):
                        x0 = xx * 4
                        dx0 = base_x + x0

                        out[dy0, dx0]     = tile[y0,     x0]
                        out[dy0, dx0 + 1] = tile[y0,     x0 + 1]
                        out[dy0, dx0 + 2] = tile[y0,     x0 + 2]
                        out[dy0, dx0 + 3] = tile[y0,     x0 + 3]

                        out[dy1, dx0]     = tile[y0 + 1, x0]
                        out[dy1, dx0 + 1] = tile[y0 + 1, x0 + 1]
                        out[dy1, dx0 + 2] = tile[y0 + 1, x0 + 2]
                        out[dy1, dx0 + 3] = tile[y0 + 1, x0 + 3]

                        out[dy2, dx0]     = tile[y0 + 2, x0]
                        out[dy2, dx0 + 1] = tile[y0 + 2, x0 + 1]
                        out[dy2, dx0 + 2] = tile[y0 + 2, x0 + 2]
                        out[dy2, dx0 + 3] = tile[y0 + 2, x0 + 3]

                        out[dy3, dx0]     = tile[y0 + 3, x0]
                        out[dy3, dx0 + 1] = tile[y0 + 3, x0 + 1]
                        out[dy3, dx0 + 2] = tile[y0 + 3, x0 + 2]
                        out[dy3, dx0 + 3] = tile[y0 + 3, x0 + 3]

                if cw_rem != 0:
                    x0 = cw4 * 4
                    for yy in range(ch):
                        dy = base_y + yy
                        for k in range(cw_rem):
                            out[dy, base_x + x0 + k] = tile[yy, x0 + k]

                if ch_rem != 0:
                    y0 = ch4 * 4
                    for k in range(ch_rem):
                        dy = base_y + y0 + k
                        for xx in range(cw):
                            out[dy, base_x + xx] = tile[y0 + k, xx]

        return out

    @njit(parallel=True, fastmath=True)
    def resize_nearest_numba(img, new_h, new_w):
        h, w, c = img.shape
        out = np.empty((new_h, new_w, c), dtype=np.uint8)
        r_h = h / new_h
        r_w = w / new_w
        for i in prange(new_h):
            for j in range(new_w):
                y = int(i * r_h)
                x = int(j * r_w)
                out[i, j] = img[y, x]
        return out

    _dummy = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb_to_ascii_idx_numba(_dummy, False, LUT_LUM_TO_ASCII)
    _dummy_idx = np.zeros((10, 10), dtype=np.uint8)
    ascii_to_gray_numba(_dummy_idx, CHAR_CACHE)
    _dummy_rgb = np.zeros((10,10,3),dtype=np.uint8)
    ascii_to_color_numba(_dummy_idx, _dummy_rgb, CHAR_CACHE)
    resize_nearest_numba(_dummy, 5, 5)

else:

    def resize_nearest_numba(img, new_h, new_w):
        return cv2.resize(img, (new_w, new_h))

def rgb_to_ascii_np(small, invert):
    lum = (small[:, :, 0] * 77 + small[:, :, 1] * 150 + small[:, :, 2] * 29) >> 8
    if invert:
        lum = 255 - lum
    return LUT_LUM_TO_ASCII[lum]

def ascii_to_image_color(idx, small_rgb):
    h, w = idx.shape
    out_h = h * CH
    out_w = w * CW

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            char_idx = idx[i, j]
            tile = CHAR_CACHE[char_idx]
            color = small_rgb[i, j]

            tile_rgb = (tile[:, :, None] * color[None, None, :] / 255).astype(np.uint8)

            out[i*CH:(i+1)*CH, j*CW:(j+1)*CW] = tile_rgb

    return out

def frame_to_ascii(frame, ascii_w, ascii_h, invert=False):
    small = resize_nearest_numba(frame, ascii_h, ascii_w)
    if NUMBA_AVAILABLE:
        return rgb_to_ascii_idx_numba(small, invert, LUT_LUM_TO_ASCII)
    return rgb_to_ascii_np(small, invert)

def ascii_to_image(idx):
    if NUMBA_AVAILABLE:
        gray = ascii_to_gray_numba(idx, CHAR_CACHE)
        return np.stack([gray, gray, gray], axis=2).astype(np.uint8)
    h, w = idx.shape
    out = np.zeros((h * CH, w * CW), dtype=np.uint8)
    for y in range(h):
        tile_row = CHAR_CACHE[idx[y]]
        tile_row = tile_row.transpose(1, 0, 2)
        out[y * CH:(y + 1) * CH, :] = tile_row.reshape(CH, w * CW)
    return np.stack([out, out, out], axis=2).astype(np.uint8)

def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 30.0, 0, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total, w, h

def compute_ascii_dims(vw, vh, ascii_width):
    aspect = vh / vw
    ascii_height = int(ascii_width * aspect * (CW / CH))
    if ascii_height <= 0:
        ascii_height = 1
    out_w = ascii_width * CW
    out_h = ascii_height * CH
    if out_w % 2 != 0:
        out_w += CW
        ascii_width += 1
    if out_h % 2 != 0:
        out_h += CH
        ascii_height += 1
    return ascii_width, ascii_height, out_w, out_h

def open_ffmpeg_encode(video_path, output_path, ow, oh, fps):
    cmd = [
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
        "-c:v", "libx264",
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

def ascii_video_to_mp4(video_path, output_path, ascii_width=100, invert=False, color = False):
    fps, total, vw, vh = get_video_info(video_path)
    if vw == 0 or vh == 0:
        print("Failed to read video:", video_path)
        return

    ascii_w, ascii_h, ow, oh = compute_ascii_dims(vw, vh, ascii_width)

    print("Mode:", "Multithread CPU + Numba" if NUMBA_AVAILABLE else "Multithread CPU")
    print("Frames:", total)
    print("ASCII grid:", ascii_w, "x", ascii_h)

    decode_q = queue.Queue(maxsize=16)
    encode_q = queue.Queue(maxsize=16)
    stop_signal = object()

    def decode_thread():
        cap = cv2.VideoCapture(video_path)
        fid = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            decode_q.put((fid, frame))
            fid += 1
        cap.release()

        for _ in range(NUM_WORKERS):
            decode_q.put(stop_signal)

    def worker_thread():
        while True:
            item = decode_q.get()
            if item is stop_signal:
                encode_q.put(stop_signal)
                return
            
            fid, frame = item
            small_rgb = resize_nearest_numba(frame, ascii_h, ascii_w)

            if NUMBA_AVAILABLE:
                idx = rgb_to_ascii_idx_numba(small_rgb, invert, LUT_LUM_TO_ASCII)
            else:
                idx = rgb_to_ascii_np(small_rgb, invert)

            if color and NUMBA_AVAILABLE:
                out_img = ascii_to_color_numba(idx, small_rgb, CHAR_CACHE)
            elif color:
                out_img = ascii_to_image_color(idx, small_rgb) 
            else:
                out_img = ascii_to_image(idx)

            encode_q.put((fid, out_img))
    
    def encode_thread():
        proc = open_ffmpeg_encode(video_path, output_path, ow, oh, fps)
        next_id = 0
        buffer = {}

        stop_count = 0

        while True:
            item = encode_q.get()
            if item is stop_signal:
                stop_count += 1
                if stop_count == NUM_WORKERS:
                    break
                continue

            fid, img = item
            buffer[fid] = img

            while next_id in buffer:
                proc.stdin.write(buffer[next_id].tobytes())
                del buffer[next_id]
                next_id += 1
                pbar.update(1)

        while next_id in buffer:
            proc.stdin.write(buffer[next_id].tobytes())
            del buffer[next_id]
            next_id += 1
            pbar.update(1)

        proc.stdin.close()
        proc.wait()


    NUM_WORKERS = os.cpu_count()
    pbar = tqdm(total=total, desc="Rendering ASCII")

    threads = []
    threads.append(threading.Thread(target=decode_thread))
    for _ in range(NUM_WORKERS):
        threads.append(threading.Thread(target=worker_thread))
    threads.append(threading.Thread(target=encode_thread))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    pbar.close()
    print("Done:", output_path)

def main():
    if len(sys.argv) < 3:
        print("Usage: python ascii_to_mp4.py input.mp4 output.mp4 [width] [--invert] [--color]")
        return
    video = sys.argv[1]
    output = sys.argv[2]
    ascii_width = 100
    invert = False
    color = False
    for a in sys.argv[3:]:
        if a.isdigit():
            ascii_width = int(a)
        elif a == "--invert":
            invert = True
        elif a == "--color":
            color = True
    ascii_video_to_mp4(video, output, ascii_width, invert, color)

if __name__ == "__main__":
    main()