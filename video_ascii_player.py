import cv2
import os
import sys
import time
import subprocess
from PIL import Image, ImageDraw, ImageFont

ASCII_CODE = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", " "]
WEIGHTS = [0.33, 0.33, 0.33]

def rgb_to_weighted_luminance(r,g,b,rw,gw,bw):
    return r*rw + g*gw + b*bw

def frame_to_ascii(frame,width,invert,rw,gw,bw):
    img = Image.fromarray(frame).convert("RGB")
    ow,oh = img.size
    nh = int((oh/ow)*width*0.55)
    img = img.resize((width,nh))
    px = img.load()
    h = img.height
    w = img.width
    b = len(ASCII_CODE)-1
    lines=[]
    for y in range(h):
        row=[]
        for x in range(w):
            R,G,B = px[x,y]
            lum = rgb_to_weighted_luminance(R,G,B,rw,gw,bw)
            if invert: lum = 255-lum
            row.append(ASCII_CODE[int(lum*b/255)])
        lines.append("".join(row))
    return "\n".join(lines)

def ascii_to_image(ascii_text, font, bg=(0,0,0), fg=(255,255,255)):
    lines = ascii_text.split("\n")
    w = max(font.getlength(line) for line in lines)
    h = (len(lines)) * (font.size + 1)

    img = Image.new("RGB", (int(w)+10, int(h)+10), bg)
    draw = ImageDraw.Draw(img)

    y = 5
    for line in lines:
        draw.text((5, y), line, font=font, fill=fg)
        y += font.size + 1
    return img

def ascii_video_to_mp4(video_path, out_path, width=100, invert=False):
    if not os.path.exists(video_path):
        print("Video not found", video_path)
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30

    tmp_dir = "_ascii_frames"
    os.makedirs(tmp_dir, exist_ok=True)

    font = ImageFont.load_default()

    print("Converting video to ASCII frames...")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ascii_text = frame_to_ascii(frame, width, invert, *WEIGHTS)

        img = ascii_to_image(ascii_text, font)
        img.save(f"{tmp_dir}/{frame_id:05d}.png")

        frame_id += 1

        percent = frame_id / total_frames
        bar = "#" * int(percent * 30)
        bar = bar.ljust(30, '-')

        elapsed = time.time() - start_time
        eta = elapsed / percent - elapsed if percent > 0 else 0

        print(f"\r[{bar}] {percent*100:5.1f}%  "
            f"{frame_id}/{total_frames}  "
            f"Elapsed: {elapsed:5.1f}s  ETA: {eta:5.1f}s",
            end="")

    cap.release()

    print("Encoding MP4 via ffmpeg...")

    cmd = (
        f'ffmpeg -y -framerate {fps} -i "{tmp_dir}/%05d.png" '
        f'-pix_fmt yuv420p "{out_path}"'
    )
    os.system(cmd)

    print("Cleaning temp files...")
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)

    print("Done! Output saved:", out_path)

def main():
    if len(sys.argv) < 3:
        print("Usage: python ascii_to_mp4.py input.mp4 output_ascii.mp4 [width] [--invert]")
        return

    input_video = sys.argv[1]
    output_video = sys.argv[2]
    width = 100
    invert = False

    args = sys.argv[3:]
    for a in args:
        if a.isdigit():
            width = int(a)
        elif a == "--invert":
            invert = True

    ascii_video_to_mp4(input_video, output_video, width, invert)

if __name__ == "__main__":
    main()
