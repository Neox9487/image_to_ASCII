import cv2
import os
import sys
import time
import numpy as np
import subprocess

ASCII_CHARS = np.array(list("@#S%?*+;:, "))
WEIGHTS = np.array([0.33, 0.33, 0.33])  # r, g, b weights


def frame_to_ascii(frame, width, invert):
    h, w, _ = frame.shape
    new_h = int(h * (width / w) * 0.55)

    small = cv2.resize(frame, (width, new_h))

    lum = (small * WEIGHTS).sum(axis=2)

    if invert:
        lum = 255 - lum

    idx = (lum * (len(ASCII_CHARS)-1) / 255).astype(np.uint8)
    return ASCII_CHARS[idx]


def ascii_to_image(ascii_img):
    line_h = 12
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1

    img_h = ascii_img.shape[0] * line_h
    img_w = ascii_img.shape[1] * 8

    out = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    for i, row in enumerate(ascii_img):
        cv2.putText(out, "".join(row),
                    (2, (i+1)*line_h),
                    font, font_scale, (255,255,255),
                    thickness, cv2.LINE_AA)

    return out


def ascii_video_to_mp4(video_path, output_path, width=100, invert=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Converting video to ASCII frames...")

    tmp_dir = "_ascii_frames"
    os.makedirs(tmp_dir, exist_ok=True)

    start_time = time.time()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ascii_img = frame_to_ascii(frame, width, invert)

        img = ascii_to_image(ascii_img)
        cv2.imwrite(f"{tmp_dir}/{frame_id:05d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])

        frame_id += 1

        percent = frame_id / total_frames
        bar_len = 30
        bar = "#" * int(percent * bar_len)
        bar = bar.ljust(bar_len, '-')

        elapsed = time.time() - start_time
        eta = (elapsed / percent - elapsed) if percent > 0 else 0

        print(
            f"\r[{bar}] {percent*100:5.1f}%  {frame_id}/{total_frames}  "
            f"Elapsed: {elapsed:5.1f}s  ETA: {eta:5.1f}s",
            end=""
        )

    cap.release()
    print("\nEncoding MP4 with ffmpeg...")

    os.system(
        f'ffmpeg -y -framerate {fps} -i "{tmp_dir}/%05d.jpg" '
        f'-pix_fmt yuv420p "{output_path}"'
    )

    # cleanup
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)

    print("Done! Output saved:", output_path)


def main():
    if len(sys.argv) < 3:
        print("Usage: python ascii_to_mp4.py input.mp4 output.mp4 [width] [--invert]")
        return

    video = sys.argv[1]
    output = sys.argv[2]

    width = 100
    invert = False

    for a in sys.argv[3:]:
        if a.isdigit():
            width = int(a)
        elif a == "--invert":
            invert = True

    ascii_video_to_mp4(video, output, width, invert)


if __name__ == "__main__":
    main()
