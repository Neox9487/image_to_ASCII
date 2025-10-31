from PIL import Image
import sys
import os

ASCII_CODE = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", " "]
WEIGHTS = [0.42, 0.11, 0.47]

def rgb_to_weighted_luminance(r, g, b, r_w=0.33, g_w=0.33, b_w=0.33):
    """
    計算 RGB 加權亮度
    """
    return r * r_w + g * g_w + b * b_w

def image_to_ascii(img_path, width=80, scale=0.55, invert=False,r_w=0.33, g_w=0.33, b_w=0.33):
    """
    參數:
      img_path: 圖片檔路徑
      width: 輸出列寬（字元數）
      scale: 縮放（字元通常比寬窄）
      invert: 是否反相（亮/暗互換）
      r_w, g_w, b_w: RGB 權重
    回傳: ASCII 字串（包含換行）
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"找不到檔案: {img_path}")

    img = Image.open(img_path).convert("RGB")
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


def main():
    if len(sys.argv) < 2:
        print("python ascii_art.py <image_path> [width] [--invert] [--out output.txt]")
        return

    img_path = sys.argv[1]
    width = 80
    invert = False
    out_file = None

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        a = args[i]
        if a.isdigit():
            width = int(a)
        elif a == "--invert":
            invert = True
        elif a == "--out":
            i += 1
            if i < len(args):
                out_file = args[i]
            else:
                print("錯誤! --out 後面要接檔名")
                return
        else:
            print(f"未知參數: {a}")
            return
        i += 1

    ascii_art = image_to_ascii(img_path, width=width, invert=invert, r_w=WEIGHTS[0], g_w=WEIGHTS[1], b_w=WEIGHTS[2])
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(ascii_art)
        print(f"已儲存 ASCII 圖到: {out_file}")
    else:
        print(ascii_art)


if __name__ == "__main__":
    main()