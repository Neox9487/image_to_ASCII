# Usage
## Image transfer
```bash
python ascii_art.py <image_path> [width] [--invert] [--out output.txt]
```

| Parameter            | Description                                            |
| -------------------- | ------------------------------------------------------ |
| `<image_path>`       | Path to the input image (`.jpg`, `.png`, `.bmp`, etc.) |
| `[width]`            | Output ASCII width (default: `80`)                     |
| `--invert`           | Invert brightness (light ↔ dark)                       |
| `--out <output.txt>` | Save the ASCII result to a specified text file         |

## Video transfer
```bash
python ascii_to_mp4.py input.mp4 output.mp4 [width] [--invert] [--color] [--vibrance value] [--gamma value]
```

| Parameter      | Description                        |
| -------------- | ---------------------------------- |
| `<input.mp4>`  | Input video to convert             |
| `<output.mp4>` | Name of the output video file      |
| `[width]`      | Output ASCII width (default: `80`) |
| `--invert`     | Invert brightness (light ↔ dark)   |
| `--color`      | Apply color shading                |
| `--vibrance`   | Adjust color intensity/saturation (default: 1.0) |
| `--gamma`      | Adjust color contrast/brightness curve (default: 1.0) |

> You can modify WEIGHTS to adjust colors weighting.