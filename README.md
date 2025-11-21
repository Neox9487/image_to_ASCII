# 使用方法
## Image transfer
```bash
python ascii_art.py <image_path> [width] [--invert] [--out output.txt]
```

| 參數                   | 說明                                   |
| -------------------- | ------------------------------------ |
| `<image_path>`       | 圖片路徑（支援 `.jpg`, `.png`, `.bmp`, ...） |
| `[width]`            | ASCII 輸出寬度（預設 `80`）                  |
| `--invert`           | 反相顯示（亮 ~ 暗）                          |
| `--out <output.txt>` | 將結果輸出到指定文字檔                       |

## Video player
```bash
python video_ascii_player.py <video_path> [width] [--invert]
```

| 參數                   | 說明                                   |
| -------------------- | ------------------------------------ |
| `<video_path>`       | 影片路徑（支援 `mp4`）                       |
| `[width]`            | ASCII 輸出寬度（預設 `80`）                  |
| `--invert`           | 反相顯示（亮 ~ 暗）                          |

> 可以更改 `ascii_art.py` 內的 WEIGHTS 來改變顏色權重