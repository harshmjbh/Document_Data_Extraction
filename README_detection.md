# Page detection (Option A – Python + OpenCV)

Detect a document/page and draw a bounding box. **Default: Method 1 — U-Net segmentation** (rembg/U²-Net: mask → contours → quad). Optional: Canny+contours (fast) or auto (Canny then U-Net fallback).

For a high-level summary of what was implemented and how outputs are organized, see **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**.

## Setup

```bash
pip install -r requirements.txt
```

Requirements include **rembg[cpu]** for U-Net-style segmentation (default method). First run may download the U²-Net model (~176 MB). For GPU: `pip install "rembg[gpu]"`.

## Scripts

| Script | Input | Output |
|--------|--------|--------|
| **run_detection.py** / **run_detection_images.py** | Image(s) or folder | Image(s) with bounding box |
| **run_detection_video.py** | Video file | **One video file** with boxes on each frame |

---

### Images

**Single image:**

```bash
python run_detection.py photo.jpg
# -> saves photo_with_box.jpg

python run_detection_images.py photo.jpg -o output.png
```

**Batch (folder, same structure):**

```bash
python run_detection.py Test_Images -o Test_Images_Output
python run_detection_images.py Test_Images -o Test_Images_Output --downscale 1024
```

**Method 1 — U-Net segmentation** (default; robust to shadows, similar table/paper, curved spines):

```bash
# U-Net segmentation (default) — mask → findContours → quad
python run_detection.py Test_Images -o Test_Images_Output
python run_detection.py Test_Images -o Test_Images_Output --method unet

# Auto: try Canny first, then U-Net if no quad
python run_detection.py Test_Images -o Test_Images_Output --method auto

# Canny only (fast, can fail on low contrast)
python run_detection.py Test_Images -o Test_Images_Output --method canny
```

- `--method unet` (default): U-Net-style segmentation (rembg/U²-Net); predicts document mask then contours → quad. Industry-standard, no threshold tuning.
- `--method auto`: Canny first, then U-Net fallback if no quad.
- `--method rembg`: Same as unet.
- `--method canny`: OpenCV Canny + contours only; fastest when contrast is high.
- `--fast`: For unet/rembg, skip corner refinement (faster, slightly less accurate box).
- `--downscale MAX_PIX`: Resize so max dimension is MAX_PIX before detection (faster).

---

### Video (file or live camera)

**Video file** → output video with boxes:

```bash
python run_detection_video.py video.mp4
# -> saves video_with_boxes.mp4

python run_detection_video.py video.mp4 -o out.mp4 --downscale 800 --sample-every 3
```

**Live camera (real time):** use camera index `0` (default webcam) or `1`, etc. A window shows the feed with the page box; press **q** to quit. Optionally record with `-o`.

```bash
python run_detection_video.py 0
python run_detection_video.py 0 -o live_recording.mp4 --downscale 640 --fast
```

**Extract each new page as an image** (folder next to the video):

```bash
python run_detection_video.py video.mp4 --extract-pages ./video_pages
# -> video_with_boxes.mp4 + folder video_pages/page_0001.png, page_0002.png, ...

python run_detection_video.py video.mp4 -o out.mp4 --extract-pages
# -> out.mp4 + folder out_pages/ (default name when DIR omitted)
```

**Smoother bounding box (less jitter):**

```bash
python run_detection_video.py video.mp4 --smooth 0.4
```

- `--downscale MAX_PIX`: run detection on smaller frames (faster); box is drawn on full resolution.
- `--sample-every N`: run detection every N frames and reuse the last quad for in-between frames (faster, smoother).
- `--extract-pages [DIR]`: save each new page as page_0001.png, page_0002.png, … in DIR (or `<output_stem>_pages` if DIR omitted). Uses center/area change to detect a new page; `--min-frames-between-pages N` (default 10) avoids duplicates.
- `--smooth ALPHA`: smooth the box over time (ALPHA in (0,1], e.g. 0.4) to reduce jitter.
- `--method unet|auto|canny|rembg`, `--fast`: same as image CLI (default unet; use `--fast` for faster unet/rembg).

---

## Layout

- `page_detection/detector.py` – `detect_page(bgr_frame, method="unet")` → quad (4,2) or None; `detect_page_rembg(bgr_frame)` for U-Net/rembg path
- `page_detection/draw.py` – `draw_quad_on_frame(frame, quad)`
- `run_detection.py` – CLI for **images only** (single + batch)
- `run_detection_images.py` – same as above (explicit name)
- `run_detection_video.py` – **video in → video out** with bounding boxes

Algorithm is aligned with `scanner_api.cpp` for easy porting to mobile (C++/OpenCV).
