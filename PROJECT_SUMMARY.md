# Document Data Extraction – Project Summary

Summary of what was implemented and how the repo is organized as of the latest cleanup.

---

## 1. Page / document detection

**Goal:** Detect a document page in an image (or video frame) and get a bounding quad so it can be drawn or used for cropping.

### Methods

| Method | Description | Use case |
|--------|-------------|----------|
| **unet** (default) | U-Net–style segmentation via rembg (U²-Net): foreground mask → contours → 4-point quad. | Best quality; text pages, book spreads, tilted/curved spines. |
| **rembg** | Same as unet. | Alias. |
| **auto** | Run Canny first; if no quad, fall back to unet. | Slightly faster when Canny works. |
| **canny** | OpenCV Canny + morphology + contours only. | Fast, works when contrast is high. |

### Pipeline (unet/rembg)

- Optional resize to `max_dim` (640) before inference for speed.
- Lazy rembg session reuse (u2netp) so the model is loaded once.
- Mask → largest document-like contour → convex hull → 4 extreme points (TL, TR, BR, BL).
- Optional **refinement**: subpixel corner refinement + snap corners to strongest edges (disabled with `--fast`).
- Quad is in original image coordinates; output images stay full resolution unless `--downscale` is used.

### Key files

- **`page_detection/detector.py`** – `detect_page(bgr_frame, method="unet", fast=False)`, `detect_page_rembg()`, Canny path, `_detect_from_mask()`, refinement/snap.
- **`page_detection/draw.py`** – `draw_quad_on_frame(frame, quad)` (green box by default).
- **`page_detection/__init__.py`** – Exposes `detect_page`, `draw_quad_on_frame`.

---

## 2. CLI and usage

### Images: `run_detection.py`

- **Single image:** `python run_detection.py image.jpg` → saves `image_with_box.jpg` (or `-o path`).
- **Batch (folder):** `python run_detection.py Test_Images -o Test_Images_Output` → same folder structure in output dir.

**Options:**

- `--method unet|auto|canny|rembg` (default: `unet`).
- `--fast` – Skip corner refinement for unet/rembg (faster, slightly less accurate).
- `--downscale MAX_PIX` – Resize so max dimension is MAX_PIX before detection; **output images are resized** (e.g. “small” runs).
- Batch runs write **`detection_metrics.json`** in the output folder (total time, per-image ms, page_detected count and list).

### Video / live camera: `run_detection_video.py`

- **Video file:** input video → output video with bounding boxes on each frame. Options: `--downscale`, `--sample-every N`, `--extract-pages [DIR]`, `--smooth ALPHA`, `--method`, `--fast`.
- **Live camera (real time):** run with camera index `0` (or `1`, …), e.g. `python run_detection_video.py 0`. A window shows the camera feed with the page box in real time; press **q** to quit. Use `-o recording.mp4` to save, and `--downscale 640 --fast` for better frame rate.

See **`README_detection.md`** for full usage and options.

---

## 3. Test runs and outputs

Detection was run on two test sets with three variants each:

| Input folder   | Variant | Output folder (kept) | Notes |
|----------------|---------|----------------------|--------|
| Test_Images    | normal  | **Test_Images_Output_normal** | Default; full-res, with refinement. **Recommended.** |
| Test_Images_2  | normal  | **Test_Images_2_Output_normal** | Same. |

**Archived (redundant) outputs** are in **`archived_detection_outputs.zip`**:

- `Test_Images_Output`, `Test_Images_Output_fast`, `Test_Images_Output_small`
- `Test_Images_2_Output`, `Test_Images_2_Output_fast`, `Test_Images_2_Output_small`

So currently only the **normal** (best-quality) results are kept on disk; fast and small variants are in the archive.

---

## 4. Recommendation

- **Best quality (full-res, most accurate box):** use default (normal) – no `--fast`, no `--downscale`.
- **Faster, still full-res:** `--fast`.
- **Fastest, smaller output:** `--downscale 640` (output images max 640 px).

---

## 5. Dependencies and setup

- **`requirements.txt`** – Includes `rembg[cpu]`, OpenCV, Pillow, etc. First run may download the U²-Net model (~4.5 MB for u2netp).
- GPU: `pip install "rembg[gpu]"` if needed.

---

## 6. Other artifacts in the repo

- **`run_detection_images.py`** – Same as `run_detection.py` (images only).
- **`scanner_api.cpp` / `scanner_api.h`** – C++ scanner API; algorithm aligned for possible mobile/C++ port.
- **`convert_to_h264.py`**, **`build_wasm.sh`**, **WASM/JS** – Video conversion and web build (separate from detection CLI).
- **`emsdk`** – Emscripten SDK for WASM build (if present).

---

## 7. Cleanup performed

- **Kept:** `Test_Images_Output_normal`, `Test_Images_2_Output_normal` (recommended results).
- **Archived:** All other detection output folders listed above → **`archived_detection_outputs.zip`**.
- **Deleted:** Those redundant folders after archiving.

To restore any variant, unzip **`archived_detection_outputs.zip`** and use the desired folder.
