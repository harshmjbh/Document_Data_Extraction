#!/usr/bin/env python3
"""
Page detection for video or live camera: draw bounding box on each frame.

Video file: input video -> output video with boxes. Optionally extract each new page as an image.
Live camera: real-time detection in a window; press 'q' to quit. Optionally record to a file.

Usage:
  # Video file
  python run_detection_video.py video.mp4 [--output video_with_boxes.mp4]
  python run_detection_video.py video.mp4 --extract-pages ./video_pages
  python run_detection_video.py video.mp4 -o out.mp4 --smooth 0.4 --sample-every 2

  # Live camera (real time)
  python run_detection_video.py 0
  python run_detection_video.py 0 -o live_recording.mp4 --downscale 640 --fast
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from page_detection import detect_page, draw_quad_on_frame


def _quad_center(quad: np.ndarray) -> tuple[float, float]:
    return (float(quad[:, 0].mean()), float(quad[:, 1].mean()))


def _quad_area(quad: np.ndarray) -> float:
    return float(cv2.contourArea(quad))


def process_camera(
    camera_index: int,
    output_path: Path | None,
    downscale: int | None,
    sample_every: int,
    smooth_alpha: float | None,
    method: str = "unet",
    fast: bool = False,
) -> None:
    """
    Live webcam: read from camera, run page detection, show in a window in real time.
    Press 'q' to quit. Optionally record to output_path.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: could not open camera {camera_index}", file=sys.stderr)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            writer = None
            print(f"Warning: could not create output video {output_path}", file=sys.stderr)
        else:
            print(f"Recording to {output_path}")

    print(f"Live page detection (camera {camera_index}). Press 'q' to quit.")
    print(f"Method: {method}, fast: {fast}, downscale: {downscale}, sample-every: {sample_every}")

    last_quad = None
    smoothed_quad = None
    alpha = smooth_alpha if smooth_alpha is not None else 1.0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        run_detection = frame_index % sample_every == 0
        if run_detection:
            work = frame
            if downscale and max(h, w) > downscale:
                scale = downscale / max(h, w)
                work = cv2.resize(frame, (int(w * scale), int(h * scale)))
            quad = detect_page(work, method=method, fast=fast)
            if quad is not None:
                if work is not frame:
                    scale_x = w / work.shape[1]
                    scale_y = h / work.shape[0]
                    quad = quad.copy()
                    quad[:, 0] *= scale_x
                    quad[:, 1] *= scale_y
                last_quad = quad
                if alpha < 1.0 and smoothed_quad is not None:
                    smoothed_quad = alpha * quad + (1 - alpha) * smoothed_quad
                else:
                    smoothed_quad = quad.copy()
            else:
                last_quad = None
                smoothed_quad = None

        draw_quad = smoothed_quad if smoothed_quad is not None else last_quad
        if draw_quad is not None:
            draw_quad_on_frame(frame, draw_quad, in_place=True)

        cv2.imshow("Page detection (live)", frame)
        if writer is not None:
            writer.write(frame)

        frame_index += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("Stopped.")


def _is_new_page(
    quad: np.ndarray,
    last_saved_quad: np.ndarray | None,
    frame_w: int,
    frame_h: int,
    min_center_shift: float = 0.12,
    min_area_ratio: float = 0.65,
    max_area_ratio: float = 1.5,
) -> bool:
    """True if this quad is considered a different page from last saved."""
    if last_saved_quad is None:
        return True
    cx, cy = _quad_center(quad)
    lx, ly = _quad_center(last_saved_quad)
    diag = (frame_w**2 + frame_h**2) ** 0.5
    dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5 / max(diag, 1)
    if dist > min_center_shift:
        return True
    area = _quad_area(quad)
    last_area = _quad_area(last_saved_quad)
    if last_area <= 0:
        return True
    ratio = area / last_area
    if ratio < min_area_ratio or ratio > max_area_ratio:
        return True
    return False


def process_video(
    input_path: Path,
    output_path: Path,
    downscale: int | None,
    sample_every: int,
    smooth_alpha: float | None,
    extract_pages_dir: Path | None,
    min_frames_between_pages: int,
    method: str = "unet",
    fast: bool = False,
) -> None:
    """
    Read video, run page detection, draw box (with optional smoothing), write output video.
    If extract_pages_dir is set, save each new page as an image there.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: could not open video {input_path}", file=sys.stderr)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"Error: could not create output video {output_path}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    if extract_pages_dir is not None:
        extract_pages_dir = Path(extract_pages_dir)
        extract_pages_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting new pages to: {extract_pages_dir}")

    last_quad = None
    smoothed_quad = None
    last_saved_quad = None
    last_saved_frame_index = -min_frames_between_pages - 1
    frame_index = 0
    detected_count = 0
    pages_saved = 0
    alpha = smooth_alpha if smooth_alpha is not None else 1.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        run_detection = frame_index % sample_every == 0
        if run_detection:
            work = frame
            if downscale and max(h, w) > downscale:
                scale = downscale / max(h, w)
                work = cv2.resize(frame, (int(w * scale), int(h * scale)))
            quad = detect_page(work, method=method, fast=fast)
            if quad is not None:
                if work is not frame:
                    scale_x = w / work.shape[1]
                    scale_y = h / work.shape[0]
                    quad = quad.copy()
                    quad[:, 0] *= scale_x
                    quad[:, 1] *= scale_y
                last_quad = quad
                detected_count += 1
                if alpha < 1.0 and smoothed_quad is not None:
                    smoothed_quad = alpha * quad + (1 - alpha) * smoothed_quad
                else:
                    smoothed_quad = quad.copy()
            else:
                last_quad = None
                smoothed_quad = None

        draw_quad = smoothed_quad if smoothed_quad is not None else last_quad
        if draw_quad is not None:
            draw_quad_on_frame(frame, draw_quad, in_place=True)

        # Extract new page when we have a detection and it's a new page and enough frames since last save
        if extract_pages_dir is not None and last_quad is not None:
            if _is_new_page(last_quad, last_saved_quad, w, h) and (
                frame_index - last_saved_frame_index >= min_frames_between_pages
            ):
                pages_saved += 1
                out_name = extract_pages_dir / f"page_{pages_saved:04d}.png"
                cv2.imwrite(str(out_name), frame)
                last_saved_quad = last_quad.copy()
                last_saved_frame_index = frame_index
                print(f"  Saved new page: {out_name.name} (frame {frame_index})")

        writer.write(frame)
        frame_index += 1
        if frame_index % 100 == 0 or frame_index == total_frames:
            print(f"  Frame {frame_index}/{total_frames}")

    cap.release()
    writer.release()
    print(f"Done. Output video: {output_path}")
    print(f"  Frames with page detected: {detected_count}")
    if extract_pages_dir is not None:
        print(f"  Pages extracted: {pages_saved} -> {extract_pages_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect document page in video; output video with bounding boxes; optionally extract each new page as an image."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input video file path, or camera index (0, 1, ...) for live real-time detection",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output video path (default: input_with_boxes.mp4)",
    )
    parser.add_argument(
        "--extract-pages",
        nargs="?",
        type=Path,
        default=None,
        const=Path("."),
        metavar="DIR",
        help="Save each new page as an image in DIR (page_0001.png, ...). If DIR omitted, uses <output_stem>_pages next to output video.",
    )
    parser.add_argument(
        "--min-frames-between-pages",
        type=int,
        default=10,
        metavar="N",
        help="Minimum frames between extracted pages to avoid duplicates (default 10)",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=None,
        metavar="MAX_PIX",
        help="Run detection on resized frame (max dimension MAX_PIX) for speed; box drawn on full res",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        metavar="N",
        help="Run detection every N frames; reuse last quad for in-between (faster, smoother)",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=None,
        metavar="ALPHA",
        help="Smooth bounding box over time: ALPHA in (0,1], e.g. 0.4 = less jitter (default: no smoothing)",
    )
    parser.add_argument(
        "--method",
        choices=("unet", "auto", "canny", "rembg"),
        default="unet",
        help="Detection method (default: unet)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Faster unet/rembg: skip corner refinement",
    )
    args = parser.parse_args()

    # Live camera: input "0" or "1" etc.
    try:
        cam_index = int(str(args.input))
        is_camera = True
    except (ValueError, TypeError):
        is_camera = False

    if is_camera:
        output_path = Path(args.output) if args.output is not None else None
        process_camera(
            cam_index,
            output_path,
            args.downscale,
            args.sample_every,
            args.smooth,
            args.method,
            args.fast,
        )
        return

    if not args.input.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_with_boxes.mp4"
    else:
        args.output = Path(args.output)

    extract_dir = args.extract_pages
    if extract_dir is not None:
        if extract_dir == Path("."):
            extract_dir = args.output.parent / f"{args.output.stem}_pages"
        else:
            extract_dir = Path(extract_dir)

    process_video(
        args.input,
        args.output,
        args.downscale,
        args.sample_every,
        args.smooth,
        extract_dir,
        args.min_frames_between_pages,
        args.method,
        args.fast,
    )


if __name__ == "__main__":
    main()
