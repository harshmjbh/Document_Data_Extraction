#!/usr/bin/env python3
"""
Page detection for images (single file or batch folder).
For video input -> output video with boxes, use: python run_detection_video.py video.mp4 -o out.mp4

Usage:
  python run_detection.py image.jpg [--output image_with_box.jpg]
  python run_detection.py Test_Images -o Test_Images_Output   # batch: same folder structure
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from page_detection import detect_page, draw_quad_on_frame

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def process_image(
    input_path: Path,
    output_path: Path,
    downscale: int | None,
    method: str = "auto",
    fast: bool = False,
) -> bool:
    """Detect page in image, draw box, save. Returns True if a quad was found."""
    frame = cv2.imread(str(input_path))
    if frame is None:
        print(f"Error: could not read image {input_path}", file=sys.stderr)
        return False
    h, w = frame.shape[:2]
    if downscale and max(h, w) > downscale:
        scale = downscale / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    quad = detect_page(frame, method=method, fast=fast)
    if quad is not None:
        draw_quad_on_frame(frame, quad, in_place=True)
        cv2.imwrite(str(output_path), frame)
        print(f"Page detected. Saved to {output_path}")
        return True
    cv2.imwrite(str(output_path), frame)
    print(f"No page detected. Saved original to {output_path}")
    return False


def process_batch(
    input_dir: Path,
    output_dir: Path,
    downscale: int | None,
    method: str = "auto",
    fast: bool = False,
) -> None:
    """Process all images in input_dir, mirror folder structure in output_dir, time each and total."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            rel = p.relative_to(input_dir)
            files.append((p, output_dir / rel))

    if not files:
        print(f"No image files found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(files)} image(s) from {input_dir}")
    print(f"Output folder (same structure): {output_dir}")
    print(f"Detection method: {method}\n")

    total_start = time.perf_counter()
    detected = 0
    per_image: list[dict] = []

    for i, (in_path, out_path) in enumerate(files, 1):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        found = process_image(in_path, out_path, downscale, method, fast)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if found:
            detected += 1
        per_image.append({"file": in_path.name, "ms": round(elapsed_ms, 1), "page_detected": found})
        print(f"  [{i}/{len(files)}] {in_path.name}  {elapsed_ms:.1f} ms  {'[page]' if found else '[no page]'}")

    total_ms = (time.perf_counter() - total_start) * 1000
    print()
    print("--- Timing summary ---")
    print(f"  Total: {total_ms:.1f} ms  ({total_ms/1000:.2f} s)")
    print(f"  Per image (avg): {total_ms/len(files):.1f} ms")
    print(f"  Page detected: {detected}/{len(files)}")

    metrics = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "method": method,
        "fast": fast,
        "downscale": downscale,
        "total_ms": round(total_ms, 1),
        "total_seconds": round(total_ms / 1000, 2),
        "num_images": len(files),
        "avg_ms_per_image": round(total_ms / len(files), 1),
        "page_detected_count": detected,
        "per_image": per_image,
    }
    metrics_path = output_dir / "detection_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect document page and draw bounding box on image(s). For video use run_detection_video.py"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input image file or directory (batch: same structure output)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file or directory (required for batch)",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=None,
        metavar="MAX_PIX",
        help="Resize so max dimension is MAX_PIX before detection (faster)",
    )
    parser.add_argument(
        "--method",
        choices=("unet", "auto", "canny", "rembg"),
        default="unet",
        help="Detection method: unet (U-Net segmentation, default), auto (Canny then unet), canny only, rembg (same as unet)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Faster unet/rembg: skip corner refinement (slightly less accurate box)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    ext = args.input.suffix.lower() if args.input.is_file() else ""
    if ext in VIDEO_EXTENSIONS:
        print("For video input, use: python run_detection_video.py <video> -o <output.mp4>", file=sys.stderr)
        sys.exit(1)

    if args.input.is_dir():
        if args.output is None:
            print("Error: for batch mode specify -o/--output directory.", file=sys.stderr)
            sys.exit(1)
        process_batch(args.input, Path(args.output), args.downscale, args.method, args.fast)
        return

    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_with_box{args.input.suffix}"
    else:
        args.output = Path(args.output)

    process_image(args.input, args.output, args.downscale, args.method, args.fast)


if __name__ == "__main__":
    main()
