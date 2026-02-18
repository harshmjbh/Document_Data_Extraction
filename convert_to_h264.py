#!/usr/bin/env python3
"""
Convert video file(s) to MP4 with H.264 video and AAC audio.
Uses bundled ffmpeg from imageio-ffmpeg (no separate install).

Usage:
  pip install imageio-ffmpeg
  python convert_to_h264.py <video_file> [video_file2 ...]
  python convert_to_h264.py   (converts all .mp4/.mov in current folder)
"""

import os
import subprocess
import sys

OUTPUT_SUFFIX = "_h264"
OUTPUT_EXT = ".mp4"


def get_ffmpeg():
    """Return path to ffmpeg: system PATH first, then imageio-ffmpeg bundle."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return "ffmpeg"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.isfile(path):
            return path
    except ImportError:
        pass
    return None


def convert_to_h264(ffmpeg_exe, input_path):
    """Convert one file to H.264 MP4. Returns output path or None on failure."""
    if not os.path.isfile(input_path):
        print(f"Skip (not a file): {input_path}")
        return None

    base, ext = os.path.splitext(input_path)
    if base.endswith(OUTPUT_SUFFIX):
        print(f"Skip (already converted): {input_path}")
        return None

    output_path = base + OUTPUT_SUFFIX + OUTPUT_EXT
    if os.path.exists(output_path):
        print(f"Output exists, overwriting: {output_path}")

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path,
    ]
    print(f"Converting: {input_path} -> {output_path}")
    try:
        subprocess.run(cmd, check=True)
        print(f"Done: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e}", file=sys.stderr)
        return None


def main():
    ffmpeg_exe = get_ffmpeg()
    if not ffmpeg_exe:
        print(
            "ffmpeg not found. Install the lightweight bundle:\n"
            "  pip install imageio-ffmpeg\n"
            "Then run this script again.",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        cwd = os.getcwd()
        paths = [
            os.path.join(cwd, f)
            for f in os.listdir(cwd)
            if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm"))
            and OUTPUT_SUFFIX not in f
        ]
        if not paths:
            print("No video files in current folder. Usage: python convert_to_h264.py <file.mp4>")
            sys.exit(0)

    ok = 0
    for p in paths:
        if convert_to_h264(ffmpeg_exe, p):
            ok += 1
    print(f"\nConverted {ok} of {len(paths)} file(s).")
    sys.exit(0 if ok == len(paths) else 1)


if __name__ == "__main__":
    main()
