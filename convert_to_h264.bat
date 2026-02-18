@echo off
REM Convert video to MP4 H.264 for use in the page-detection demo.
REM Usage: convert_to_h264.bat "path\to\video.mp4"
REM Requires: ffmpeg on PATH (https://ffmpeg.org/download.html)

if "%~1"=="" (
  echo Usage: convert_to_h264.bat "path\to\video.mp4"
  echo Or drag and drop a video file onto this script.
  exit /b 1
)

set "INPUT=%~1"
set "DIR=%~dp1"
set "NAME=%~n1"
set "OUTPUT=%DIR%%NAME%_h264.mp4"

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg not found. Install from https://ffmpeg.org/download.html
  exit /b 1
)

echo Converting: %INPUT%
echo Output:     %OUTPUT%
ffmpeg -y -i "%INPUT%" -c:v libx264 -c:a aac -movflags +faststart "%OUTPUT%"
if errorlevel 1 (
  echo Conversion failed.
  exit /b 1
)
echo Done: %OUTPUT%
exit /b 0
