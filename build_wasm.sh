#!/usr/bin/env bash
# build_wasm.sh
#
# Build script for compiling the C++ document scanner core to WebAssembly
# using Emscripten. This produces a modularized JS loader (scanner.js) and
# the corresponding WebAssembly binary (scanner.wasm).
#
# Prerequisites:
#   - Emscripten SDK installed and activated:
#       https://emscripten.org/docs/getting_started/downloads.html
#   - OpenCV built/available for Emscripten, or replace OpenCV calls with
#     your own image-processing code.
#
# Usage:
#   emsdk_env  # or source emsdk_env.sh
#   ./build_wasm.sh

set -euo pipefail

SRC_FILES="scanner_api.cpp"
OUT_JS="scanner.js"

emcc ${SRC_FILES} \
  -O3 \
  -s WASM=1 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=\"createScannerModule\" \
  -s EXPORTED_FUNCTIONS='[_detect_document,_warp_document,_malloc,_free]' \
  -s EXPORTED_RUNTIME_METHODS='[ccall,cwrap,HEAPU8,HEAPF32]' \
  -o "${OUT_JS}"

echo "Built WebAssembly module: ${OUT_JS} and scanner.wasm"

