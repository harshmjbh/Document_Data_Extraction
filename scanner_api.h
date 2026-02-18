// scanner_api.h
// Public C API for document detection and perspective warp, suitable for WebAssembly exports.
//
// All image buffers use 8-bit RGBA interleaved layout: width * height * 4 bytes.
// Coordinates are in pixel space with origin at the top-left of the image.
//
// detect_document:
//   - inPixels: input frame pixels (RGBA8).
//   - width, height: input image dimensions.
//   - outCorners: output array of 8 floats: [x0,y0, x1,y1, x2,y2, x3,y3] in TL,TR,BR,BL order.
//   - outConfidence: pointer to a single float in [0,1], higher = more confident.
//   - returns: 1 if a document was detected, 0 otherwise.
//
// warp_document:
//   - inPixels: input frame pixels (RGBA8).
//   - width, height: input image dimensions.
//   - corners: 8 floats [x0,y0, x1,y1, x2,y2, x3,y3] in TL,TR,BR,BL order (typically from detect_document).
//   - outPixels: output buffer for warped image pixels (RGBA8) of size outWidth * outHeight * 4 bytes.
//   - outWidth, outHeight: desired output dimensions.
//   - returns: 1 on success, 0 on failure.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Detect a document quadrilateral in an RGBA image.
int detect_document(
    const uint8_t* inPixels,
    int width,
    int height,
    float* outCorners,
    float* outConfidence);

// Warp the detected document to a fronto-parallel rectangle.
int warp_document(
    const uint8_t* inPixels,
    int width,
    int height,
    const float* corners,
    uint8_t* outPixels,
    int outWidth,
    int outHeight);

#ifdef __cplusplus
}
#endif

