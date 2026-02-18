// scanner_api.cpp
// C API implementation for document detection and perspective warp.
// For native builds, this uses OpenCV internally. For WebAssembly builds
// (Emscripten, where __EMSCRIPTEN__ is defined), we provide a lightweight
// implementation that avoids OpenCV so it can compile without extra setup.

#include "scanner_api.h"

#include <cstring>

#ifdef __EMSCRIPTEN__

// ---------------------------------------------------------------------------
// Emscripten (WASM) stub implementation
// ---------------------------------------------------------------------------
// This version does NOT depend on OpenCV. It treats the whole image as the
// document and applies a simple nearest-neighbor resize for warp_document.
// This is sufficient to validate the end-to-end WASM + JS pipeline.

int detect_document(
    const uint8_t* inPixels,
    int width,
    int height,
    float* outCorners,
    float* outConfidence) {
    if (!inPixels || !outCorners || !outConfidence || width <= 0 || height <= 0) {
        return 0;
    }

    // Full-frame quad: TL, TR, BR, BL.
    outCorners[0] = 0.0f;
    outCorners[1] = 0.0f;
    outCorners[2] = static_cast<float>(width - 1);
    outCorners[3] = 0.0f;
    outCorners[4] = static_cast<float>(width - 1);
    outCorners[5] = static_cast<float>(height - 1);
    outCorners[6] = 0.0f;
    outCorners[7] = static_cast<float>(height - 1);

    *outConfidence = 1.0f;
    return 1;
}

int warp_document(
    const uint8_t* inPixels,
    int width,
    int height,
    const float* /*corners*/,
    uint8_t* outPixels,
    int outWidth,
    int outHeight) {
    if (!inPixels || !outPixels ||
        width <= 0 || height <= 0 ||
        outWidth <= 0 || outHeight <= 0) {
        return 0;
    }

    // Simple nearest-neighbor scaling from the full input frame to the
    // requested output size.
    for (int y = 0; y < outHeight; ++y) {
        int srcY = static_cast<int>(
            static_cast<long long>(y) * height / (outHeight > 0 ? outHeight : 1));
        if (srcY >= height) srcY = height - 1;

        for (int x = 0; x < outWidth; ++x) {
            int srcX = static_cast<int>(
                static_cast<long long>(x) * width / (outWidth > 0 ? outWidth : 1));
            if (srcX >= width) srcX = width - 1;

            const size_t srcIndex = (static_cast<size_t>(srcY) * width + srcX) * 4u;
            const size_t dstIndex = (static_cast<size_t>(y) * outWidth + x) * 4u;

            outPixels[dstIndex + 0] = inPixels[srcIndex + 0];
            outPixels[dstIndex + 1] = inPixels[srcIndex + 1];
            outPixels[dstIndex + 2] = inPixels[srcIndex + 2];
            outPixels[dstIndex + 3] = inPixels[srcIndex + 3];
        }
    }

    return 1;
}

#else  // __EMSCRIPTEN__

// ---------------------------------------------------------------------------
// Native (non-Emscripten) implementation using OpenCV
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace {

struct Point2f {
    float x;
    float y;
};

struct Quad {
    Point2f pts[4];
};

// Compute Euclidean distance between two points.
float distance(const Point2f& a, const Point2f& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Order four points into TL, TR, BR, BL.
Quad orderCorners(const Quad& input) {
    Quad ordered{};

    float minSum = std::numeric_limits<float>::max();
    float maxSum = std::numeric_limits<float>::lowest();
    float minDiff = std::numeric_limits<float>::max();
    float maxDiff = std::numeric_limits<float>::lowest();

    int tl = 0, br = 0, tr = 0, bl = 0;

    for (int i = 0; i < 4; ++i) {
        float sum = input.pts[i].x + input.pts[i].y;
        float diff = input.pts[i].x - input.pts[i].y;

        if (sum < minSum) { minSum = sum; tl = i; }
        if (sum > maxSum) { maxSum = sum; br = i; }
        if (diff < minDiff) { minDiff = diff; tr = i; }
        if (diff > maxDiff) { maxDiff = diff; bl = i; }
    }

    ordered.pts[0] = input.pts[tl]; // TL
    ordered.pts[1] = input.pts[tr]; // TR
    ordered.pts[2] = input.pts[br]; // BR
    ordered.pts[3] = input.pts[bl]; // BL

    return ordered;
}

// Compute a simple confidence score based on area coverage and rectangularity.
float computeConfidence(const std::vector<cv::Point>& contour, const cv::Size& imageSize) {
    double contourArea = cv::contourArea(contour);
    if (contourArea <= 0.0) {
        return 0.0f;
    }

    double imageArea = static_cast<double>(imageSize.width) * static_cast<double>(imageSize.height);
    double areaRatio = contourArea / std::max(imageArea, 1.0);

    cv::Rect bbox = cv::boundingRect(contour);
    double bboxArea = static_cast<double>(bbox.width) * static_cast<double>(bbox.height);
    double rectangularity = bboxArea > 0.0 ? contourArea / bboxArea : 0.0;

    double score = areaRatio * rectangularity;
    // Clamp to [0,1].
    if (score < 0.0) score = 0.0;
    if (score > 1.0) score = 1.0;
    return static_cast<float>(score);
}

// Core page detection using OpenCV contours.
// Returns true and fills quad + confidence if a plausible document is found.
bool detectPageQuad(const cv::Mat& bgrImage, Quad& outQuad, float& outConfidence) {
    cv::Mat gray;
    cv::cvtColor(bgrImage, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    cv::Mat edges;
    cv::Canny(blurred, edges, 75, 200);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    double imageArea = static_cast<double>(bgrImage.cols) * static_cast<double>(bgrImage.rows);
    const double minArea = imageArea * 0.1; // ignore very small contours

    float bestConfidence = 0.0f;
    Quad bestQuad{};
    bool found = false;

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < minArea) {
            continue;
        }

        double peri = cv::arcLength(contour, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.02 * peri, true);

        if (approx.size() != 4 || !cv::isContourConvex(approx)) {
            continue;
        }

        // Compute confidence for this contour.
        float conf = computeConfidence(contour, bgrImage.size());
        if (conf <= bestConfidence) {
            continue;
        }

        Quad q{};
        for (int i = 0; i < 4; ++i) {
            q.pts[i].x = static_cast<float>(approx[i].x);
            q.pts[i].y = static_cast<float>(approx[i].y);
        }

        bestConfidence = conf;
        bestQuad = q;
        found = true;
    }

    if (!found) {
        return false;
    }

    Quad ordered = orderCorners(bestQuad);
    outQuad = ordered;
    outConfidence = bestConfidence;
    return true;
}

} // namespace

// C API implementation (native).

int detect_document(
    const uint8_t* inPixels,
    int width,
    int height,
    float* outCorners,
    float* outConfidence) {
    if (!inPixels || !outCorners || !outConfidence || width <= 0 || height <= 0) {
        return 0;
    }

    // Wrap input RGBA buffer into OpenCV Mat and convert to BGR.
    cv::Mat rgba(height, width, CV_8UC4, const_cast<uint8_t*>(inPixels));
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);

    Quad quad{};
    float confidence = 0.0f;
    bool ok = detectPageQuad(bgr, quad, confidence);
    if (!ok) {
        *outConfidence = 0.0f;
        return 0;
    }

    *outConfidence = confidence;

    // Write TL, TR, BR, BL into outCorners (8 floats).
    for (int i = 0; i < 4; ++i) {
        outCorners[i * 2 + 0] = quad.pts[i].x;
        outCorners[i * 2 + 1] = quad.pts[i].y;
    }

    return 1;
}

int warp_document(
    const uint8_t* inPixels,
    int width,
    int height,
    const float* corners,
    uint8_t* outPixels,
    int outWidth,
    int outHeight) {
    if (!inPixels || !corners || !outPixels ||
        width <= 0 || height <= 0 ||
        outWidth <= 0 || outHeight <= 0) {
        return 0;
    }

    // Wrap input RGBA buffer.
    cv::Mat rgba(height, width, CV_8UC4, const_cast<uint8_t*>(inPixels));
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);

    // Build source quad from corners (already TL, TR, BR, BL).
    std::vector<cv::Point2f> src(4);
    for (int i = 0; i < 4; ++i) {
        src[i].x = corners[i * 2 + 0];
        src[i].y = corners[i * 2 + 1];
    }

    // Destination rectangle corners.
    std::vector<cv::Point2f> dst(4);
    dst[0] = cv::Point2f(0.0f,                  0.0f);
    dst[1] = cv::Point2f(static_cast<float>(outWidth - 1), 0.0f);
    dst[2] = cv::Point2f(static_cast<float>(outWidth - 1),
                         static_cast<float>(outHeight - 1));
    dst[3] = cv::Point2f(0.0f,
                         static_cast<float>(outHeight - 1));

    cv::Mat H = cv::getPerspectiveTransform(src, dst);

    cv::Mat warpedBgr;
    cv::warpPerspective(
        bgr,
        warpedBgr,
        H,
        cv::Size(outWidth, outHeight),
        cv::INTER_LINEAR,
        cv::BORDER_REPLICATE);

    // Convert back to RGBA into the provided output buffer.
    cv::Mat warpedRgba;
    cv::cvtColor(warpedBgr, warpedRgba, cv::COLOR_BGR2RGBA);

    if (warpedRgba.cols != outWidth || warpedRgba.rows != outHeight) {
        return 0;
    }

    const size_t numBytes = static_cast<size_t>(outWidth) *
                            static_cast<size_t>(outHeight) * 4u;
    std::memcpy(outPixels, warpedRgba.data, numBytes);

    return 1;
}

#endif // __EMSCRIPTEN__

