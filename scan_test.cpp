// scan_test.cpp
// Simple desktop test harness for the document scanner C API.
//
// Usage (example):
//   ./scan_test input.jpg output.jpg
//
// This program:
//   - loads an input image using OpenCV,
//   - converts it to RGBA buffer,
//   - calls detect_document to find the document quad,
//   - computes a reasonable output size from the quad,
//   - calls warp_document to obtain a cropped, perspective-corrected image,
//   - saves the warped image to the provided output path.

#include "scanner_api.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>\n";
        return 1;
    }

    const std::string inputPath = argv[1];
    const std::string outputPath = argv[2];

    cv::Mat bgr = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "Failed to load image: " << inputPath << "\n";
        return 1;
    }

    const int width = bgr.cols;
    const int height = bgr.rows;

    // Convert BGR to RGBA buffer as expected by the C API.
    cv::Mat rgba;
    cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);

    const size_t numBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
    std::vector<uint8_t> inPixels(numBytes);
    std::memcpy(inPixels.data(), rgba.data, numBytes);

    float corners[8] = {0};
    float confidence = 0.0f;

    int found = detect_document(inPixels.data(), width, height, corners, &confidence);
    if (!found) {
        std::cerr << "No document detected in image.\n";
        return 1;
    }

    std::cout << "Document detected with confidence: " << confidence << "\n";
    std::cout << "Corners (TL, TR, BR, BL):\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "  (" << corners[i * 2 + 0] << ", " << corners[i * 2 + 1] << ")\n";
    }

    // Compute output size based on corner distances.
    auto dist = [](float x0, float y0, float x1, float y1) {
        float dx = x0 - x1;
        float dy = y0 - y1;
        return std::sqrt(dx * dx + dy * dy);
    };

    float widthTop = dist(corners[0], corners[1], corners[2], corners[3]);   // TL->TR
    float widthBottom = dist(corners[6], corners[7], corners[4], corners[5]); // BL->BR
    float maxWidth = std::max(widthTop, widthBottom);

    float heightLeft = dist(corners[0], corners[1], corners[6], corners[7]);  // TL->BL
    float heightRight = dist(corners[2], corners[3], corners[4], corners[5]); // TR->BR
    float maxHeight = std::max(heightLeft, heightRight);

    int outWidth = static_cast<int>(std::round(maxWidth));
    int outHeight = static_cast<int>(std::round(maxHeight));

    // Clamp the longest side to a reasonable maximum (e.g. 1500 px).
    const int maxSide = 1500;
    float scale = 1.0f;
    int longest = std::max(outWidth, outHeight);
    if (longest > maxSide) {
        scale = static_cast<float>(maxSide) / static_cast<float>(longest);
        outWidth = static_cast<int>(outWidth * scale);
        outHeight = static_cast<int>(outHeight * scale);
    }

    if (outWidth <= 0 || outHeight <= 0) {
        std::cerr << "Computed invalid output size.\n";
        return 1;
    }

    std::vector<uint8_t> outPixels(static_cast<size_t>(outWidth) *
                                   static_cast<size_t>(outHeight) * 4u);

    int ok = warp_document(
        inPixels.data(),
        width,
        height,
        corners,
        outPixels.data(),
        outWidth,
        outHeight);

    if (!ok) {
        std::cerr << "warp_document failed.\n";
        return 1;
    }

    // Convert the RGBA output buffer back to BGR for saving.
    cv::Mat outRgba(outHeight, outWidth, CV_8UC4, outPixels.data());
    cv::Mat outBgr;
    cv::cvtColor(outRgba, outBgr, cv::COLOR_RGBA2BGR);

    if (!cv::imwrite(outputPath, outBgr)) {
        std::cerr << "Failed to write output image: " << outputPath << "\n";
        return 1;
    }

    std::cout << "Warped document saved to: " << outputPath << "\n";
    return 0;
}

