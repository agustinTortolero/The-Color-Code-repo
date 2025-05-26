// helper.h
#ifndef HELPER_H
#define HELPER_H

#include <string>
#include <opencv2/opencv.hpp>

struct Thresholds { double v_low, v_high, minV, range; };

// Read a FITS file into a CV_64F single-channel Mat
bool readFITS(const std::string& path, cv::Mat& outImage);

// Print basic info (width × height)
void printInfo(const cv::Mat& image);

// Compute percentile-based thresholds from a CV_64F Mat
Thresholds computeThresholds(const cv::Mat& image);

// Display a Mat (8-bit 1- or 3-channel) in a resizable window
void showImage(const cv::Mat& image,
    const std::string& title,
    double scale = 0.25);

// Save a Mat (8-bit 1- or 3-channel) to disk, with optional resize
bool saveImage(const cv::Mat& image,
    double scale,
    const std::string& filename);

#endif // HELPER_H