#ifndef IMAGEANALYTICS_H
#define IMAGEANALYTICS_H

#include <opencv2/opencv.hpp>
#include <tuple>

// Function to calculate the image statistics (mean, stddev, min, max, dynamic range)
std::tuple<double, double, double, double, double, double, double> get_image_statistics(const cv::Mat& input);

double get_colorfulness(const cv::Mat& input);
double get_entropy(const cv::Mat& input);

void print_statistics(const std::tuple<double, double, double, double, double, double, double>& stats);

#endif // IMAGEANALYTICS_H
