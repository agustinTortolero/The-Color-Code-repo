#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <opencv2/opencv.hpp>
#include <vector>

class Histogram {
protected:
    static constexpr int histSize = 256;
    std::vector<float> range = { 0, 256 };
    const float* histRange = range.data();

    cv::Mat drawHistogram(const std::vector<cv::Mat>& hists, const std::vector<cv::Scalar>& colors, int hist_h = 400, int hist_w = 512);

public:
    void showHistogram(const cv::Mat& image);
    cv::Mat equalizeHistogram(const cv::Mat& image);
};

#endif
