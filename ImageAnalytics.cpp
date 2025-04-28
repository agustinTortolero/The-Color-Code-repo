/*
 * Image Statistics and Analytics Utility
 *
 * Author: Agustin Tortolero
 * Website: https://thecolorcode.net
 *
 * Description:
 * This C++ program, designed using OpenCV and modern C++17 features, computes a comprehensive set of image
 * statistics for both grayscale and color images. The main functionalities include:
 *
 *  - Mean and standard deviation of grayscale pixel intensities
 *  - Minimum and maximum grayscale values
 *  - Dynamic range calculation (max - min)
 *  - Colorfulness metric (based on Hasler and Süsstrunk's method)
 *  - Entropy measurement (indicates the information content or randomness of the image)
 *
 * The function `get_image_statistics` returns a `std::tuple` containing all these values in a structured form,
 * and the helper function `print_statistics` outputs them to the console.
 *
 * Image Support:
 *  - Grayscale (1-channel)
 *  - BGR Color (3-channel)
 *
 * Applications:
 *  - Image quality analysis
 *  - Pre-processing diagnostics
 *  - Content-based image retrieval or enhancement pipelines
 *
 *  Note: Ensure your compiler is configured to support C++17 or later, as the code uses `std::tuple` structured bindings.
 * For example, compile with: `g++ -std=c++17 your_file.cpp -o your_program $(pkg-config --cflags --libs opencv4)`
 *
 * Explanation Source:
 * All algorithmic techniques and image processing insights are detailed on the blog:
 * https://thecolorcode.net — authored by Agustin Tortolero
 */

// internally, cv::cvtColor(input, luminance, cv::COLOR_BGR2GRAY); does luminanse conversion.

#include "ImageAnalytics.h"
#include <iostream>

#include <tuple>
#include <opencv2/opencv.hpp>
#include <iostream>


std::tuple<double, double, double, double, double, double, double> get_image_statistics(const cv::Mat& input) {
    cv::Scalar mean, stddev;
    double minVal, maxVal;
    double entropy = get_entropy(input);  // Assuming you have this function implemented
    double colorfulness = 0.0;  // Default value for grayscale

    if (input.channels() == 1) {
        // Grayscale image
        cv::meanStdDev(input, mean, stddev);
        cv::minMaxLoc(input, &minVal, &maxVal);
    }
    else if (input.channels() == 3) {
        std::vector<cv::Mat> channels(3);
        cv::split(input, channels);

        // Compute luminance
        cv::Mat luminance = 0.299 * channels[2] + 0.587 * channels[1] + 0.114 * channels[0];

        cv::meanStdDev(luminance, mean, stddev);
        cv::minMaxLoc(luminance, &minVal, &maxVal);

        colorfulness = get_colorfulness(input);
    }
    else {
        std::cerr << "Unsupported image format. Only grayscale and 3-channel color images are supported.\n";
        return {};
    }

    return std::make_tuple(
        mean[0],           // luminance mean or grayscale mean
        stddev[0],         // stddev of luminance or grayscale
        minVal,
        maxVal,
        maxVal - minVal,   // dynamic range
        colorfulness,
        entropy
    );
}


double get_entropy(const cv::Mat& input) {
    cv::Mat gray;

    // Convert to grayscale if necessary
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else if (input.channels() == 1) {
        gray = input;
    }
    else {
        std::cerr << "Unsupported image format for entropy calculation.\n";
        return 0.0;
    }

    // Calculate histogram
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist;

    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

    // Normalize the histogram to get probabilities
    hist /= gray.total();

    double entropy = 0.0;
    for (int i = 0; i < histSize; ++i) {
        float p = hist.at<float>(i);
        if (p > 0) {
            entropy -= p * std::log2(p);
        }
    }

    return entropy;
}


double get_colorfulness(const cv::Mat& input) {
    cv::Mat image;
    input.convertTo(image, CV_32F);

    // Convert the image from BGR (RGB) to YCbCr
    cv::Mat ycbcr_image;
    cv::cvtColor(image, ycbcr_image, cv::COLOR_BGR2YCrCb);

    // Split the YCbCr image into its channels (Y, Cb, Cr)
    std::vector<cv::Mat> channels(3);
    cv::split(ycbcr_image, channels);

    cv::Mat Cb = channels[1];  // Cb channel
    cv::Mat Cr = channels[2];  // Cr channel

    // Calculate the differences (Cb - Cr) and (0.5 * (Cb + Cr) - Y) if necessary
    cv::Mat cb_cr = Cb - Cr;
    cv::Mat yb = 0.5 * (Cb + Cr) - channels[0];  // Y is the first channel (channels[0])

    // Compute the mean and standard deviation for both channels
    cv::Scalar mean_cb_cr, stddev_cb_cr, mean_yb, stddev_yb;
    cv::meanStdDev(cb_cr, mean_cb_cr, stddev_cb_cr);
    cv::meanStdDev(yb, mean_yb, stddev_yb);

    // Get standard deviations and means
    double std_cb_cr = stddev_cb_cr[0];
    double std_yb = stddev_yb[0];
    double mean_cb_cr_val = mean_cb_cr[0];
    double mean_yb_val = mean_yb[0];

    // Combine the standard deviations and means
    double std_root = std::sqrt(std_cb_cr * std_cb_cr + std_yb * std_yb);
    double mean_root = std::sqrt(mean_cb_cr_val * mean_cb_cr_val + mean_yb_val * mean_yb_val);

    // Return the colorfulness score
    return std_root + 0.3 * mean_root;
}


void print_statistics(const std::tuple<double, double, double, double, double, double, double>& stats) {
    auto [mean, stddev, min, max, dynamic_range, colorfulness, entropy] = stats;

    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Standard Deviation: " << stddev << std::endl;
    std::cout << "Min Value: " << min << std::endl;
    std::cout << "Max Value: " << max << std::endl;
    std::cout << "Dynamic Range: " << dynamic_range << std::endl;
    std::cout << "Colorfulness: " << colorfulness << std::endl;
    std::cout << "entropy: " << entropy << std::endl;
}
