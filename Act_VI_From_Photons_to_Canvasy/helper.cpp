// helper.cpp
#include "helper.h"
#include <cfitsio/fitsio.h>
#include <iostream>
#include <vector>
#include <algorithm>

bool readFITS(const std::string& path, cv::Mat& outImage) {
    fitsfile* fptr = nullptr;
    int status = 0;
    // Open FITS and move to SCI extension
    if (fits_open_file(&fptr, path.c_str(), READONLY, &status) ||
        fits_movnam_hdu(fptr, IMAGE_HDU, const_cast<char*>("SCI"), 0, &status)) {
        fits_report_error(stderr, status);
        return false;
    }
    int bitpix, naxis;
    long naxes[2] = { 0, 0 };
    fits_get_img_param(fptr, 2, &bitpix, &naxis, naxes, &status);
    long width = naxes[0], height = naxes[1];
    // Create a 64F Mat for raw data
    outImage.create(height, width, CV_64F);
    // Read pixels directly into Mat data
    long fpixel = 1;
    long nelements = width * height;
    if (fits_read_img(fptr, TDOUBLE, fpixel, nelements,
        nullptr, outImage.ptr<double>(),
        nullptr, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return false;
    }
    fits_close_file(fptr, &status);
    return true;
}

void printInfo(const cv::Mat& image) {
    std::cout << "Image size: "
        << image.cols << " x " << image.rows << std::endl;
}

Thresholds computeThresholds(const cv::Mat& image) {
    // Flatten into vector for percentile calculation
    std::vector<double> buf;
    buf.reserve(image.total());
    for (int r = 0; r < image.rows; ++r) {
        const double* ptr = image.ptr<double>(r);
        buf.insert(buf.end(), ptr, ptr + image.cols);
    }
    size_t N = buf.size();
    size_t i_low = static_cast<size_t>(0.01 * (N - 1));
    size_t i_high = static_cast<size_t>(0.99 * (N - 1));
    std::nth_element(buf.begin(), buf.begin() + i_low, buf.end());
    std::nth_element(buf.begin(), buf.begin() + i_high, buf.end());
    double v_low = buf[i_low];
    double v_high = buf[i_high];
    double minV = *std::min_element(buf.begin(), buf.end());
    double maxV = *std::max_element(buf.begin(), buf.end());
    return { v_low, v_high, minV, maxV - minV };
}

void showImage(const cv::Mat& image,
    const std::string& title,
    double scale) {
    cv::Mat disp;
    cv::resize(image, disp, cv::Size(), scale, scale,
        scale < 1.0 ? cv::INTER_AREA : cv::INTER_CUBIC);
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, disp);
}

bool saveImage(const cv::Mat& image,
    double scale,
    const std::string& filename) {
    cv::Mat out;
    if (std::abs(scale - 1.0) > 1e-6) {
        cv::resize(image, out,
            cv::Size(), scale, scale,
            scale < 1.0 ? cv::INTER_AREA : cv::INTER_CUBIC);
    }
    else {
        out = image;
    }
    if (!cv::imwrite(filename, out)) {
        std::cerr << "Error: could not write image to "
            << filename << std::endl;
        return false;
    }
    return true;
}