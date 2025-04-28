#include "Histogram.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

cv::Mat Histogram::drawHistogram(const std::vector<cv::Mat>& hists, const std::vector<cv::Scalar>& colors, int hist_h, int hist_w) {
    int bin_w = cvRound((double)hist_w / histSize);
    int channels = static_cast<int>(hists.size());

    int type = channels == 1 ? CV_8UC1 : CV_8UC3;
    cv::Mat histImage(hist_h, hist_w, type, cv::Scalar(0));

    for (int c = 0; c < channels; ++c) {
        for (int i = 1; i < histSize; ++i) {
            cv::line(histImage,
                cv::Point(bin_w * (i - 1), hist_h - cvRound(hists[c].at<float>(i - 1))),
                cv::Point(bin_w * i, hist_h - cvRound(hists[c].at<float>(i))),
                colors[c], 2);
        }
    }

    return histImage;
}

void Histogram::showHistogram(const cv::Mat& image) {
    if (image.channels() == 1) {
        // Grayscale
        cv::Mat hist;
        cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
        cv::normalize(hist, hist, 0, 400, cv::NORM_MINMAX);
        cv::Mat grayHistImage = drawHistogram({ hist }, { cv::Scalar(255) });

        cv::imshow("Grayscale Image", image);
        cv::imshow("Grayscale Histogram", grayHistImage);
    }
    else if (image.channels() == 3) {
        // Color
        std::vector<cv::Mat> bgr_planes;
        cv::split(image, bgr_planes);

        cv::Mat b_hist, g_hist, r_hist;
        cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange);
        cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange);
        cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange);

        cv::normalize(b_hist, b_hist, 0, 400, cv::NORM_MINMAX);
        cv::normalize(g_hist, g_hist, 0, 400, cv::NORM_MINMAX);
        cv::normalize(r_hist, r_hist, 0, 400, cv::NORM_MINMAX);

        cv::Mat colorHistImage = drawHistogram({ b_hist, g_hist, r_hist },
            { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255) });

        cv::imshow("Color Image", image);
        cv::imshow("Color Histogram", colorHistImage);
    }
    else {
        std::cerr << "Unsupported image format.\n";
    }

    cv::waitKey(0);
}


//borrar este?
cv::Mat Histogram::equalizeHistogram(const cv::Mat& image) {
    if (image.channels() == 1) {
        // Grayscale equalization
        cv::Mat equalized;
        cv::equalizeHist(image, equalized);
        return equalized;
    }
    else if (image.channels() == 3) {
        // Convert to YCrCb and equalize the Y channel
        cv::Mat ycrcb;
        cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);

        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);
        cv::equalizeHist(channels[0], channels[0]); // Equalize luminance

        cv::merge(channels, ycrcb);
        cv::Mat result;
        cv::cvtColor(ycrcb, result, cv::COLOR_YCrCb2BGR);
        return result;
    }
    else {
        std::cerr << "Unsupported image format for equalization.\n";
        return image.clone(); // Return a copy to be safe
    }
}
