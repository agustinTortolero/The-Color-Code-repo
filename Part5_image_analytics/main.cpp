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

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <tuple>

#include "Histogram.h"
#include "ImageAnalytics.h"


int main() {

    std::string imagePath = "PATH_TO_IMG";

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_UNCHANGED); 

    if (image.empty()) {
        std::cerr << "Could not open or find the image at " << imagePath << std::endl;
        return -1;
    }

    if (image.channels() != 1 && image.channels() != 3) {
        std::cerr << "Unsupported image format. Only grayscale and 3-channel color images are supported.\n";
        return -1;
    }

    auto stats = get_image_statistics(image);

    print_statistics(stats);

    return 0;
}

