#include "helper.h"
#include "gpu_processing.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    const std::string path =
        "hlsp_hlastarclusters_hst_acs-wfc_ngc4038-10188-10_F814W_v1_sci_sci.fits";

    cv::Mat raw64;
    if (!readFITS(path, raw64)) return 1;
    printInfo(raw64);
    // Save unprocessed raw for reference
    cv::Mat raw8;
    raw64.convertTo(raw8, CV_8U, 255.0 / (computeThresholds(raw64).range),
        -computeThresholds(raw64).minV * 255.0 / computeThresholds(raw64).range);
    saveImage(raw8, 0.08, "raw_unprocessed.jpg");

    GpuProcessor gpu(raw64);

    auto lin_t = gpu.benchmarkLinear();
    std::cout << "GPU Linear+Color min:" << lin_t.first
        << " s, avg:" << lin_t.second << " s\n";

    auto log_t = gpu.benchmarkLog();
    std::cout << "GPU Log+Color    min:" << log_t.first
        << " s, avg:" << log_t.second << " s\n";

    cv::Mat imgLin = gpu.getLinearColor();
    cv::Mat imgLog = gpu.getLogColor();

    showImage(imgLin, "GPU Linear+Color");
    showImage(imgLog, "GPU Log+Color");

    saveImage(imgLin,1, "output_linear.jpg");
    saveImage(imgLog, 1, "output_log.jpg");

    saveImage(raw8, 1, "raw_unprocessed.png");

    cv::waitKey(0);
    return 0;
}
