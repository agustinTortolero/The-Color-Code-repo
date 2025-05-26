// gpu_processing.cuh
#ifndef GPU_PROCESSING_CUH
#define GPU_PROCESSING_CUH

#include <opencv2/opencv.hpp>
#include <utility>

struct Thresholds;

class GpuProcessor {
public:
    // hostImage must be CV_64F single-channel
    GpuProcessor(const cv::Mat& hostImage);
    ~GpuProcessor();

    std::pair<float, float> benchmarkLinear(int runs = 10);
    std::pair<float, float> benchmarkLog(int runs = 10);

    // Returns CV_8UC3 colorized output
    cv::Mat getLinearColor();
    cv::Mat getLogColor();

private:
    void allocateDevice(const cv::Mat& hostImage);
    void freeDevice();
    void computeThresholds(const cv::Mat& hostImage);

    size_t N;
    int width, height;
    Thresholds th;

    double* d_buf;
    unsigned char* d_lin;
    unsigned char* d_log;
    unsigned char* d_blur;
    unsigned char* d_col_lin;
    unsigned char* d_col_log;
};

#endif // GPU_PROCESSING_CUH