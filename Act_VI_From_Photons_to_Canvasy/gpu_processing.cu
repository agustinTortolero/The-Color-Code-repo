// gpu_processing.cu
#include "helper.h"    
#include "gpu_processing.cuh"
#include "kernel.cuh"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <iostream>

#define CUDA_CHECK(err) \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; exit(EXIT_FAILURE); }

GpuProcessor::GpuProcessor(const cv::Mat& hostImage) {
    CV_Assert(hostImage.type() == CV_64F && hostImage.channels() == 1);
    width = hostImage.cols;
    height = hostImage.rows;
    N = static_cast<size_t>(width) * height;
    allocateDevice(hostImage);
    computeThresholds(hostImage);
}

GpuProcessor::~GpuProcessor() {
    freeDevice();
}

void GpuProcessor::allocateDevice(const cv::Mat& hostImage) {
    size_t bufBytes = N * sizeof(double);
    size_t grayBytes = N * sizeof(unsigned char);
    size_t colBytes = N * 3 * sizeof(unsigned char);

    CUDA_CHECK(cudaMalloc(&d_buf, bufBytes));
    CUDA_CHECK(cudaMalloc(&d_lin, grayBytes));
    CUDA_CHECK(cudaMalloc(&d_log, grayBytes));
    CUDA_CHECK(cudaMalloc(&d_blur, grayBytes));
    CUDA_CHECK(cudaMalloc(&d_col_lin, colBytes));
    CUDA_CHECK(cudaMalloc(&d_col_log, colBytes));

    CUDA_CHECK(cudaMemcpy(d_buf, hostImage.ptr<double>(), bufBytes,
        cudaMemcpyHostToDevice));
}

void GpuProcessor::freeDevice() {
    cudaFree(d_buf);
    cudaFree(d_lin);
    cudaFree(d_log);
    cudaFree(d_blur);
    cudaFree(d_col_lin);
    cudaFree(d_col_log);
}

void GpuProcessor::computeThresholds(const cv::Mat& hostImage) {
    th = ::computeThresholds(hostImage);
}

std::pair<float, float> GpuProcessor::benchmarkLinear(int runs) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(runs);
    int threads = 256;
    int blocks = static_cast<int>((N + threads - 1) / threads);

    for (int i = 0; i < runs; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        gpu_linear_percentile_stretch << <blocks, threads >> > (d_buf, d_lin,
            N, th.v_low, th.v_high);
        gpu_gaussian_blur << <blocks, threads >> > (d_lin, d_blur, N, width, height);
        gpu_colorize << <blocks, threads >> > (d_blur, d_col_lin, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float min_t = *std::min_element(times.begin(), times.end());
    float avg_t = std::accumulate(times.begin(), times.end(), 0.0f) / runs;
    return { min_t / 1000.0f, avg_t / 1000.0f };
}

std::pair<float, float> GpuProcessor::benchmarkLog(int runs) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times(runs);
    int threads = 256;
    int blocks = static_cast<int>((N + threads - 1) / threads);

    for (int i = 0; i < runs; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        gpu_log_scale_stretch << <blocks, threads >> > (d_buf, d_log,
            N, th.minV, th.range);
        gpu_gaussian_blur << <blocks, threads >> > (d_log, d_blur, N, width, height);
        gpu_colorize << <blocks, threads >> > (d_blur, d_col_log, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float min_t = *std::min_element(times.begin(), times.end());
    float avg_t = std::accumulate(times.begin(), times.end(), 0.0f) / runs;
    return { min_t / 1000.0f, avg_t / 1000.0f };
}

cv::Mat GpuProcessor::getLinearColor() {
    cv::Mat img(height, width, CV_8UC3);
    CUDA_CHECK(cudaMemcpy(img.data, d_col_lin,
        N * 3 * sizeof(unsigned char),
        cudaMemcpyDeviceToHost));
    return img;
}

cv::Mat GpuProcessor::getLogColor() {
    cv::Mat img(height, width, CV_8UC3);
    CUDA_CHECK(cudaMemcpy(img.data, d_col_log,
        N * 3 * sizeof(unsigned char),
        cudaMemcpyDeviceToHost));
    return img;
}
