#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#include <cmath>

__global__ void gpu_linear_percentile_stretch(
    const double* in, unsigned char* out,
    size_t N, double v_low, double v_high)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double range = v_high - v_low;
        double v = in[i];
        v = (v < v_low) ? v_low : (v > v_high ? v_high : v);
        double scaled = (v - v_low) / range * 255.0;
        scaled = fmin(fmax(scaled, 0.0), 255.0);
        out[i] = static_cast<unsigned char>(scaled + 0.5);
    }
}

__global__ void gpu_log_scale_stretch(
    const double* in, unsigned char* out,
    size_t N, double minV, double range)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    double denom = log(256.0);
    if (i < N) {

        double norm = (in[i] - minV) / range;
        double lg = log(1.0 + norm * 255.0) / denom;
        double scaled = fmin(fmax(lg * 255.0, 0.0), 255.0);
        out[i] = static_cast<unsigned char>(scaled + 0.5);
    }
}

__global__ void gpu_colorize(
    const unsigned char* gray, unsigned char* rgb, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned char v = gray[i];
        double f = (v <= 128) ? (v / 128.0) : ((v - 128) / 127.0);
        unsigned char r = (v <= 128) ? static_cast<unsigned char>(f * 255 + 0.5) : 255;
        unsigned char g = (v <= 128) ? 0 : static_cast<unsigned char>(f * 255 + 0.5);
        unsigned char b = (v <= 128) ? static_cast<unsigned char>(f * 255 + 0.5) : 255;
        rgb[3 * i + 0] = b;
        rgb[3 * i + 1] = g;
        rgb[3 * i + 2] = r;
    }
}

// 3×3 Gaussian blur kernel implementation
__global__ void gpu_gaussian_blur(
    const unsigned char* in,
    unsigned char* out,
    size_t               N,
    long                 width,
    long                 height)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int x = idx % width;
    int y = idx / width;
    int sum = 0;
    // weights: center=4, edges=2, corners=1
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = min(max(x + dx, 0), (int)width - 1);
            int ny = min(max(y + dy, 0), (int)height - 1);
            int w = (dx == 0 && dy == 0) ? 4
                : (dx == 0 || dy == 0) ? 2
                : 1;
            sum += in[ny * width + nx] * w;
        }
    }
    out[idx] = static_cast<unsigned char>(sum / 16 + 0.5);
}

