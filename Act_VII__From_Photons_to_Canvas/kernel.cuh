// ===== kernel.cuh =====
#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>

// Pure CUDA kernels – no host code here
__global__ void gpu_linear_percentile_stretch(
    const double* in,
    unsigned char* out,
    size_t         N,
    double         v_low,
    double         v_high
);

__global__ void gpu_log_scale_stretch(
    const double* in,
    unsigned char* out,
    size_t         N,
    double         minV,
    double         range
);

__global__ void gpu_colorize(
    const unsigned char* gray,
    unsigned char*       rgb,
    size_t               N
);


__global__ void gpu_gaussian_blur(
    const unsigned char* in,
    unsigned char* out,
    size_t               N,
    long                 width,
    long                 height
);


#endif // KERNEL_CUH