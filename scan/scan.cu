#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result

template <int TPB>
__global__ void block_exclusive_scan(const int* __restrict__ in, int* __restrict__ out,
                                     int N, int* __restrict__ block_sums) {
    __shared__ int sh[TPB];

    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // number of valid elements in this block
    const int n = min(TPB, max(0, N - bid * TPB));

    // Initialize shared memory (zeros for inactive lanes)
    int x = (tid < n) ? in[gid] : 0;
    sh[tid] = x;
    __syncthreads();

    // Hillis–Steele inclusive scan over the first n lanes
    for (int offset = 1; offset < TPB; offset <<= 1) {
        int t = 0;
        if (tid < n && tid >= offset) t = sh[tid - offset];
        __syncthreads();
        if (tid < n && tid >= offset) sh[tid] += t;
        __syncthreads();
    }

    // Convert to exclusive
    if (tid < n && gid < N) {
        out[gid] = (tid == 0) ? 0 : sh[tid - 1];
    }

    // Write this block's total (the inclusive last valid element)
    if (block_sums && n > 0 && tid == n - 1) {
        block_sums[bid] = sh[tid];
    }
}

// Kernel 2: uniform add — add each block’s scanned offset to its elements
template <int TPB>
__global__ void add_block_offsets(int* __restrict__ out, int N,
                                  const int* __restrict__ scanned_block_sums) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    // offset must be the exclusive scan value for THIS block
    int offset = scanned_block_sums[bid];

    if (gid < N) out[gid] += offset;
}

// Recursively scan an array in-place on the GPU using the same kernels.
static void device_exclusive_scan_inplace(int* d_arr, int N) {
    constexpr int TPB = THREADS_PER_BLOCK;
    int num_blocks = (N + TPB - 1) / TPB;

    if (num_blocks <= 1) {
        // Single-block scan
        block_exclusive_scan<TPB><<<1, TPB>>>(d_arr, d_arr, N, nullptr);
        return;
    }

    // Multi-block: scan each block and collect their totals
    int* d_block_sums = nullptr;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(int));
    block_exclusive_scan<TPB><<<num_blocks, TPB>>>(d_arr, d_arr, N, d_block_sums);

    // Recursively scan the per-block totals (in-place)
    device_exclusive_scan_inplace(d_block_sums, num_blocks);

    // Uniform add offsets to each block
    add_block_offsets<TPB><<<num_blocks, TPB>>>(d_arr, N, d_block_sums);

    cudaFree(d_block_sums);
}

void exclusive_scan(const int* d_in, int N, int* d_out) {
    constexpr int TPB = THREADS_PER_BLOCK;
    int num_blocks = (N + TPB - 1) / TPB;

    if (num_blocks <= 1) {
        block_exclusive_scan<TPB><<<1, TPB>>>(d_in, d_out, N, nullptr);
        return;
    }

    // First pass: per-block exclusive scan from d_in -> d_out and collect block sums
    int* d_block_sums = nullptr;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(int));
    block_exclusive_scan<TPB><<<num_blocks, TPB>>>(d_in, d_out, N, d_block_sums);

    // Scan block sums in-place (proper recursive/global scan)
    device_exclusive_scan_inplace(d_block_sums, num_blocks);

    // Add scanned block offsets back to each block’s output
    add_block_offsets<TPB><<<num_blocks, TPB>>>(d_out, N, d_block_sums);

    cudaFree(d_block_sums);
}

//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray) {
    int* device_result;
    int* device_input;
    int N = end - inarray;

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);

    cudaMalloc((void**)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void**)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration;
}

// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {
    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration;
}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {
    return 0;
}

//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int* input, int length, int* output, int* output_length) {
    int* device_input;
    int* device_output;
    int rounded_length = nextPow2(length);

    cudaMalloc((void**)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void**)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();

    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime;
    return duration;
}

void printCudaInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
