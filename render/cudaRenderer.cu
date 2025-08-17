// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <driver_functions.h>
// #include <math.h>
// #include <stdio.h>

// #include <algorithm>
// #include <string>
// #include <vector>

// #include "cudaRenderer.h"
// #include "image.h"
// #include "noise.h"
// #include "sceneLoader.h"
// #include "util.h"

// ////////////////////////////////////////////////////////////////////////////////////////
// // Putting all the cuda kernels here
// ///////////////////////////////////////////////////////////////////////////////////////

// struct GlobalConstants {
//     SceneName sceneName;

//     int numCircles;
//     float* position;
//     float* velocity;
//     float* color;
//     float* radius;

//     int imageWidth;
//     int imageHeight;
//     float* imageData;
// };

// // Global variable that is in scope, but read-only, for all cuda
// // kernels.  The __constant__ modifier designates this variable will
// // be stored in special "constant" memory on the GPU. (we didn't talk
// // about this type of memory in class, but constant memory is a fast
// // place to put read-only variables).
// __constant__ GlobalConstants cuConstRendererParams;

// // read-only lookup tables used to quickly compute noise (needed by
// // advanceAnimation for the snowflake scene)
// __constant__ int cuConstNoiseYPermutationTable[256];
// __constant__ int cuConstNoiseXPermutationTable[256];
// __constant__ float cuConstNoise1DValueTable[256];

// // color ramp table needed for the color ramp lookup shader
// #define COLOR_MAP_SIZE 5
// __constant__ float cuConstColorRamp[COLOR_MAP_SIZE][3];

// // including parts of the CUDA code from external files to keep this
// // file simpler and to seperate code that should not be modified
// #include "lookupColor.cu_inl"
// #include "noiseCuda.cu_inl"

// // kernelClearImageSnowflake -- (CUDA device code)
// //
// // Clear the image, setting the image to the white-gray gradation that
// // is used in the snowflake image
// __global__ void kernelClearImageSnowflake() {
//     int imageX = blockIdx.x * blockDim.x + threadIdx.x;
//     int imageY = blockIdx.y * blockDim.y + threadIdx.y;

//     int width = cuConstRendererParams.imageWidth;
//     int height = cuConstRendererParams.imageHeight;

//     if (imageX >= width || imageY >= height)
//         return;

//     int offset = 4 * (imageY * width + imageX);
//     float shade = .4f + .45f * static_cast<float>(height - imageY) / height;
//     float4 value = make_float4(shade, shade, shade, 1.f);

//     // write to global memory: As an optimization, I use a float4
//     // store, that results in more efficient code than if I coded this
//     // up as four seperate fp32 stores.
//     *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
// }

// // kernelClearImage --  (CUDA device code)
// //
// // Clear the image, setting all pixels to the specified color rgba
// __global__ void kernelClearImage(float r, float g, float b, float a) {
//     int imageX = blockIdx.x * blockDim.x + threadIdx.x;
//     int imageY = blockIdx.y * blockDim.y + threadIdx.y;

//     int width = cuConstRendererParams.imageWidth;
//     int height = cuConstRendererParams.imageHeight;

//     if (imageX >= width || imageY >= height)
//         return;

//     int offset = 4 * (imageY * width + imageX);
//     float4 value = make_float4(r, g, b, a);

//     // write to global memory: As an optimization, I use a float4
//     // store, that results in more efficient code than if I coded this
//     // up as four seperate fp32 stores.
//     *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
// }

// // kernelAdvanceFireWorks
// //
// // Update the position of the fireworks (if circle is firework)
// __global__ void kernelAdvanceFireWorks() {
//     const float dt = 1.f / 60.f;
//     const float pi = 3.14159;
//     const float maxDist = 0.25f;

//     float* velocity = cuConstRendererParams.velocity;
//     float* position = cuConstRendererParams.position;
//     float* radius = cuConstRendererParams.radius;

//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index >= cuConstRendererParams.numCircles)
//         return;

//     if (0 <= index && index < NUM_FIREWORKS) {  // firework center; no update
//         return;
//     }

//     // determine the fire-work center/spark indices
//     int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
//     int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

//     int index3i = 3 * fIdx;
//     int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
//     int index3j = 3 * sIdx;

//     float cx = position[index3i];
//     float cy = position[index3i + 1];

//     // update position
//     position[index3j] += velocity[index3j] * dt;
//     position[index3j + 1] += velocity[index3j + 1] * dt;

//     // fire-work sparks
//     float sx = position[index3j];
//     float sy = position[index3j + 1];

//     // compute vector from firework-spark
//     float cxsx = sx - cx;
//     float cysy = sy - cy;

//     // compute distance from fire-work
//     float dist = sqrt(cxsx * cxsx + cysy * cysy);
//     if (dist > maxDist) {  // restore to starting position
//         // random starting position on fire-work's rim
//         float angle = (sfIdx * 2 * pi) / NUM_SPARKS;
//         float sinA = sin(angle);
//         float cosA = cos(angle);
//         float x = cosA * radius[fIdx];
//         float y = sinA * radius[fIdx];

//         position[index3j] = position[index3i] + x;
//         position[index3j + 1] = position[index3i + 1] + y;
//         position[index3j + 2] = 0.0f;

//         // travel scaled unit length
//         velocity[index3j] = cosA / 5.0;
//         velocity[index3j + 1] = sinA / 5.0;
//         velocity[index3j + 2] = 0.0f;
//     }
// }

// // kernelAdvanceHypnosis
// //
// // Update the radius/color of the circles
// __global__ void kernelAdvanceHypnosis() {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index >= cuConstRendererParams.numCircles)
//         return;

//     float* radius = cuConstRendererParams.radius;

//     float cutOff = 0.5f;
//     // place circle back in center after reaching threshold radisus
//     if (radius[index] > cutOff) {
//         radius[index] = 0.02f;
//     } else {
//         radius[index] += 0.01f;
//     }
// }

// // kernelAdvanceBouncingBalls
// //
// // Update the positino of the balls
// __global__ void kernelAdvanceBouncingBalls() {
//     const float dt = 1.f / 60.f;
//     const float kGravity = -2.8f;  // sorry Newton
//     const float kDragCoeff = -0.8f;
//     const float epsilon = 0.001f;

//     int index = blockIdx.x * blockDim.x + threadIdx.x;

//     if (index >= cuConstRendererParams.numCircles)
//         return;

//     float* velocity = cuConstRendererParams.velocity;
//     float* position = cuConstRendererParams.position;

//     int index3 = 3 * index;
//     // reverse velocity if center position < 0
//     float oldVelocity = velocity[index3 + 1];
//     float oldPosition = position[index3 + 1];

//     if (oldVelocity == 0.f && oldPosition == 0.f) {  // stop-condition
//         return;
//     }

//     if (position[index3 + 1] < 0 && oldVelocity < 0.f) {  // bounce ball
//         velocity[index3 + 1] *= kDragCoeff;
//     }

//     // update velocity: v = u + at (only along y-axis)
//     velocity[index3 + 1] += kGravity * dt;

//     // update positions (only along y-axis)
//     position[index3 + 1] += velocity[index3 + 1] * dt;

//     if (fabsf(velocity[index3 + 1] - oldVelocity) < epsilon && oldPosition < 0.0f && fabsf(position[index3 + 1] - oldPosition) < epsilon) {  // stop ball
//         velocity[index3 + 1] = 0.f;
//         position[index3 + 1] = 0.f;
//     }
// }

// // kernelAdvanceSnowflake -- (CUDA device code)
// //
// // move the snowflake animation forward one time step.  Updates circle
// // positions and velocities.  Note how the position of the snowflake
// // is reset if it moves off the left, right, or bottom of the screen.
// __global__ void kernelAdvanceSnowflake() {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;

//     if (index >= cuConstRendererParams.numCircles)
//         return;

//     const float dt = 1.f / 60.f;
//     const float kGravity = -1.8f;  // sorry Newton
//     const float kDragCoeff = 2.f;

//     int index3 = 3 * index;

//     float* positionPtr = &cuConstRendererParams.position[index3];
//     float* velocityPtr = &cuConstRendererParams.velocity[index3];

//     // loads from global memory
//     float3 position = *((float3*)positionPtr);
//     float3 velocity = *((float3*)velocityPtr);

//     // hack to make farther circles move more slowly, giving the
//     // illusion of parallax
//     float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f);  // clamp

//     // add some noise to the motion to make the snow flutter
//     float3 noiseInput;
//     noiseInput.x = 10.f * position.x;
//     noiseInput.y = 10.f * position.y;
//     noiseInput.z = 255.f * position.z;
//     float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
//     noiseForce.x *= 7.5f;
//     noiseForce.y *= 5.f;

//     // drag
//     float2 dragForce;
//     dragForce.x = -1.f * kDragCoeff * velocity.x;
//     dragForce.y = -1.f * kDragCoeff * velocity.y;

//     // update positions
//     position.x += velocity.x * dt;
//     position.y += velocity.y * dt;

//     // update velocities
//     velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
//     velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

//     float radius = cuConstRendererParams.radius[index];

//     // if the snowflake has moved off the left, right or bottom of
//     // the screen, place it back at the top and give it a
//     // pseudorandom x position and velocity.
//     if ((position.y + radius < 0.f) ||
//         (position.x + radius) < -0.f ||
//         (position.x - radius) > 1.f) {
//         noiseInput.x = 255.f * position.x;
//         noiseInput.y = 255.f * position.y;
//         noiseInput.z = 255.f * position.z;
//         noiseForce = cudaVec2CellNoise(noiseInput, index);

//         position.x = .5f + .5f * noiseForce.x;
//         position.y = 1.35f + radius;

//         // restart from 0 vertical velocity.  Choose a
//         // pseudo-random horizontal velocity.
//         velocity.x = 2.f * noiseForce.y;
//         velocity.y = 0.f;
//     }

//     // store updated positions and velocities to global memory
//     *((float3*)positionPtr) = position;
//     *((float3*)velocityPtr) = velocity;
// }

// // shadePixel -- (CUDA device code)
// //
// // given a pixel and a circle, determines the contribution to the
// // pixel from the circle.  Update of the image is done in this
// // function.  Called by kernelRenderCircles()
// __device__ __inline__ void
// shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {
//     float diffX = p.x - pixelCenter.x;
//     float diffY = p.y - pixelCenter.y;
//     float pixelDist = diffX * diffX + diffY * diffY;

//     float rad = cuConstRendererParams.radius[circleIndex];
//     float maxDist = rad * rad;

//     // circle does not contribute to the image
//     if (pixelDist > maxDist)
//         return;

//     float3 rgb;
//     float alpha;

//     // there is a non-zero contribution.  Now compute the shading value

//     // suggestion: This conditional is in the inner loop.  Although it
//     // will evaluate the same for all threads, there is overhead in
//     // setting up the lane masks etc to implement the conditional.  It
//     // would be wise to perform this logic outside of the loop next in
//     // kernelRenderCircles.  (If feeling good about yourself, you
//     // could use some specialized template magic).
//     if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
//         const float kCircleMaxAlpha = .5f;
//         const float falloffScale = 4.f;

//         float normPixelDist = sqrt(pixelDist) / rad;
//         rgb = lookupColor(normPixelDist);

//         float maxAlpha = .6f + .4f * (1.f - p.z);
//         maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);  // kCircleMaxAlpha * clamped value
//         alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

//     } else {
//         // simple: each circle has an assigned color
//         int index3 = 3 * circleIndex;
//         rgb = *(float3*)&(cuConstRendererParams.color[index3]);
//         alpha = .5f;
//     }

//     float oneMinusAlpha = 1.f - alpha;

//     // BEGIN SHOULD-BE-ATOMIC REGION
//     // global memory read

//     float4 existingColor = *imagePtr;
//     float4 newColor;
//     newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
//     newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
//     newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
//     newColor.w = alpha + existingColor.w;

//     // global memory write
//     *imagePtr = newColor;

//     // END SHOULD-BE-ATOMIC REGION
// }

// // No global memory here — blend into accum in registers
// __device__ __forceinline__ void shadePixelAccum(int circleIndex, float2 pixelCenter, float3 p,
//                                                 float4& accum, bool snowScene) {
//     const float dx = p.x - pixelCenter.x;
//     const float dy = p.y - pixelCenter.y;
//     const float d2 = dx * dx + dy * dy;

//     const float r = cuConstRendererParams.radius[circleIndex];
//     const float r2 = r * r;
//     if (d2 > r2) return;

//     float3 rgb;
//     float alpha;
//     if (snowScene) {
//         const float kCircleMaxAlpha = .5f;
//         const float falloffScale = 4.f;
//         const float normDist = sqrtf(d2) / r;
//         rgb = lookupColor(normDist);
//         float maxAlpha = .6f + .4f * (1.f - p.z);
//         maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
//         alpha = maxAlpha * expf(-falloffScale * normDist * normDist);
//     } else {
//         rgb = *(float3*)&(cuConstRendererParams.color[3 * circleIndex]);
//         alpha = .5f;
//     }

//     const float oma = 1.f - alpha;
//     accum.x = alpha * rgb.x + oma * accum.x;
//     accum.y = alpha * rgb.y + oma * accum.y;
//     accum.z = alpha * rgb.z + oma * accum.z;
//     accum.w = accum.w + alpha;
// }

// // kernelRenderCircles -- (CUDA device code)
// //
// // Each thread renders a circle.  Since there is no protection to
// // ensure order of update or mutual exclusion on the output image, the
// // resulting image will be incorrect.
// __device__ __forceinline__ int circleInBoxConservative(float cx, float cy, float r,
//                                                        float boxL, float boxR, float boxT, float boxB) {
//     // boxT <= y <= boxB in screen space
//     return (cx >= (boxL - r)) && (cx <= (boxR + r)) &&
//            (cy >= (boxT - r)) && (cy <= (boxB + r));
// }

// __global__ void kernelRenderCircles() {
//     int global_i = blockIdx.x * blockDim.x + threadIdx.x;
//     int global_j = blockIdx.y * blockDim.y + threadIdx.y;

//     short imageWidth = cuConstRendererParams.imageWidth;
//     short imageHeight = cuConstRendererParams.imageHeight;

//     float invWidth = 1.f / imageWidth;
//     float invHeight = 1.f / imageHeight;

//     if (global_i >= imageWidth || global_j >= imageHeight) {
//         return;
//     }

//     const int tileMinX = blockIdx.x * blockDim.x;
//     const int tileMaxX = min(tileMinX + blockDim.x, imageWidth);
//     const int tileMinY = blockIdx.y * blockDim.y;
//     const int tileMaxY = min(tileMinY + blockDim.y, imageHeight);

//     float boxL = static_cast<float>(tileMinX) * invWidth;
//     float boxR = static_cast<float>(tileMaxX) * invWidth;
//     float boxT = static_cast<float>(tileMinY) * invHeight;
//     float boxB = static_cast<float>(tileMaxY) * invHeight;

//     float normX = invWidth * (static_cast<float>(global_i) + 0.5f);
//     float normY = invHeight * (static_cast<float>(global_j) + 0.5f);
//     float2 pixelCenterNorm = make_float2(normX, normY);
//     float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (global_j * imageWidth + global_i)]);
//     float4 accum = *imgPtr;

//     const bool snow = (cuConstRendererParams.sceneName == SNOWFLAKES) ||
//                       (cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME);

//     int numCircles = cuConstRendererParams.numCircles;
//     for (int circleIndex = 0; circleIndex < numCircles; circleIndex++) {
//         // read position and radius
//         float3 p = *(float3*)(&cuConstRendererParams.position[3 * circleIndex]);
//         float rad = cuConstRendererParams.radius[circleIndex];

//         if (!circleInBoxConservative(p.x, p.y, rad, boxL, boxR, boxT, boxB)) {
//             continue;
//         }

//         shadePixelAccum(circleIndex, pixelCenterNorm, p, accum, snow);
//     }

//     *imgPtr = accum;
// }

// ////////////////////////////////////////////////////////////////////////////////////////

// CudaRenderer::CudaRenderer() {
//     image = NULL;

//     numCircles = 0;
//     position = NULL;
//     velocity = NULL;
//     color = NULL;
//     radius = NULL;

//     cudaDevicePosition = NULL;
//     cudaDeviceVelocity = NULL;
//     cudaDeviceColor = NULL;
//     cudaDeviceRadius = NULL;
//     cudaDeviceImageData = NULL;
// }

// CudaRenderer::~CudaRenderer() {
//     if (image) {
//         delete image;
//     }

//     if (position) {
//         delete[] position;
//         delete[] velocity;
//         delete[] color;
//         delete[] radius;
//     }

//     if (cudaDevicePosition) {
//         cudaFree(cudaDevicePosition);
//         cudaFree(cudaDeviceVelocity);
//         cudaFree(cudaDeviceColor);
//         cudaFree(cudaDeviceRadius);
//         cudaFree(cudaDeviceImageData);
//     }
// }

// const Image*
// CudaRenderer::getImage() {
//     // need to copy contents of the rendered image from device memory
//     // before we expose the Image object to the caller

//     printf("Copying image data from device\n");

//     cudaMemcpy(image->data,
//                cudaDeviceImageData,
//                sizeof(float) * 4 * image->width * image->height,
//                cudaMemcpyDeviceToHost);

//     return image;
// }

// void CudaRenderer::loadScene(SceneName scene) {
//     sceneName = scene;
//     loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
// }

// void CudaRenderer::setup() {
//     int deviceCount = 0;
//     std::string name;
//     cudaError_t err = cudaGetDeviceCount(&deviceCount);

//     printf("---------------------------------------------------------\n");
//     printf("Initializing CUDA for CudaRenderer\n");
//     printf("Found %d CUDA devices\n", deviceCount);

//     for (int i = 0; i < deviceCount; i++) {
//         cudaDeviceProp deviceProps;
//         cudaGetDeviceProperties(&deviceProps, i);
//         name = deviceProps.name;

//         printf("Device %d: %s\n", i, deviceProps.name);
//         printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
//         printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
//         printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
//     }
//     printf("---------------------------------------------------------\n");

//     // By this time the scene should be loaded.  Now copy all the key
//     // data structures into device memory so they are accessible to
//     // CUDA kernels
//     //
//     // See the CUDA Programmer's Guide for descriptions of
//     // cudaMalloc and cudaMemcpy

//     cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
//     cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
//     cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
//     cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
//     cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

//     cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
//     cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
//     cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
//     cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

//     // Initialize parameters in constant memory.  We didn't talk about
//     // constant memory in class, but the use of read-only constant
//     // memory here is an optimization over just sticking these values
//     // in device global memory.  NVIDIA GPUs have a few special tricks
//     // for optimizing access to constant memory.  Using global memory
//     // here would have worked just as well.  See the Programmer's
//     // Guide for more information about constant memory.

//     GlobalConstants params;
//     params.sceneName = sceneName;
//     params.numCircles = numCircles;
//     params.imageWidth = image->width;
//     params.imageHeight = image->height;
//     params.position = cudaDevicePosition;
//     params.velocity = cudaDeviceVelocity;
//     params.color = cudaDeviceColor;
//     params.radius = cudaDeviceRadius;
//     params.imageData = cudaDeviceImageData;

//     cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

//     // also need to copy over the noise lookup tables, so we can
//     // implement noise on the GPU
//     int* permX;
//     int* permY;
//     float* value1D;
//     getNoiseTables(&permX, &permY, &value1D);
//     cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
//     cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
//     cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

//     // last, copy over the color table that's used by the shading
//     // function for circles in the snowflake demo

//     float lookupTable[COLOR_MAP_SIZE][3] = {
//         {1.f, 1.f, 1.f},
//         {1.f, 1.f, 1.f},
//         {.8f, .9f, 1.f},
//         {.8f, .9f, 1.f},
//         {.8f, 0.8f, 1.f},
//     };

//     cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);
// }

// // allocOutputImage --
// //
// // Allocate buffer the renderer will render into.  Check status of
// // image first to avoid memory leak.
// void CudaRenderer::allocOutputImage(int width, int height) {
//     if (image)
//         delete image;
//     image = new Image(width, height);
// }

// // clearImage --
// //
// // Clear's the renderer's target image.  The state of the image after
// // the clear depends on the scene being rendered.
// void CudaRenderer::clearImage() {
//     // 256 threads per block is a healthy number
//     dim3 blockDim(16, 16, 1);
//     dim3 gridDim(
//         (image->width + blockDim.x - 1) / blockDim.x,
//         (image->height + blockDim.y - 1) / blockDim.y);

//     if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
//         kernelClearImageSnowflake<<<gridDim, blockDim>>>();
//     } else {
//         kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
//     }
//     cudaDeviceSynchronize();
// }

// // advanceAnimation --
// //
// // Advance the simulation one time step.  Updates all circle positions
// // and velocities
// void CudaRenderer::advanceAnimation() {
//     // 256 threads per block is a healthy number
//     dim3 blockDim(256, 1);
//     dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

//     // only the snowflake scene has animation
//     if (sceneName == SNOWFLAKES) {
//         kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
//     } else if (sceneName == BOUNCING_BALLS) {
//         kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
//     } else if (sceneName == HYPNOSIS) {
//         kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
//     } else if (sceneName == FIREWORKS) {
//         kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
//     }
//     cudaDeviceSynchronize();
// }

// void CudaRenderer::render() {
//     // 256 threads per block is a healthy number
//     dim3 blockDim(16, 16);
//     dim3 gridDim((image->width + blockDim.x - 1) / blockDim.x, (image->height + blockDim.y - 1) / blockDim.y);

//     kernelRenderCircles<<<gridDim, blockDim>>>();
//     cudaDeviceSynchronize();
// }

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <string>
#include <vector>

// Thrust for device scans/sort/reduce
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

// ===== Tuning knobs =====
#ifndef TILE_W
#define TILE_W 32
#endif
#ifndef TILE_H
#define TILE_H 8
#endif
#ifndef FUSE_CLEAR
// 0 = read cleared framebuffer to match reference bit-for-bit
// 1 = compute clear color in-kernel and DO NOT call clearImage() in the driver
#define FUSE_CLEAR 0
#endif

////////////////////////////////////////////////////////////////////////////////////////
// CUDA-side globals
////////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {
    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int cuConstNoiseYPermutationTable[256];
__constant__ int cuConstNoiseXPermutationTable[256];
__constant__ float cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float cuConstColorRamp[COLOR_MAP_SIZE][3];

// including parts of the CUDA code from external files to keep this
// file simpler and to separate code that should not be modified
#include "lookupColor.cu_inl"
#include "noiseCuda.cu_inl"

////////////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ bool circleIntersectsBoxNorm(float2 c, float r, float2 bmin, float2 bmax) {
    // find nearest point in box to circle center
    float nx = fmaxf(bmin.x, fminf(c.x, bmax.x));
    float ny = fmaxf(bmin.y, fminf(c.y, bmax.y));
    float dx = c.x - nx;
    float dy = c.y - ny;
    return (dx * dx + dy * dy) <= (r * r);
}

////////////////////////////////////////////////////////////////////////////////////////
// Clearing kernels (unchanged)
////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernelClearImageSnowflake() {
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height) return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height - imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

__global__ void kernelClearImage(float r, float g, float b, float a) {
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height) return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

////////////////////////////////////////////////////////////////////////////////////////
/* Animation kernels (unchanged from starter) */
////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159f;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) return;

    if (0 <= index && index < NUM_FIREWORKS) return;

    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i + 1];

    position[index3j] += velocity[index3j] * dt;
    position[index3j + 1] += velocity[index3j + 1] * dt;

    float sx = position[index3j];
    float sy = position[index3j + 1];

    float cxsx = sx - cx;
    float cysy = sy - cy;

    float dist = sqrtf(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) {
        float angle = (sfIdx * 2 * pi) / NUM_SPARKS;
        float sinA = sinf(angle);
        float cosA = cosf(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j + 1] = position[index3i + 1] + y;
        position[index3j + 2] = 0.0f;

        velocity[index3j] = cosA / 5.0f;
        velocity[index3j + 1] = sinA / 5.0f;
        velocity[index3j + 2] = 0.0f;
    }
}

__global__ void kernelAdvanceHypnosis() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) return;

    float* radius = cuConstRendererParams.radius;

    const float cutOff = 0.5f;
    if (radius[index] > cutOff) {
        radius[index] = 0.02f;
    } else {
        radius[index] += 0.01f;
    }
}

__global__ void kernelAdvanceBouncingBalls() {
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f;
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) return;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;

    int index3 = 3 * index;
    float oldVelocity = velocity[index3 + 1];
    float oldPosition = position[index3 + 1];

    if (oldVelocity == 0.f && oldPosition == 0.f) return;

    if (position[index3 + 1] < 0.f && oldVelocity < 0.f) {
        velocity[index3 + 1] *= kDragCoeff;
    }

    velocity[index3 + 1] += kGravity * dt;
    position[index3 + 1] += velocity[index3 + 1] * dt;

    if (fabsf(velocity[index3 + 1] - oldVelocity) < epsilon &&
        oldPosition < 0.0f &&
        fabsf(position[index3 + 1] - oldPosition) < epsilon) {
        velocity[index3 + 1] = 0.f;
        position[index3 + 1] = 0.f;
    }
}

__global__ void kernelAdvanceSnowflake() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f;
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    float forceScaling = fminf(fmaxf(1.f - position.z, .1f), 1.f);

    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;

    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // (kept as in starter)
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    if ((position.y + radius < 0.f) ||
        (position.x + radius) < -0.f ||
        (position.x - radius) > 1.f) {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

////////////////////////////////////////////////////////////////////////////////////////
// Shade helper (pixel-owned accum; no global RMW)
////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void shadePixelAccum(int circleIndex, float2 pixelCenter, float3 p,
                                                float4& accum, bool snowScene) {
    const float dx = p.x - pixelCenter.x;
    const float dy = p.y - pixelCenter.y;
    const float d2 = dx * dx + dy * dy;

    const float r = cuConstRendererParams.radius[circleIndex];
    const float r2 = r * r;
    if (d2 > r2) return;

    float3 rgb;
    float alpha;

    if (snowScene) {
        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;
        const float normDist = sqrtf(d2) / r;

        rgb = lookupColor(normDist);

        float maxAlpha = .6f + .4f * (1.f - p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
        alpha = maxAlpha * expf(-falloffScale * normDist * normDist);
    } else {
        rgb = *(float3*)&(cuConstRendererParams.color[3 * circleIndex]);
        alpha = .5f;
    }

    const float oma = 1.f - alpha;
    accum.x = alpha * rgb.x + oma * accum.x;
    accum.y = alpha * rgb.y + oma * accum.y;
    accum.z = alpha * rgb.z + oma * accum.z;
    accum.w = accum.w + alpha;
}

////////////////////////////////////////////////////////////////////////////////////////
// GPU binning: count (with precise circle-vs-tile test)
////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernelCountPairs(int tilesX, int tilesY, int imgW, int imgH,
                                 int* __restrict__ counts) {
    const int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= cuConstRendererParams.numCircles) return;

    const float* pos = cuConstRendererParams.position;
    const float* rad = cuConstRendererParams.radius;

    const float3 p = *((const float3*)&pos[3 * ci]);
    const float r = rad[ci];

    // Reference-style pixel bounds (half-open)
    const float minXf = p.x - r;
    const float maxXf = p.x + r;
    const float minYf = p.y - r;
    const float maxYf = p.y + r;

    int screenMinX = max(0, min(static_cast<int>(minXf * imgW), imgW));
    int screenMaxX = max(0, min(static_cast<int>(maxXf * imgW) + 1, imgW));
    int screenMinY = max(0, min(static_cast<int>(minYf * imgH), imgH));
    int screenMaxY = max(0, min(static_cast<int>(maxYf * imgH) + 1, imgH));

    if (screenMinX >= screenMaxX || screenMinY >= screenMaxY) {
        counts[ci] = 0;
        return;
    }

    int tminX = screenMinX / TILE_W;
    int tmaxX = (screenMaxX - 1) / TILE_W;
    int tminY = screenMinY / TILE_H;
    int tmaxY = (screenMaxY - 1) / TILE_H;

    tminX = max(0, min(tminX, tilesX - 1));
    tmaxX = max(0, min(tmaxX, tilesX - 1));
    tminY = max(0, min(tminY, tilesY - 1));
    tmaxY = max(0, min(tmaxY, tilesY - 1));

    const float invW = 1.f / imgW;
    const float invH = 1.f / imgH;
    const float2 c = make_float2(p.x, p.y);

    int ccount = 0;
    for (int ty = tminY; ty <= tmaxY; ++ty) {
        const float boxMinY = (ty * TILE_H) * invH;
        const float boxMaxY = (min((ty + 1) * TILE_H, imgH)) * invH;
        for (int tx = tminX; tx <= tmaxX; ++tx) {
            const float boxMinX = (tx * TILE_W) * invW;
            const float boxMaxX = (min((tx + 1) * TILE_W, imgW)) * invW;

            if (circleIntersectsBoxNorm(c, r, make_float2(boxMinX, boxMinY),
                                        make_float2(boxMaxX, boxMaxY)))
                ++ccount;
        }
    }
    counts[ci] = ccount;
}

////////////////////////////////////////////////////////////////////////////////////////
// GPU binning: write (tileId, circleId) pairs using per-circle offsets
////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernelWritePairs(int tilesX, int tilesY, int imgW, int imgH,
                                 const int* __restrict__ circleOffsets,
                                 int* __restrict__ outTileKeys,
                                 int* __restrict__ outCircleVals) {
    const int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= cuConstRendererParams.numCircles) return;

    const float* pos = cuConstRendererParams.position;
    const float* rad = cuConstRendererParams.radius;

    const float3 p = *((const float3*)&pos[3 * ci]);
    const float r = rad[ci];

    const float minXf = p.x - r;
    const float maxXf = p.x + r;
    const float minYf = p.y - r;
    const float maxYf = p.y + r;

    int screenMinX = max(0, min(static_cast<int>(minXf * imgW), imgW));
    int screenMaxX = max(0, min(static_cast<int>(maxXf * imgW) + 1, imgW));
    int screenMinY = max(0, min(static_cast<int>(minYf * imgH), imgH));
    int screenMaxY = max(0, min(static_cast<int>(maxYf * imgH) + 1, imgH));

    if (screenMinX >= screenMaxX || screenMinY >= screenMaxY) return;

    int tminX = screenMinX / TILE_W;
    int tmaxX = (screenMaxX - 1) / TILE_W;
    int tminY = screenMinY / TILE_H;
    int tmaxY = (screenMaxY - 1) / TILE_H;

    tminX = max(0, min(tminX, tilesX - 1));
    tmaxX = max(0, min(tmaxX, tilesX - 1));
    tminY = max(0, min(tminY, tilesY - 1));
    tmaxY = max(0, min(tmaxY, tilesY - 1));

    const float invW = 1.f / imgW;
    const float invH = 1.f / imgH;
    const float2 c = make_float2(p.x, p.y);

    int base = circleOffsets[ci];
    for (int ty = tminY; ty <= tmaxY; ++ty) {
        const float boxMinY = (ty * TILE_H) * invH;
        const float boxMaxY = (min((ty + 1) * TILE_H, imgH)) * invH;
        const int rowBase = ty * tilesX;
        for (int tx = tminX; tx <= tmaxX; ++tx) {
            const float boxMinX = (tx * TILE_W) * invW;
            const float boxMaxX = (min((tx + 1) * TILE_W, imgW)) * invW;

            if (circleIntersectsBoxNorm(c, r, make_float2(boxMinX, boxMinY),
                                        make_float2(boxMaxX, boxMaxY))) {
                const int tileId = rowBase + tx;
                outTileKeys[base] = tileId;  // key
                outCircleVals[base] = ci;    // value
                ++base;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// Tile kernel (shared-memory broadcast; one block per ACTIVE tile segment)
////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernelRenderActiveTilesShared(const int* __restrict__ activeTileIds,
                                              const int* __restrict__ activeOffsets,
                                              const int* __restrict__ circleIdxSorted,
                                              int numActiveTiles) {
    const int width = cuConstRendererParams.imageWidth;
    const int height = cuConstRendererParams.imageHeight;

    const int tilesX = (width + TILE_W - 1) / TILE_W;

    const int aIdx = blockIdx.x;
    if (aIdx >= numActiveTiles) return;

    const int tileId = activeTileIds[aIdx];
    const int tileX = tileId % tilesX;
    const int tileY = tileId / tilesX;

    // Per-thread pixel
    const int px = tileX * TILE_W + threadIdx.x;
    const int py = tileY * TILE_H + threadIdx.y;

    // Keep OOB threads alive to honor __syncthreads
    const bool inBounds = (px < width) && (py < height);

    float4* pixPtr = nullptr;
    if (inBounds) {
        pixPtr = (float4*)(&cuConstRendererParams.imageData[4 * (py * width + px)]);
    }

    // Base color
    float4 accum;
#if FUSE_CLEAR
    if ((cuConstRendererParams.sceneName == SNOWFLAKES) ||
        (cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME)) {
        float shade = .4f + .45f * (float)(height - py) / height;
        accum = make_float4(shade, shade, shade, 1.f);
    } else {
        accum = make_float4(1.f, 1.f, 1.f, 1.f);
    }
#else
    accum = inBounds ? *pixPtr : make_float4(0, 0, 0, 0);
#endif

    const float invW = 1.f / width;
    const float invH = 1.f / height;
    float2 pcenter = make_float2(0.f, 0.f);
    if (inBounds) {
        pcenter = make_float2(invW * (px + 0.5f), invH * (py + 0.5f));
    }

    const bool snow = (cuConstRendererParams.sceneName == SNOWFLAKES) ||
                      (cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME);

    // Shared broadcast of one circle per iteration
    __shared__ float3 s_p;
    __shared__ float s_rad;
    __shared__ float3 s_rgb;  // non-snow only

    const int begin = activeOffsets[aIdx];
    const int end = activeOffsets[aIdx + 1];

#pragma unroll 1
    for (int k = begin; k < end; ++k) {
        const int ci = circleIdxSorted[k];
        const int idx3 = 3 * ci;

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            s_p = *((const float3*)&cuConstRendererParams.position[idx3]);
            s_rad = cuConstRendererParams.radius[ci];
            if (!snow) s_rgb = *((const float3*)&cuConstRendererParams.color[idx3]);
        }
        __syncthreads();

        if (inBounds) {
            const float dx = s_p.x - pcenter.x;
            const float dy = s_p.y - pcenter.y;
            const float d2 = dx * dx + dy * dy;
            const float r2 = s_rad * s_rad;

            if (d2 <= r2) {
                float3 rgb;
                float alpha;
                if (snow) {
                    const float kCircleMaxAlpha = .5f;
                    const float falloffScale = 4.f;
                    const float normDist = sqrtf(d2) / s_rad;
                    rgb = lookupColor(normDist);
                    float maxAlpha = .6f + .4f * (1.f - s_p.z);
                    maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
                    alpha = maxAlpha * expf(-falloffScale * normDist * normDist);
                } else {
                    rgb = s_rgb;
                    alpha = .5f;
                }

                const float oma = 1.f - alpha;
                accum.x = alpha * rgb.x + oma * accum.x;
                accum.y = alpha * rgb.y + oma * accum.y;
                accum.z = alpha * rgb.z + oma * accum.z;
                accum.w = accum.w + alpha;
            }
        }
        __syncthreads();
    }

    if (inBounds) {
        *pixPtr = accum;  // single coalesced write per pixel
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// Host-side renderer
////////////////////////////////////////////////////////////////////////////////////////

CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {
    if (image) {
        delete image;
    }

    if (position) {
        delete[] position;
        delete[] velocity;
        delete[] color;
        delete[] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image* CudaRenderer::getImage() {
    printf("Copying image data from device\n");
    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);
    return image;
}

void CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void CudaRenderer::setup() {
    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    // Allocate device buffers
    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // noise tables
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // color ramp
    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };
    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);
}

void CudaRenderer::allocOutputImage(int width, int height) {
    if (image) delete image;
    image = new Image(width, height);
}

void CudaRenderer::clearImage() {
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((image->width + blockDim.x - 1) / blockDim.x,
                 (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

void CudaRenderer::advanceAnimation() {
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();
}

static inline bool isStaticScene(SceneName s) {
    // If you wanted, you could cache binning for static scenes; for now we rebuild each frame on GPU.
    return !(s == SNOWFLAKES || s == BOUNCING_BALLS || s == HYPNOSIS || s == FIREWORKS);
}

void CudaRenderer::render() {
    // Keep constant params in sync (width/height may change)
    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;
    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    const int width = image->width;
    const int height = image->height;
    const int tilesX = (width + TILE_W - 1) / TILE_W;
    const int tilesY = (height + TILE_H - 1) / TILE_H;

    // --------------------------
    // GPU binning pipeline
    // --------------------------
    thrust::device_vector<int> dCounts(numCircles, 0);
    thrust::device_vector<int> dOffsets(numCircles, 0);

    {
        dim3 block(256, 1);
        dim3 grid((numCircles + block.x - 1) / block.x);
        kernelCountPairs<<<grid, block>>>(tilesX, tilesY, width, height,
                                          thrust::raw_pointer_cast(dCounts.data()));
    }
    cudaDeviceSynchronize();

    // total number of (tile, circle) pairs
    int totalPairs = thrust::reduce(dCounts.begin(), dCounts.end(), 0, thrust::plus<int>());

    if (totalPairs == 0) {
        // Nothing to do (blank frame); just return
        return;
    }

    // offsets per circle
    thrust::exclusive_scan(dCounts.begin(), dCounts.end(), dOffsets.begin());

    thrust::device_vector<int> dTileKeys(totalPairs);
    thrust::device_vector<int> dCircleVals(totalPairs);

    {
        dim3 block(256, 1);
        dim3 grid((numCircles + block.x - 1) / block.x);
        kernelWritePairs<<<grid, block>>>(tilesX, tilesY, width, height,
                                          thrust::raw_pointer_cast(dOffsets.data()),
                                          thrust::raw_pointer_cast(dTileKeys.data()),
                                          thrust::raw_pointer_cast(dCircleVals.data()));
    }
    cudaDeviceSynchronize();

    // Sort pairs by tile key, STABLY, so circle order within a tile is preserved
    thrust::stable_sort_by_key(dTileKeys.begin(), dTileKeys.end(), dCircleVals.begin());

    // Build active tile list and segment lengths: reduce_by_key over sorted keys
    thrust::device_vector<int> dActiveTileIds(totalPairs);
    thrust::device_vector<int> dActiveCounts(totalPairs);
    auto newEnds = thrust::reduce_by_key(dTileKeys.begin(), dTileKeys.end(),
                                         thrust::make_constant_iterator(1),
                                         dActiveTileIds.begin(),
                                         dActiveCounts.begin());
    const int numActiveTiles = static_cast<int>(newEnds.first - dActiveTileIds.begin());
    dActiveTileIds.resize(numActiveTiles);
    dActiveCounts.resize(numActiveTiles);

    // Build segment offsets into dCircleVals (size = numActiveTiles+1)
    thrust::device_vector<int> dActiveOffsets(numActiveTiles + 1);
    thrust::exclusive_scan(dActiveCounts.begin(), dActiveCounts.end(), dActiveOffsets.begin());
    // last offset = totalPairs
    dActiveOffsets[numActiveTiles] = totalPairs;

    // --------------------------
    // Render only active tiles
    // --------------------------
    dim3 blockDim(TILE_W, TILE_H, 1);
    dim3 gridDim(numActiveTiles, 1, 1);

    kernelRenderActiveTilesShared<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(dActiveTileIds.data()),
        thrust::raw_pointer_cast(dActiveOffsets.data()),
        thrust::raw_pointer_cast(dCircleVals.data()),
        numActiveTiles);

    cudaDeviceSynchronize();
}