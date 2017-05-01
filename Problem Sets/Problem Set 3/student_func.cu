/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__global__ void shmem_min_max_kernel(float * d_out, const float * d_in, int img_size, bool is_min) {
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    if (myId < img_size) {
      sdata[tid] = d_in[myId];
    } else {
      sdata[tid] = is_min ? 1e10 : -1e10;
    }
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
          if (sdata[tid + s] >= 0) {
            sdata[tid] = is_min ? min(sdata[tid], sdata[tid + s]) : max(sdata[tid], sdata[tid + s]);
          }
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0 && myId < img_size)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

__global__ void simple_histo(unsigned int *d_bins, const float *d_in, int actual_size, const int numBins, float lumMin, float lumRange)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if (myId < actual_size) {
      float value = d_in[myId];
      int bin = max(0, min(numBins -1, (int)((value - lumMin) / lumRange * numBins)));
      atomicAdd(&(d_bins[bin]), 1);
    }
}

// from https://cudacomputing.com/the-main-page/scan-implementation-blelloch-scan/implementation-of-blelloch-scan/
__global__ void scan_bel(unsigned int* outputarray, int loop, int number)
{
  unsigned int thIdx = blockIdx.x * blockDim.x + threadIdx.x;

  int divisor = 2;
  int adder = 1;
  int temp;

  for(int i=0;i<loop;i++) {
    if(thIdx%(divisor) == divisor-1) {
      outputarray[thIdx] = outputarray[thIdx-adder]+outputarray[thIdx];
    }
    __syncthreads();
    divisor*=2;
    adder*=2;
  }

  divisor = number;
  adder = divisor/2;

  outputarray[number-1] = 0;
  for(int i=0;i<loop;i++) {
    if(thIdx%(divisor) == divisor-1) {
      temp = outputarray[thIdx];
      outputarray[thIdx] = outputarray[thIdx-adder]+outputarray[thIdx];
      outputarray[thIdx-adder] = temp;
    }
    __syncthreads();
    divisor/=2;
    adder/=2;
  }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  int blocks = ceil(sqrt(numRows * numCols));
  assert(blocks < 1024);

  float *d_intermediate;
  float *d_min_max;
  checkCudaErrors(cudaMalloc((void **) &d_intermediate, sizeof(float)*blocks*blocks)); // overallocated
  checkCudaErrors(cudaMalloc((void **) &d_min_max, sizeof(float)));
  // 1) comptue min
  shmem_min_max_kernel<<<blocks, blocks, blocks * sizeof(float)>>>
    (d_intermediate, d_logLuminance, numRows * numCols, true);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  shmem_min_max_kernel<<<1, blocks, blocks * sizeof(float)>>>
    (d_min_max, d_intermediate, blocks, true);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  cudaMemcpy(&min_logLum, d_min_max, sizeof(float), cudaMemcpyDeviceToHost);

  // do the same for max
  shmem_min_max_kernel<<<blocks, blocks, blocks * sizeof(float)>>>
    (d_intermediate, d_logLuminance, numRows * numCols, false);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  shmem_min_max_kernel<<<1, blocks, blocks * sizeof(float)>>>
    (d_min_max, d_intermediate, blocks, false);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  cudaMemcpy(&max_logLum, d_min_max, sizeof(float), cudaMemcpyDeviceToHost);

  float range_logLum = max_logLum - min_logLum;
  std::cout << "min " << min_logLum << " max " << max_logLum << std::endl;

  // 3) generate histo
  simple_histo<<<blocks, blocks>>>(d_cdf, d_logLuminance, numRows * numCols, numBins, min_logLum, range_logLum);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  int h_bins[numBins];
  cudaMemcpy(h_bins, d_cdf, sizeof(int)*numBins, cudaMemcpyDeviceToHost);
  int total = 0;
  for (int i = 0; i < numBins; ++i) {
    total += h_bins[i];
    std::cout << " " << h_bins[i];
  }
  std::cout << std::endl << "total " << total << " pixels " << numCols * numRows << std::endl;
  assert(total == numCols * numRows);

  // 4) cdf Blelloch scan, must be a power of 2
  scan_bel<<<1, numBins>>>(d_cdf, int(log2f(numBins)), numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  int h_cdf[numBins];
  cudaMemcpy(h_cdf, d_cdf, sizeof(int)*numBins, cudaMemcpyDeviceToHost);
  std::cout << "h_cdf\n";
  for (int i = 0; i < numBins; ++i) {
    std::cout << " " << h_cdf[i];
  }
  std::cout << std::endl;

  checkCudaErrors(cudaFree(d_intermediate));
  checkCudaErrors(cudaFree(d_min_max));
}
