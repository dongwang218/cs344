//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

const int K = 1024;

__global__ void bit_digit_histo(unsigned int * d_bins, unsigned int *d_digit_mask, unsigned int * const d_inputVals, int numElems, int which_digit) {
  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  if (myId < numElems) {
    int index = (d_inputVals[myId] & (1u << which_digit)) >> which_digit;
    atomicAdd(&(d_bins[index]), 1);
    d_digit_mask[myId] = index;
  }
}

// work on [start.. start+K], then we already have a startValue prefixSum to begin with
// work on summing digit
__global__ void scan_hillis(unsigned int * d_out, unsigned int * const d_digits,
                            int start_index, int numElems, unsigned int digit,
                            int zero_value) {
  assert( blockIdx.x == 0); // only work for one threadblock
  unsigned int thIdx = blockIdx.x * blockDim.x + threadIdx.x + start_index;

  if (thIdx >= numElems) {
    d_out[thIdx] = 0;
  } else {
    d_out[thIdx] = (d_digits[thIdx] == digit) ? 1 : 0;
    if (threadIdx.x == 0) {
      d_out[thIdx] += start_index > 0 ? d_out[start_index-1] : zero_value; // from last sequential run
    }
  }
  __syncthreads();


  for (int d = 0, step = 1; d < 10; ++d, step *= 2) {
    int temp;
    if (threadIdx.x >= step) {
      temp = d_out[thIdx] + d_out[thIdx - step];
    }
    __syncthreads();
    if (threadIdx.x >= step) {
      d_out[thIdx] = temp;
    }
    __syncthreads();
  }
}

__global__ void move(unsigned int * const d_input_vals, unsigned int * const d_input_pos,
                     unsigned int * const d_output_vals, unsigned int * const d_output_pos,
                     unsigned int * d_digits, unsigned int * d_position, int numElems, unsigned int digit) {
  unsigned int thIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thIdx < numElems && d_digits[thIdx] == digit) {
    int where = thIdx == 0 ? 0 : d_position[thIdx-1];
    d_output_vals[where] = d_input_vals[thIdx];
    d_output_pos[where] = d_input_pos[thIdx];
  }
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  //TODO
  //PUT YOUR SORT HERE
  unsigned int* d_bins;
  checkCudaErrors(cudaMalloc((void**)&d_bins, sizeof(unsigned int)*2));

  int over_size = int((numElems + (K - 1))/ K) * K;
  unsigned int *d_prefix_sum, *d_digits;
  // the desired location
  checkCudaErrors(cudaMalloc((void**)&d_prefix_sum, sizeof(unsigned int)*over_size));
  // whether that bit is 1 or 0
  checkCudaErrors(cudaMalloc((void**)&d_digits, sizeof(unsigned int)*over_size));

  unsigned int * d_input_vals = d_outputVals;
  unsigned int * d_input_pos = d_outputPos;
  unsigned int * d_output_vals = d_inputVals;
  unsigned int * d_output_pos = d_inputPos;

  for (int which_bit = 0; which_bit < 32; ++which_bit) {
    unsigned int * temp = d_input_vals;
    d_input_vals = d_output_vals;
    d_output_vals = temp;

    temp = d_input_pos;
    d_input_pos = d_output_pos;
    d_output_pos = temp;


    // bit histogram
    checkCudaErrors(cudaMemset(d_bins, 0, sizeof(unsigned int)*2));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    bit_digit_histo<<<over_size/K, K>>>(d_bins, d_digits, d_input_vals, numElems, which_bit);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    unsigned int h_bins[2];
    checkCudaErrors(cudaMemcpy(h_bins, d_bins, sizeof(unsigned int)*2, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    std::cout << "Iteration " << which_bit << " histo " << h_bins[0] << ", " << h_bins[1] << std::endl;

    // location
    for (int digit = 0, zero_value = 0; digit <= 1; ++digit, zero_value = h_bins[0]) {
      for (int start = 0; start < numElems; start += K) {
        scan_hillis<<<1, K>>>(d_prefix_sum, d_digits, start, numElems, digit, zero_value);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      }
      move<<<over_size/K, K>>>(d_input_vals, d_input_pos, d_output_vals, d_output_pos,
                               d_digits, d_prefix_sum, numElems, digit);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }
  }
  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(int)*numElems, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, sizeof(int)*numElems, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaFree(d_bins));
  checkCudaErrors(cudaFree(d_prefix_sum));
  checkCudaErrors(cudaFree(d_digits));

}
