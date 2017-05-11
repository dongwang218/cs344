//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.

      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly -
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>


struct splitChannels : thrust::unary_function<uchar4, thrust::tuple<unsigned char, unsigned char, unsigned char> >{
  __host__ __device__
  thrust::tuple<unsigned char, unsigned char, unsigned char> operator()(uchar4 pixel) {
    return thrust::make_tuple(pixel.x, pixel.y, pixel.z);
  }
};

struct combineChannels : thrust::unary_function<thrust::tuple<float, float, float>, uchar4> {
  __host__ __device__
  uchar4 operator()(thrust::tuple<float, float, float> t) {
    return make_uchar4(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t), 255);
  }
};

struct createMask : thrust::unary_function<uchar4, unsigned char >{
  __host__ __device__
  unsigned char operator()(uchar4 pixel) {
    return (pixel.x + pixel.y + pixel.z < 3*255) ? 1 : 0;
  }
};

struct charToFloat : thrust::unary_function<unsigned char, float >{
  __host__ __device__
  float operator()(unsigned char v) {
    return float(v);
  }
};

struct blendInterior : thrust::unary_function<thrust::tuple<uchar4, unsigned char, uchar4>, uchar4> {
  __host__ __device__
  uchar4 operator()(thrust::tuple<uchar4, unsigned char, uchar4> t) {
    return thrust::get<1>(t) ? thrust::get<0>(t) : thrust::get<2>(t);
  }
};

__global__ void compute_border(unsigned char *d_borderPixels, unsigned char *d_strictInteriorPixels,
                               unsigned char *d_mask,
                               const size_t numRowsSource, const size_t numColsSource) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int index = r * numColsSource + c;

  if (c >= numColsSource || r >= numRowsSource) return;

  if (d_mask[index] == 0) {
    d_borderPixels[index] = 0;
    d_strictInteriorPixels[index] = 0;
  } else if (d_mask[(r -1) * numColsSource + c] && d_mask[(r + 1) * numColsSource + c] &&
             d_mask[r * numColsSource + c - 1] && d_mask[r * numColsSource + c + 1]) {
    d_strictInteriorPixels[index] = 1;
    d_borderPixels[index] = 0;
  }
  else {
    d_strictInteriorPixels[index] = 0;
    d_borderPixels[index] = 1;
  }
}

__global__ void computeG(unsigned char *d_channel,
                         float * d_g,
                         int numColsSource, int numRowsSource,
                         unsigned char *d_strictInteriorPixels) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int offset = r * numColsSource + c;

  if (c >= numColsSource || r >= numRowsSource) return;
  float sum = 0;
  if (d_strictInteriorPixels[offset]) {
    sum = 4.f * d_channel[offset];
    sum -= (float)d_channel[offset - 1] + (float)d_channel[offset + 1];
    sum -= (float)d_channel[offset + numColsSource] + (float)d_channel[offset - numColsSource];
  }
  d_g[offset] = sum;
}

__global__ void computeIteration(unsigned char *d_dstImg,
                                 unsigned char *d_strictInteriorPixels,
                                 unsigned char *d_borderPixels,
                                 int numRowsSource, int numColsSource,
                                 float *f,
                                 float *g,
                                 float *f_next)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int offset = r * numColsSource + c;

  if (c >= numColsSource || r >= numRowsSource) return;

  float blendedSum = 0.f;
  float borderSum  = 0.f;

  if (d_strictInteriorPixels[offset - 1]) {
    blendedSum += f[offset - 1];
  }
  else {
    borderSum += d_dstImg[offset - 1];
  }

  if (d_strictInteriorPixels[offset + 1]) {
    blendedSum += f[offset + 1];
  }
  else {
    borderSum += d_dstImg[offset + 1];
  }

  if (d_strictInteriorPixels[offset - numColsSource]) {
    blendedSum += f[offset - numColsSource];
  }
  else {
    borderSum += d_dstImg[offset - numColsSource];
  }

  if (d_strictInteriorPixels[offset + numColsSource]) {
    blendedSum += f[offset + numColsSource];
  }
  else {
    borderSum += d_dstImg[offset + numColsSource];
  }

  float f_next_val = (blendedSum + borderSum + g[offset]) / 4.f;

  f_next[offset] = min(255.f, max(0.f, f_next_val)); //clip to [0, 255]

}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_dstImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement

     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */

  size_t srcSize = numRowsSource * numColsSource;
  thrust::device_vector<uchar4> d_sourceImg(h_sourceImg, h_sourceImg + srcSize);
  thrust::device_vector<uchar4> d_dstImg(h_dstImg, h_dstImg + srcSize);

  // mask
  thrust::device_vector<unsigned char> d_mask(srcSize);
  thrust::transform(d_sourceImg.begin(), d_sourceImg.end(), d_mask.begin(), createMask());
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  // border
  thrust::device_vector<unsigned char> d_borderPixels(srcSize);
  thrust::device_vector<unsigned char> d_strictInteriorPixels(srcSize);

  const dim3 gridSize(numColsSource, numRowsSource, 1);
  compute_border<<<gridSize, 1>>>(thrust::raw_pointer_cast(d_borderPixels.data()),
                                  thrust::raw_pointer_cast(d_strictInteriorPixels.data()),
                                  thrust::raw_pointer_cast(d_mask.data()),
                                  numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  thrust::device_vector<unsigned char> d_red;
  thrust::device_vector<unsigned char> d_blue;
  thrust::device_vector<unsigned char> d_green;
  d_red.  resize(srcSize);
  d_blue. resize(srcSize);
  d_green.resize(srcSize);

  //split the image
  thrust::transform(d_sourceImg.begin(), d_sourceImg.end(), thrust::make_zip_iterator(
                                                  thrust::make_tuple(d_red.begin(),
                                                                     d_blue.begin(),
                                                                     d_green.begin())),
                                                splitChannels());
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  thrust::device_vector<unsigned char> d_red_dst(srcSize);
  thrust::device_vector<unsigned char> d_blue_dst(srcSize);
  thrust::device_vector<unsigned char> d_green_dst(srcSize);
  thrust::transform(d_dstImg.begin(), d_dstImg.end(), thrust::make_zip_iterator(
                                                  thrust::make_tuple(d_red_dst.begin(),
                                                                     d_blue_dst.begin(),
                                                                     d_green_dst.begin())),
                                                splitChannels());
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // copy channels
  const size_t numIterations = 800;
  thrust::device_vector<float> d_red1(srcSize);
  thrust::device_vector<float> d_blue1(srcSize);
  thrust::device_vector<float> d_green1(srcSize);


  thrust::transform(d_red.begin(), d_red.end(), d_red1.begin(), charToFloat());
  thrust::transform(d_green.begin(), d_green.end(), d_green1.begin(), charToFloat());
  thrust::transform(d_blue.begin(), d_blue.end(), d_blue1.begin(), charToFloat());

  thrust::device_vector<float> d_red2(d_red1);
  thrust::device_vector<float> d_blue2(d_blue1);
  thrust::device_vector<float> d_green2(d_green1);
  thrust::device_vector<float> * ptr1, *ptr2, *temp;


  thrust::device_vector<float> d_g_red(srcSize);
  thrust::device_vector<float> d_g_green(srcSize);
  thrust::device_vector<float> d_g_blue(srcSize);
  computeG<<<gridSize, 1>>>(thrust::raw_pointer_cast(d_red.data()),
                            thrust::raw_pointer_cast(d_g_red.data()),
                            numColsSource, numRowsSource,
                            thrust::raw_pointer_cast(d_strictInteriorPixels.data()));
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  computeG<<<gridSize, 1>>>(thrust::raw_pointer_cast(d_green.data()),
                            thrust::raw_pointer_cast(d_g_green.data()),
                            numColsSource, numRowsSource,
                            thrust::raw_pointer_cast(d_strictInteriorPixels.data()));
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  computeG<<<gridSize, 1>>>(thrust::raw_pointer_cast(d_blue.data()),
                            thrust::raw_pointer_cast(d_g_blue.data()),
                            numColsSource, numRowsSource,
                            thrust::raw_pointer_cast(d_strictInteriorPixels.data()));
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  ptr1 = &d_red1;
  ptr2 = &d_red2;
  for (size_t i = 0; i < numIterations; ++i) {
    computeIteration<<<gridSize, 1>>>(thrust::raw_pointer_cast(d_red_dst.data()),
                                      thrust::raw_pointer_cast(d_strictInteriorPixels.data()),
                                      thrust::raw_pointer_cast(d_borderPixels.data()),
                                      numRowsSource, numColsSource,
                                      thrust::raw_pointer_cast(ptr1->data()),
                                      thrust::raw_pointer_cast(d_g_red.data()),
                                      thrust::raw_pointer_cast(ptr2->data()));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    temp = ptr2;
    ptr2 = ptr1;
    ptr1 = temp;
  }

  ptr1 = &d_green1;
  ptr2 = &d_green2;
  for (size_t i = 0; i < numIterations; ++i) {
    computeIteration<<<gridSize, 1>>>(thrust::raw_pointer_cast(d_green_dst.data()),
                                      thrust::raw_pointer_cast(d_strictInteriorPixels.data()),
                                      thrust::raw_pointer_cast(d_borderPixels.data()),
                                      numRowsSource, numColsSource,
                                      thrust::raw_pointer_cast(ptr1->data()),
                                      thrust::raw_pointer_cast(d_g_green.data()),
                                      thrust::raw_pointer_cast(ptr2->data()));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    temp = ptr2;
    ptr2 = ptr1;
    ptr1 = temp;
  }

  ptr1 = &d_blue1;
  ptr2 = &d_blue2;
  for (size_t i = 0; i < numIterations; ++i) {
    computeIteration<<<gridSize, 1>>>(thrust::raw_pointer_cast(d_blue_dst.data()),
                                      thrust::raw_pointer_cast(d_strictInteriorPixels.data()),
                                      thrust::raw_pointer_cast(d_borderPixels.data()),
                                      numRowsSource, numColsSource,
                                      thrust::raw_pointer_cast(ptr1->data()),
                                      thrust::raw_pointer_cast(d_g_blue.data()),
                                      thrust::raw_pointer_cast(ptr2->data()));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    temp = ptr2;
    ptr2 = ptr1;
    ptr1 = temp;
  }

  // merge
  thrust::device_vector<uchar4> d_outputImg(srcSize);

  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                          d_red1.begin(),
                          d_blue1.begin(),
                          d_green1.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                          d_red1.end(),
                          d_blue1.end(),
                          d_green1.end())),
                    d_outputImg.begin(),
                    combineChannels());
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // merge with origin dst
  thrust::device_vector<uchar4> d_blendedImg(srcSize);
  thrust::transform(
                    thrust::make_zip_iterator(thrust::make_tuple(d_outputImg.begin(), d_strictInteriorPixels.begin(), d_dstImg.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(d_outputImg.end(), d_strictInteriorPixels.end(), d_dstImg.end())),
    d_blendedImg.begin(),
    blendInterior());

  checkCudaErrors(cudaMemcpy(h_blendedImg, thrust::raw_pointer_cast(d_blendedImg.data()), sizeof(uchar4)*srcSize, cudaMemcpyDeviceToHost));

}
