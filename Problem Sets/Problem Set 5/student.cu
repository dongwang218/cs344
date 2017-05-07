/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numBins, int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
}

// https://www.ecse.rpi.edu/~wrf/wiki/ParallelComputingSpring2014/thrust/histogram.cu
void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel

  //if you want to use/launch more than one kernel,
  //feel free
  const thrust::device_ptr<unsigned int> data = thrust::device_pointer_cast(const_cast<unsigned int*>(d_vals));
  thrust::device_ptr<unsigned int> histogram(d_histo);
  thrust::sort(data, data+numElems);
  thrust::counting_iterator<unsigned int> search_begin(0);
  thrust::upper_bound(data, data+numElems,
                      search_begin, search_begin + numBins,
                      histogram);
  thrust::adjacent_difference(histogram, histogram+numBins,
                              histogram);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
