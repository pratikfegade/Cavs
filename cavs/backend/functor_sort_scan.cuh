#ifndef CAVS_BACKEND_FUNCTOR_SORT_SCAN_CUH_
#define CAVS_BACKEND_FUNCTOR_SORT_SCAN_CUH_

#include "cavs/util/macros_gpu.h"
#include "cavs/util/logging.h"
#include <stdio.h>

namespace backend {

template <typename T>
__device__ inline void Comparator(T& valA, T& valB, bool direction) {
  if ((valA > valB) == direction) {
    T tmp;
    tmp = valA;
    valA = valB;
    valB = tmp;
  }
}

//SHARE_SIZE_LIMIT should be equal to 2*blockDim.x
//In this case, N is less or equal to SHARE_SIZE_LIMIT
/*template <typename T, unsigned int SHARE_SIZE_LIMIT>*/
template <typename T>
__global__ void BatchedOddEvenSortInCache(T* out, const T* in,
    bool direction, unsigned int N) {
  extern __shared__ T s_val[];
  const T* in_val = in + blockIdx.x*N+ threadIdx.x;
  T* out_val = out + blockIdx.x*N+ threadIdx.x;

  for (int round = 0;
       round < (N+2*blockDim.x-1)/(2*blockDim.x); round++) {
    int base_idx = round*2*blockDim.x;
    if (threadIdx.x + base_idx < N) {
      s_val[threadIdx.x] = in_val[base_idx];
    }
    if (threadIdx.x + base_idx + blockDim.x < N) {
      s_val[threadIdx.x+blockDim.x] = in_val[base_idx+blockDim.x];
    }
    __syncthreads();
    for (unsigned outer_stride = 1; outer_stride <= blockDim.x;
         outer_stride <<= 1) {
      unsigned stride = outer_stride;
      unsigned offset = threadIdx.x & (stride -1);
      {
        unsigned int pos = 2*threadIdx.x - offset;
        unsigned int global_pos = pos + base_idx;
        if (global_pos+stride < N) {
          Comparator(s_val[pos], s_val[pos+stride], direction);
        }
        stride >>= 1;
      }
      __syncthreads();
      for (; stride > 0; stride >>= 1) {
        unsigned pos = 2*threadIdx.x - (threadIdx.x&(stride-1));
        unsigned int global_pos = pos + base_idx;
        if (offset >= stride && global_pos < N) {
          Comparator(s_val[pos-stride], s_val[pos], direction);
        }
        __syncthreads();
      }
    }
    if (threadIdx.x + base_idx < N) {
      out_val[base_idx] = s_val[threadIdx.x];
    }
    if (threadIdx.x + base_idx+blockDim.x < N) {
      out_val[base_idx+blockDim.x] = s_val[threadIdx.x+blockDim.x];
    }
    __syncthreads();
  }
  /*if (threadIdx.x == 0) {*/
    /*for (int i = 0; i < N; i++)*/
    /*printf("1st Procedure: %d\t [%d] : %d\n", __LINE__, i, in_val[i]); */
  /*}*/
}

//Assume each strip has been sorted with BatchedOddEvenSortInCache
//This function STRIDED sort this blocks
//Instead of merge sort, we still use odd-even sort,
//but all the data are in the global memory now
//The thread configuration is exactly the same with the aforementioned function
template <typename T>
__global__ void BatchedOddEvenSortStride(T* inout,
    bool direction, unsigned int N) {
  const int SORTED_BLOCK = 2*blockDim.x;
  T* inout_slice = inout + blockIdx.x*N;

  //SORTED_BLOCK is 2x larger than any threadIdx.x
  for (unsigned int outer_stride = SORTED_BLOCK; outer_stride < N;
       outer_stride <<= 1) {
    /*printf("%d\touter_stride: %d (id:%d, N:%d, blockDim:%d)\n",*/
           /*__LINE__, outer_stride, threadIdx.x, N, blockDim.x);*/
    for (unsigned int strip = 0;
         strip < (N+2*outer_stride-1) / (2*outer_stride); strip++) {
      unsigned int offset_of_strip = strip*2*outer_stride;
      for (unsigned int stride = outer_stride; stride > 0; stride >>= 1) {
        if (stride < outer_stride) {
          for (unsigned int round = 0; round < outer_stride/blockDim.x; round++) {
            //We don't have enough threads here,
            //so each thread has to do more and
            //pretend to behave as the virtual thread id(s);
            unsigned int virtual_id = threadIdx.x + round*blockDim.x;
            unsigned int offset_within_stride = virtual_id & (outer_stride-1);
            unsigned int pos = 2*virtual_id - (virtual_id & (stride-1)) + offset_of_strip;
            /*printf("%d\t vid: %d, ows:%d, pos:%d (id:%d, stride:%d, round:%d, strip:%d)\n",*/
                   /*__LINE__, virtual_id, offset_within_stride, pos, threadIdx.x, stride, round, strip);*/
            if (offset_within_stride >= stride && pos < N) {
              Comparator(inout_slice[pos-stride], inout_slice[pos], direction);
              /*printf("%d\tCompare %d([%d]) with %d([%d]) (id:%d, stride:%d, round:%d, strip:%d)\n",*/
                     /*__LINE__, inout[pos-stride], pos-stride, */
                     /*inout[pos], pos, threadIdx.x, stride, round, strip);*/
            }
            __syncthreads();
          }
        }else {
          for (unsigned int round = 0; round < outer_stride/blockDim.x; round++) {
            unsigned int virtual_id = threadIdx.x + round*blockDim.x;
            unsigned int offset_within_stride = virtual_id & (outer_stride-1);
            unsigned int pos = 2*virtual_id - offset_within_stride + offset_of_strip;
            if (pos+stride < N) {
              Comparator(inout_slice[pos], inout_slice[pos+stride], direction);
              /*printf("%d\tCompare %d([%d]) with %d([%d]) (id:%d, stride:%d, round:%d, strip:%d)\n",*/
                     /*__LINE__, inout[pos], pos, */
                     /*inout[pos+stride], pos+stride, threadIdx.x, stride, round, strip);*/
            }
            __syncthreads();
          }
        }
      }
    }
  }
}

template <typename T>
void BatchedOddEvenSort(T* out, const T* in,
    bool direction, unsigned int N, unsigned int Batch) {
  const int MAX_THREADS_IN_BLOCK = 1 << 10;
  unsigned int threadsPerBlock = 1;
  while (threadsPerBlock < N) {
    threadsPerBlock <<= 1;
    if (threadsPerBlock == MAX_THREADS_IN_BLOCK)
      break;
  }
  if (threadsPerBlock > N)
    threadsPerBlock = (threadsPerBlock >> 1) > 0 ? (threadsPerBlock >> 1) : 1;
  unsigned int blocksPerGrid = Batch;
  /*LOG(INFO) << blocksPerGrid << "\t" << threadsPerBlock;*/

  BatchedOddEvenSortInCache<T><<<blocksPerGrid, threadsPerBlock,
                       2*threadsPerBlock*sizeof(T)>>>(out, in, direction, N);
  checkCudaError(cudaDeviceSynchronize());
  checkCudaError(cudaGetLastError());
  BatchedOddEvenSortStride<T><<<blocksPerGrid, threadsPerBlock>>>(
                                                    out, direction, N);
  checkCudaError(cudaDeviceSynchronize());
  checkCudaError(cudaGetLastError());
}

//N == blockDim.x
//N < 1024 (warp_id < 32)
//There must be less than 32 warps in one block,
//as required in the syntax of CUDA)
//SHARE_SIZE_LIMIT should be equal to blockDim.x
template <typename T>
__global__ void BatchedScanInCache(T* out, const T* in, unsigned int N) {
  extern __shared__ T s_val[];
  const int warpSize = 1 << 5;
  int lane_id = threadIdx.x & (warpSize-1);
  int warp_id = threadIdx.x >> 5;
  for (int round = 0;
      round < (N+blockDim.x-1)/blockDim.x; round++) {
    int block_id = threadIdx.x + round*blockDim.x;
    int global_id = block_id + blockIdx.x*N;
    if (block_id < N) {
      T val = in[global_id];
      #pragma unroll
      for (int i = 1; i < warpSize; i <<= 1) {
        // T pre_sum = __shfl_up(val, i, warpSize);
        T pre_sum = __shfl_up_sync(0xffffffff, val, i, warpSize);
        if (lane_id >= i) val += pre_sum;
      }
      s_val[threadIdx.x] = val;
    }
    __syncthreads();

    for (int i = 1; i <= (blockDim.x >> 5); i <<= 1) {
      T pre_sum = 0;
      if (warp_id >= i) {
        pre_sum = s_val[((warp_id-i+1) << 5)-1];
      }
      __syncthreads();
      if (warp_id >= i) {
        s_val[threadIdx.x] += pre_sum;
      }
      __syncthreads();
    }

    if (block_id < N) {
      out[global_id] = s_val[threadIdx.x];
    }
    __syncthreads();
  }
}

//Assume each stride has been scanned with BatchedScanInCache
//This function STRIDED scans this blocks
template <typename T>
__global__ void BatchedScanStride(T* out, const T* in, unsigned int N) {
  __shared__ T pre_scan;
  for (int round = 1;
      round < (N+blockDim.x-1)/blockDim.x; round++) {
    int block_id = threadIdx.x + round*blockDim.x;
    int global_id = block_id + blockIdx.x*N;
    if (threadIdx.x == 0) {
      pre_scan = in[global_id-1];
    }
    __syncthreads();
    if (block_id < N) {
      out[global_id] += pre_scan;
    }
    __syncthreads();
  }
}

template <typename T>
void BatchedScan(T* out, const T* in, unsigned int N, unsigned int Batch) {
  const int MAX_THREADS_IN_BLOCK = 1 << 10;
  unsigned int threadsPerBlock =
      (MAX_THREADS_IN_BLOCK > N)? N : MAX_THREADS_IN_BLOCK;
  unsigned int blocksPerGrid = Batch;
  /*LOG(INFO) << blocksPerGrid << "\t" << threadsPerBlock;*/

  BatchedScanInCache<T><<<blocksPerGrid, threadsPerBlock,
                       threadsPerBlock*sizeof(T)>>>(out, in, N);
  BatchedScanStride<T><<<blocksPerGrid, threadsPerBlock>>>(out, in, N);
  checkCudaError(cudaGetLastError());
}

} //namespace backend

#endif
