#ifndef CAVS_BACKEND_FUNCTOR_SORT_SCAN_CUH_
#define CAVS_BACKEND_FUNCTOR_SORT_SCAN_CUH_

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

template <typename T, unsigned int SHARE_SIZE_LIMIT>
__global__ void BatchedMergeSort(T* out, const T* in, unsigned int N, bool direction) {
  __shared__ T s_val[SHARE_SIZE_LIMIT];
  const T* in_val = in + blockIdx.x*N+ threadIdx.x;
  T* out_val = out + blockIdx.x*N+ threadIdx.x;
  s_val[threadIdx.x] = in_val[0];
  if (threadIdx.x + blockDim.x < N)
    s_val[threadIdx.x+blockDim.x] = in_val[blockDim.x];

  /*for (unsigned size = 2; size <= N; size <<= 1) {*/
  for (unsigned outer_stride = 1; outer_stride < N; outer_stride <<= 1) {
    /*unsigned stride = size / 2; */
    unsigned stride = outer_stride;
    unsigned offset = threadIdx.x & (stride -1);
    __syncthreads();
    {
      unsigned pos = 2*threadIdx.x - offset; 
      if (pos+stride < N) {
        Comparator(s_val[pos], s_val[pos+stride], direction);
      }
      /*printf("%d\t%d\t%d\n", threadIdx.x, s_val[pos], s_val[pos+stride]);*/
      stride >>= 1;
    }
    for (; stride > 0; stride >>= 1) {
      __syncthreads(); 
      unsigned pos = 2*threadIdx.x - (threadIdx.x&(stride-1));
      if (offset >= stride && pos < N) {
        Comparator(s_val[pos-stride], s_val[pos], direction);
        /*printf("%d\t%d\t%d\n", threadIdx.x, s_val[pos-stride], s_val[pos]);*/
      }
    }
  }
  __syncthreads();
  out_val[0] = s_val[threadIdx.x];
  /*if (threadIdx.x + (N+1)/2 < N)*/
    /*d_val[(N+1)/2] = s_val[threadIdx.x+(N+1)/2];*/
  if (threadIdx.x + blockDim.x < N)
    out_val[blockDim.x] = s_val[threadIdx.x+blockDim.x];
}

/*template <typename T, unsigned int SHARE_SIZE_LIMIT>*/
/*__global__ void BatchedScan(T* inout, unsigned int N) {}*/
//N == blockDim.x
//N < 1024 (warp_id < 32)
//There must be less than 32 warps in one block,
//as required in the syntax of CUDA)
/*template <unsigned int SHARE_SIZE_LIMIT>*/
template <unsigned int SHARE_SIZE_LIMIT>
__global__ void BatchedScan(float* out, const float* in, unsigned int N) {
  __shared__ float s_val[SHARE_SIZE_LIMIT]; 
  int id = threadIdx.x + blockIdx.x*N;
  const int warpSize = 1 << 5;
  int lane_id = threadIdx.x & (warpSize-1);
  int warp_id = threadIdx.x >> 5;
  float val = in[id];
  #pragma unroll
  for (int i = 1; i < warpSize; i <<= 1) {
    float pre_sum = __shfl_up(val, i, warpSize);
    if (lane_id >= i) val += pre_sum;
  }
  s_val[threadIdx.x] = val;
  __syncthreads();
  
  /*printf("%d\t%d\t%f\n", threadIdx.x, lane_id, val);*/
  for (int i = 1; i <= (N >> 5); i <<= 1) {
    if (warp_id >= i) {
      float pre_sum = s_val[((warp_id-i+1) << 5)-1];
      __syncthreads();
      s_val[threadIdx.x] += pre_sum;
      __syncthreads();
    }  
  }
  out[id] = s_val[threadIdx.x];
}

} //namespace backend

#endif
