#include "cavs/midend/allocator.h"
/*#include "cavs/midend/devices.h"*/
#include "cavs/util/macros_gpu.h"
#include "cavs/util/op_util.h"

namespace midend {

class GPUAllocator : public Allocator {
 public:
  GPUAllocator()
      : Allocator(DeviceTypeToString(GPU), GPU) {}
  void* AllocateRaw(size_t nbytes) override {
    VLOG(V_DEBUG) << "allocating " << nbytes << " bytes";
    void* ptr = NULL;
    checkCudaError(cudaMalloc(&ptr, nbytes));
    checkCudaError(cudaMemset(ptr, 0, nbytes));
    CHECK_NOTNULL(ptr);
#ifdef CORTEX_MEM_PROF
    Allocator::current_mem_usage += nbytes;
    if (Allocator::current_mem_usage > Allocator::max_mem_usage)
      Allocator::max_mem_usage.exchange(Allocator::current_mem_usage);
    Allocator::buf_size_map[ptr] = nbytes;
#endif
    return ptr;
  }
  void DeallocateRaw(void* buf) override {
#ifdef CORTEX_MEM_PROF
    Allocator::current_mem_usage -= Allocator::buf_size_map[buf];
    Allocator::buf_size_map.erase(buf);
#endif
    checkCudaError(cudaFree(buf));
  }
  void InitWithZero(void* buf, size_t nbytes) override {
    checkCudaError(cudaMemsetAsync(buf, 0, nbytes, cudaStreamDefault));
  }
};

Allocator* gpu_allocator() {
  static GPUAllocator gpu_alloc;
  return &gpu_alloc;
}

REGISTER_STATIC_ALLOCATOR("GPU", gpu_allocator());

} //namespace midend
