#include "cavs/midend/allocator.h"
//#include "cavs/midend/devices.h"
#include "cavs/util/logging.h"
#include "cavs/util/op_util.h"

#include <atomic>

using std::string;

namespace midend {

bool Allocator::mem_prof_on(false);
#ifdef CORTEX_MEM_PROF
std::atomic<long> Allocator::current_mem_usage{0};
std::atomic<long> Allocator::max_mem_usage{0};
std::unordered_map<void*, long> Allocator::buf_size_map;
#endif

class CPUAllocator : public Allocator {
 public:
  CPUAllocator()
      : Allocator(DeviceTypeToString(CPU), CPU) {}
  void* AllocateRaw(size_t nbytes) override {
    void* ptr = malloc(nbytes);
// #ifdef CORTEX_MEM_PROF
//     if (Allocator::mem_prof_on) {
//       Allocator::current_mem_usage += nbytes;
//       if (Allocator::current_mem_usage > Allocator::max_mem_usage)
// 	Allocator::max_mem_usage.exchange(Allocator::current_mem_usage);
//       Allocator::buf_size_map[ptr] = nbytes;
//       std::cout << "[CALLOC] " << nbytes << " " << Allocator::current_mem_usage << " " << Allocator::max_mem_usage << std::endl;
//     }
// #endif
    return ptr;
  }
  void DeallocateRaw(void* buf) override {
// #ifdef CORTEX_MEM_PROF
//     if (Allocator::mem_prof_on && Allocator::buf_size_map.count(buf)) {
//       Allocator::current_mem_usage -= Allocator::buf_size_map[buf];
//       std::cout << "[CFREE] " << Allocator::buf_size_map[buf] << " " <<
// 	Allocator::current_mem_usage << " " << Allocator::max_mem_usage << std::endl;
//       Allocator::buf_size_map.erase(buf);
//     }
// #endif
    free(buf);
  }
  void InitWithZero(void* buf, size_t nbytes) override {
    memset(buf, 0, nbytes);
  }
};

Allocator* cpu_allocator() {
  static CPUAllocator cpu_alloc;
  return &cpu_alloc;
}

REGISTER_STATIC_ALLOCATOR(DeviceTypeToString(CPU), cpu_allocator());

TrackingAllocator::TrackingAllocator(Allocator* allocator)
    : allocator_(allocator), capacity_(0) {}

void* TrackingAllocator::AllocateRaw(size_t nbytes) {
  void* ptr = allocator_->AllocateRaw(nbytes);
  capacity_ += nbytes;
  trace_[ptr] = nbytes;
  return ptr;
}

void TrackingAllocator::DeallocateRaw(void *buf) {
  CHECK(trace_.find(buf) != trace_.end());
  capacity_ -= trace_[buf];
  trace_.erase(buf);
}

namespace allocator_factory {

typedef std::unordered_map<string, Allocator*> AllocatorRegistry;
static AllocatorRegistry* GlobalAllocatorRegistry() {
  static AllocatorRegistry* global_allocator_registry = new AllocatorRegistry();
  return global_allocator_registry;
}
void AllocatorRegister::InitInternal(const string& name, Allocator* alloc) {
  GlobalAllocatorRegistry()->insert(std::make_pair(name, alloc));
}

} //namespace allocator_factory

Allocator* GetAllocator(const OpDef& def) {
  DeviceType dev = def.device();
  string dev_name;
  if (dev == GPU)
    dev_name = "GPU";
  else
    dev_name = "CPU";
  return GetAllocator(dev_name);
}

Allocator* GetAllocator(const string& dev) {
  if (allocator_factory::GlobalAllocatorRegistry()->count(dev) == 0)
    return NULL;
  else
    return allocator_factory::GlobalAllocatorRegistry()->at(dev);
}

} //namespace midend
