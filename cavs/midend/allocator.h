#ifndef CAVS_MIDEND_ALLOCATOR_H_
#define CAVS_MIDEND_ALLOCATOR_H_

#include "cavs/proto/devices.pb.h"
#include "cavs/proto/op_def.pb.h"
#include "cavs/util/macros.h"

#include <string>
#include <unordered_map>
#include <iostream>
#include <atomic>

namespace midend {

  // #define CORTEX_MEM_PROF

class Allocator {
public:
  static bool mem_prof_on;
#ifdef CORTEX_MEM_PROF
  static std::atomic<long> current_mem_usage;
  static std::atomic<long> max_mem_usage;
  static std::unordered_map<void*, long> buf_size_map;
#endif

 public:
  Allocator(const std::string& name, DeviceType type) :
    name_(name), type_(type) {
#ifdef CORTEX_MEM_PROF
    std::cerr << "MEMORY PROFILING ON!!" << std::endl;
#endif

  }
  FORCE_INLINE const std::string& name() const { return name_; }
  FORCE_INLINE DeviceType type() const { return type_; }

  virtual void* AllocateRaw(size_t nbytes) = 0;
  virtual void DeallocateRaw(void* buf) = 0;
  virtual void InitWithZero(void* buf, size_t nbytes) = 0;

  template <typename T>
  T* Allocate(size_t n_elements) {
    void *p = AllocateRaw(n_elements*sizeof(T));
    return reinterpret_cast<T*>(p);
  }
  template <typename T>
  void Deallocate(T* buf) {
    if (buf) {
      DeallocateRaw(buf);
    }
  }

 private:
  std::string name_;
  DeviceType type_;

 protected:
  Allocator() {}
};

inline float get_max_mem_usage() {
#ifdef CORTEX_MEM_PROF
  long nbytes = Allocator::max_mem_usage.load();
  float kbytes = ((float) nbytes) / 1024.0;
  return kbytes;
#endif
  return -10000000;
}

inline void set_mem_prof(bool value) {
  Allocator::mem_prof_on = value;
}

class TrackingAllocator : public Allocator {
 public:
  explicit TrackingAllocator(Allocator* allocator);
  FORCE_INLINE const std::string& name() const { return allocator_->name(); }
  FORCE_INLINE size_t capacity() const { return capacity_; }
  void* AllocateRaw(size_t nbytes) override;
  void DeallocateRaw(void* buf) override;
  FORCE_INLINE void InitWithZero(void* buf, size_t nbytes) override {
    allocator_->InitWithZero(buf, nbytes);
  }

 private:
  Allocator* allocator_;
  size_t capacity_;
  std::unordered_map<void*, size_t> trace_;
};

Allocator* GetAllocator(const OpDef& def);
Allocator* GetAllocator(const std::string& device);

#define REGISTER_STATIC_ALLOCATOR(key, alloc)                  \
    REGISTER_STATIC_ALLOCATOR_UNIQ(__COUNTER__, key, alloc)
#define REGISTER_STATIC_ALLOCATOR_UNIQ(ctr, key, alloc)        \
    REGISTER_STATIC_ALLOCATOR_CONCAT(ctr, key, alloc)
#define REGISTER_STATIC_ALLOCATOR_CONCAT(ctr, key, alloc)      \
    static allocator_factory::AllocatorRegister                \
        register_body_##ctr##_allocator(key, alloc)

namespace allocator_factory {

class AllocatorRegister {
 public:
  AllocatorRegister(const std::string& name, Allocator* alloc) {
    InitInternal(name, alloc);
  }
 private:
  void InitInternal(const std::string& name, Allocator* alloc);
};

} //namespace allocator_factory

} //namepsace midend

#endif
