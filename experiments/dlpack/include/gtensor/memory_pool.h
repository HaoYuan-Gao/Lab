#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace gtensor {

struct MemoryPoolStats {
  std::size_t current_bytes{0};
  std::size_t peak_bytes{0};
  std::size_t total_allocated_bytes{0};
  std::size_t total_freed_bytes{0};
  std::size_t live_blocks{0};
  std::size_t cached_blocks{0};
  std::size_t cache_bytes{0};
};

class IAllocator {
 public:
  virtual ~IAllocator() = default;
  virtual void* allocate(std::size_t bytes, std::size_t alignment = 256) = 0;
  virtual void deallocate(void* ptr, std::size_t bytes, std::size_t alignment = 256) = 0;
  virtual MemoryPoolStats stats() const = 0;
  virtual void reset_stats() = 0;
};

class CpuMemoryPool final : public IAllocator {
 public:
  void* allocate(std::size_t bytes, std::size_t alignment = 256) override;
  void deallocate(void* ptr, std::size_t bytes, std::size_t alignment = 256) override;
  MemoryPoolStats stats() const override;
  void reset_stats() override;
  void clear_cache();

 private:
  struct Block { void* ptr; std::size_t bytes; std::size_t alignment; };
  mutable std::mutex mu_;
  std::unordered_map<std::size_t, std::vector<Block>> free_lists_;
  std::unordered_map<void*, std::size_t> live_blocks_;
  MemoryPoolStats stats_;
};

std::shared_ptr<IAllocator> default_cpu_allocator();

}  // namespace gtensor
