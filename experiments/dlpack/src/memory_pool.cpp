#include "gtensor/memory_pool.h"

#include <algorithm>
#include <cstdlib>
#include <new>
#include <stdexcept>

namespace gtensor {
namespace {
std::size_t round_up(std::size_t n, std::size_t alignment) {
  return ((n + alignment - 1) / alignment) * alignment;
}
}

void* CpuMemoryPool::allocate(std::size_t bytes, std::size_t alignment) {
  if (bytes == 0) return nullptr;
  const std::size_t rounded = round_up(bytes, alignment);
  std::lock_guard<std::mutex> lock(mu_);

  auto& list = free_lists_[rounded];
  if (!list.empty()) {
    Block b = list.back();
    list.pop_back();
    stats_.cached_blocks--;
    stats_.cache_bytes -= b.bytes;
    stats_.current_bytes += b.bytes;
    stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.current_bytes);
    stats_.live_blocks++;
    live_blocks_[b.ptr] = b.bytes;
    return b.ptr;
  }

  void* ptr = nullptr;
#if defined(_MSC_VER)
  ptr = _aligned_malloc(rounded, alignment);
  if (!ptr) throw std::bad_alloc();
#else
  if (posix_memalign(&ptr, alignment, rounded) != 0) throw std::bad_alloc();
#endif
  stats_.current_bytes += rounded;
  stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.current_bytes);
  stats_.total_allocated_bytes += rounded;
  stats_.live_blocks++;
  live_blocks_[ptr] = rounded;
  return ptr;
}

void CpuMemoryPool::deallocate(void* ptr, std::size_t bytes, std::size_t alignment) {
  if (!ptr) return;
  const std::size_t rounded = round_up(bytes, alignment);
  std::lock_guard<std::mutex> lock(mu_);

  auto it = live_blocks_.find(ptr);
  const std::size_t actual = (it == live_blocks_.end()) ? rounded : it->second;
  if (it != live_blocks_.end()) live_blocks_.erase(it);

  stats_.current_bytes -= actual;
  stats_.total_freed_bytes += actual;
  stats_.live_blocks--;
  stats_.cached_blocks++;
  stats_.cache_bytes += actual;
  free_lists_[actual].push_back(Block{ptr, actual, alignment});
}

MemoryPoolStats CpuMemoryPool::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  return stats_;
}

void CpuMemoryPool::reset_stats() {
  std::lock_guard<std::mutex> lock(mu_);
  stats_.peak_bytes = stats_.current_bytes;
  stats_.total_allocated_bytes = 0;
  stats_.total_freed_bytes = 0;
}

void CpuMemoryPool::clear_cache() {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& kv : free_lists_) {
    for (auto& b : kv.second) {
#if defined(_MSC_VER)
      _aligned_free(b.ptr);
#else
      free(b.ptr);
#endif
    }
  }
  free_lists_.clear();
  stats_.cached_blocks = 0;
  stats_.cache_bytes = 0;
}

std::shared_ptr<IAllocator> default_cpu_allocator() {
  static auto pool = std::make_shared<CpuMemoryPool>();
  return pool;
}

}  // namespace gtensor
