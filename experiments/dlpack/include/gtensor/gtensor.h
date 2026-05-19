#pragma once

#include <dlpack/dlpack.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gtensor/memory_pool.h"

namespace gtensor {

struct Device {
  DLDeviceType type{kDLCPU};
  int32_t id{0};
};

std::size_t dtype_nbytes(DLDataType dtype);
std::size_t compute_numel(const std::vector<int64_t>& shape);
std::vector<int64_t> compact_strides(const std::vector<int64_t>& shape);

struct Storage {
  void* data{nullptr};
  std::size_t bytes{0};
  Device device{};
  std::shared_ptr<IAllocator> allocator{};
  DLManagedTensor* borrowed_dlmt{nullptr};

  ~Storage();
  Storage() = default;
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
};

class GTensor {
 public:
  GTensor() = default;
  GTensor(std::vector<int64_t> shape, DLDataType dtype, Device device = Device{});

  static GTensor from_dlpack(DLManagedTensor* managed);
  DLManagedTensor* to_dlpack() const;

  void* data() const { return storage_ ? storage_->data : nullptr; }
  const std::vector<int64_t>& shape() const { return shape_; }
  const std::vector<int64_t>& strides() const { return strides_; }
  DLDataType dtype() const { return dtype_; }
  Device device() const { return device_; }
  int ndim() const { return static_cast<int>(shape_.size()); }
  std::size_t numel() const { return compute_numel(shape_); }
  std::size_t nbytes() const { return numel() * dtype_nbytes(dtype_); }
  bool is_contiguous() const;
  std::string repr() const;

 private:
  std::shared_ptr<Storage> storage_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
  DLDataType dtype_{kDLFloat, 32, 1};
  Device device_{};
  std::uint64_t byte_offset_{0};
};

}  // namespace gtensor
