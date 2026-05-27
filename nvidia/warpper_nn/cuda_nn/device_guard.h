#pragma once

#include <cuda_runtime.h>

class DeviceGuard final {
 public:
  explicit DeviceGuard(int device) noexcept {
    cudaGetDevice(&old_device_);
    if (old_device_ == device) {
      return;
    }

    cudaSetDevice(device);
    active_ = true;
  }

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
  DeviceGuard(DeviceGuard&&) = delete;
  DeviceGuard& operator=(DeviceGuard&&) = delete;

  ~DeviceGuard() noexcept {
    (void)restore();
  }

  cudaError_t restore() noexcept {
    if (!active_) {
      return cudaSuccess;
    }

    active_ = false;
    return cudaSetDevice(old_device_);
  }

 private:
  int old_device_{0};
  bool active_{false};
};
