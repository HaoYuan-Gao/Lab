#pragma once

#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublasLt.h>

#include "inline_check.h"

struct CudnnHandleTraits {
    using Handle = cudnnHandle_t;

    static const char* env_name() {
        return "GTENSOR_CUDNN_HANDLES_PER_DEVICE";
    }

    static Handle create() {
        Handle handle = nullptr;
        CHECK_CUDNN(cudnnCreate(&handle));
        return handle;
    }

    static void destroy(Handle handle) {
        CHECK_CUDNN(cudnnDestroy(handle));
    }
};

struct CublasLtHandleTraits {
    using Handle = cublasLtHandle_t;

    static const char* env_name() {
        return "GTENSOR_CUBLASLT_HANDLES_PER_DEVICE";
    }

    static Handle create() {
        Handle handle = nullptr;
        CHECK_CUBLASLT(cublasLtCreate(&handle));
        return handle;
    }

    static void destroy(Handle handle) {
        CHECK_CUBLASLT(cublasLtDestroy(handle));
    }
};

template <typename Traits>
class GpuHandlePool {
public:
    using Handle = typename Traits::Handle;

    class Lease {
    public:
        Lease() = default;

        Lease(GpuHandlePool* pool, int device_id, Handle handle)
            : pool_(pool), device_id_(device_id), handle_(handle) {}

        ~Lease() {
            release();
        }

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        Lease(Lease&& other) noexcept
            : pool_(other.pool_),
              device_id_(other.device_id_),
              handle_(other.handle_) {
            other.pool_ = nullptr;
            other.handle_ = nullptr;
        }

        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                release();

                pool_ = other.pool_;
                device_id_ = other.device_id_;
                handle_ = other.handle_;

                other.pool_ = nullptr;
                other.handle_ = nullptr;
            }
            return *this;
        }

        Handle get() const {
            return handle_;
        }

        int device_id() const {
            return device_id_;
        }

        explicit operator bool() const {
            return handle_ != nullptr;
        }

    private:
        void release() {
            if (pool_ != nullptr && handle_ != nullptr) {
                pool_->release(device_id_, handle_);
                pool_ = nullptr;
                handle_ = nullptr;
            }
        }

        GpuHandlePool* pool_ = nullptr;
        int device_id_ = -1;
        Handle handle_ = nullptr;
    };

    GpuHandlePool(const GpuHandlePool&) = delete;
    GpuHandlePool& operator=(const GpuHandlePool&) = delete;

    static GpuHandlePool& instance() {
        static GpuHandlePool pool;
        return pool;
    }

    Lease acquire(int device_id) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto& handles = handles_[device_id];

        while (true) {
            for (auto& state : handles) {
                if (!state.in_use) {
                    state.in_use = true;
                    lock.unlock();

                    CHECK_CUDA(cudaSetDevice(device_id));
                    return Lease(this, device_id, state.handle);
                }
            }

            if (handles.size() < max_handles_per_device_) {
                CHECK_CUDA(cudaSetDevice(device_id));

                // Keep the lock while creating the handle so the upper bound is strict.
                // Handle creation is rare, so this is simpler and safer.
                Handle handle = Traits::create();

                handles.push_back(HandleState{handle, true});
                return Lease(this, device_id, handle);
            }

            cv_.wait(lock);
        }
    }

    Lease acquire_specific(int device_id, Handle target) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto& handles = handles_[device_id];

        while (true) {
            for (auto& state : handles) {
                if (state.handle == target) {
                    if (!state.in_use) {
                        state.in_use = true;
                        lock.unlock();

                        CHECK_CUDA(cudaSetDevice(device_id));
                        return Lease(this, device_id, state.handle);
                    }

                    break;
                }
            }

            cv_.wait(lock);
        }
    }

    std::size_t max_handles_per_device() const {
        return max_handles_per_device_;
    }

private:
    GpuHandlePool() {
        if (const char* env = std::getenv(Traits::env_name())) {
            const int value = std::atoi(env);
            if (value > 0) {
                max_handles_per_device_ = static_cast<std::size_t>(value);
            }
        }
    }

    ~GpuHandlePool() {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto& kv : handles_) {
            const int device_id = kv.first;
            cudaSetDevice(device_id);

            for (auto& state : kv.second) {
                if (state.handle != nullptr) {
                    Traits::destroy(state.handle);
                    state.handle = nullptr;
                }
            }
        }
    }

    struct HandleState {
        Handle handle = nullptr;
        bool in_use = false;
    };

    void release(int device_id, Handle handle) {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            auto it = handles_.find(device_id);
            if (it == handles_.end()) {
                return;
            }

            for (auto& state : it->second) {
                if (state.handle == handle) {
                    state.in_use = false;
                    break;
                }
            }
        }

        cv_.notify_one();
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<int, std::deque<HandleState>> handles_;
    std::size_t max_handles_per_device_ = 4;
};

using CudnnHandlePool = GpuHandlePool<CudnnHandleTraits>;
using CublasLtHandlePool = GpuHandlePool<CublasLtHandleTraits>;
