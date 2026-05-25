#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>
#include "cuda_nn/inline_check.h"

class DeviceWorkspacePool {
public:
    class Lease {
    public:
        Lease() = default;

        Lease(DeviceWorkspacePool* pool,
              int device_id,
              void* ptr,
              std::size_t offset,
              std::size_t size)
            : pool_(pool),
              device_id_(device_id),
              ptr_(ptr),
              offset_(offset),
              size_(size) {}

        ~Lease() {
            release();
        }

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        Lease(Lease&& other) noexcept
            : pool_(other.pool_),
              device_id_(other.device_id_),
              ptr_(other.ptr_),
              offset_(other.offset_),
              size_(other.size_) {
            other.pool_ = nullptr;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }

        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                release();

                pool_ = other.pool_;
                device_id_ = other.device_id_;
                ptr_ = other.ptr_;
                offset_ = other.offset_;
                size_ = other.size_;

                other.pool_ = nullptr;
                other.ptr_ = nullptr;
                other.size_ = 0;
            }

            return *this;
        }

        void* ptr() const {
            return ptr_;
        }

        std::size_t size() const {
            return size_;
        }

        explicit operator bool() const {
            return ptr_ != nullptr;
        }

    private:
        void release() {
            if (pool_ != nullptr && ptr_ != nullptr && size_ > 0) {
                pool_->release(device_id_, offset_, size_);
                pool_ = nullptr;
                ptr_ = nullptr;
                size_ = 0;
            }
        }

        DeviceWorkspacePool* pool_ = nullptr;
        int device_id_ = -1;
        void* ptr_ = nullptr;
        std::size_t offset_ = 0;
        std::size_t size_ = 0;
    };

    static DeviceWorkspacePool& instance() {
        static DeviceWorkspacePool pool;
        return pool;
    }

    Lease acquire(int device_id, std::size_t required_size) {
        if (required_size == 0) {
            return Lease();
        }

        required_size = align_up(required_size, 512);

        std::unique_lock<std::mutex> lock(mutex_);
        auto& ws = workspaces_[device_id];

        ensure_initialized_locked(device_id, ws);

        while (true) {
            auto free_it = find_free_block(ws, required_size);

            if (free_it != ws.free_blocks.end()) {
                const std::size_t offset = free_it->first;
                const std::size_t block_size = free_it->second;

                ws.free_blocks.erase(free_it);

                if (block_size > required_size) {
                    ws.free_blocks.emplace(offset + required_size, block_size - required_size);
                }

                ws.used_blocks.emplace(offset, required_size);

                char* base = static_cast<char*>(ws.ptr);
                return Lease(this,
                             device_id,
                             base + offset,
                             offset,
                             required_size);
            }

            if (ws.used_blocks.empty()) {
                grow_locked(device_id, ws, required_size);
                continue;
            }

            cv_.wait(lock);
        }
    }

private:
    struct Workspace {
        void* ptr = nullptr;
        std::size_t capacity = 0;

        // free_blocks:
        //   key   = offset
        //   value = size
        //
        // std::map keeps blocks sorted by offset, which makes merging easier.
        std::map<std::size_t, std::size_t> free_blocks;

        // used_blocks:
        //   key   = offset
        //   value = size
        std::unordered_map<std::size_t, std::size_t> used_blocks;
    };

    DeviceWorkspacePool() = default;

    ~DeviceWorkspacePool() {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto& kv : workspaces_) {
            const int device_id = kv.first;
            auto& ws = kv.second;

            if (ws.ptr != nullptr) {
                cudaSetDevice(device_id);
                cudaFree(ws.ptr);
                ws.ptr = nullptr;
            }
        }
    }

    DeviceWorkspacePool(const DeviceWorkspacePool&) = delete;
    DeviceWorkspacePool& operator=(const DeviceWorkspacePool&) = delete;

    static std::size_t align_up(std::size_t x, std::size_t alignment) {
        return ((x + alignment - 1) / alignment) * alignment;
    }

    static std::size_t default_capacity() {
        return 1 << 20; // 1 MiB
    }

    static std::size_t grow_capacity(std::size_t current,
                                     std::size_t required) {
        std::size_t next = current == 0 ? default_capacity() : current;

        while (next < required) {
            next *= 2;
        }

        return next;
    }

    typename std::map<std::size_t, std::size_t>::iterator
    find_free_block(Workspace& ws, std::size_t required_size) {
        for (auto it = ws.free_blocks.begin(); it != ws.free_blocks.end(); ++it) {
            if (it->second >= required_size) {
                return it;
            }
        }

        return ws.free_blocks.end();
    }

    void ensure_initialized_locked(int device_id, Workspace& ws) {
        if (ws.ptr != nullptr) {
            return;
        }

        CHECK_CUDA(cudaSetDevice(device_id));

        ws.capacity = default_capacity();
        CHECK_CUDA(cudaMalloc(&ws.ptr, ws.capacity));

        ws.free_blocks.clear();
        ws.used_blocks.clear();
        ws.free_blocks.emplace(0, ws.capacity);
    }

    void grow_locked(int device_id, Workspace& ws, std::size_t required_size) {
        CHECK_CUDA(cudaSetDevice(device_id));

        if (ws.ptr != nullptr) {
            CHECK_CUDA(cudaFree(ws.ptr));
            ws.ptr = nullptr;
        }

        ws.capacity = grow_capacity(ws.capacity, required_size);

        CHECK_CUDA(cudaMalloc(&ws.ptr, ws.capacity));

        ws.free_blocks.clear();
        ws.used_blocks.clear();
        ws.free_blocks.emplace(0, ws.capacity);
    }

    void release(int device_id, std::size_t offset, std::size_t size) {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            auto ws_it = workspaces_.find(device_id);
            if (ws_it == workspaces_.end()) {
                return;
            }

            auto& ws = ws_it->second;

            auto used_it = ws.used_blocks.find(offset);
            if (used_it == ws.used_blocks.end()) {
                return;
            }

            ws.used_blocks.erase(used_it);

            insert_and_merge_free_block(ws, offset, size);
        }

        cv_.notify_all();
    }

    void insert_and_merge_free_block(Workspace& ws, std::size_t offset, std::size_t size) {
        std::size_t merged_offset = offset;
        std::size_t merged_size = size;

        auto next = ws.free_blocks.lower_bound(offset);

        if (next != ws.free_blocks.end() && offset + size == next->first) {
            merged_size += next->second;
            ws.free_blocks.erase(next);
        }

        auto prev = ws.free_blocks.lower_bound(offset);

        if (prev != ws.free_blocks.begin()) {
            --prev;

            const std::size_t prev_offset = prev->first;
            const std::size_t prev_size = prev->second;

            if (prev_offset + prev_size == merged_offset) {
                merged_offset = prev_offset;
                merged_size += prev_size;
                ws.free_blocks.erase(prev);
            }
        }

        ws.free_blocks.emplace(merged_offset, merged_size);
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<int, Workspace> workspaces_;
};
