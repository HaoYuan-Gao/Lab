#include "gtensor/gtensor.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace gtensor {
namespace {
struct DLPackContext {
  std::shared_ptr<Storage> storage;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLManagedTensor managed{};
};

void dlpack_deleter(DLManagedTensor* self) {
  if (!self) return;
  delete static_cast<DLPackContext*>(self->manager_ctx);
}

const char* device_name(DLDeviceType t) {
  switch (t) {
    case kDLCPU: return "cpu";
    case kDLCUDA: return "cuda";
    case kDLROCM: return "rocm";
    default: return "device";
  }
}
}

std::size_t dtype_nbytes(DLDataType dtype) {
  if (dtype.lanes != 1) throw std::runtime_error("vector lanes are not supported yet");
  if (dtype.bits == 0 || dtype.bits % 8 != 0) throw std::runtime_error("sub-byte dtype is not supported yet");
  return dtype.bits / 8;
}

std::size_t compute_numel(const std::vector<int64_t>& shape) {
  if (shape.empty()) return 1;
  std::size_t n = 1;
  for (auto d : shape) {
    if (d < 0) throw std::runtime_error("negative dimension is invalid");
    n *= static_cast<std::size_t>(d);
  }
  return n;
}

std::vector<int64_t> compact_strides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

Storage::~Storage() {
  if (borrowed_dlmt) {
    if (borrowed_dlmt->deleter) borrowed_dlmt->deleter(borrowed_dlmt);
    borrowed_dlmt = nullptr;
    data = nullptr;
    return;
  }
  if (data && allocator) allocator->deallocate(data, bytes);
}

GTensor::GTensor(std::vector<int64_t> shape, DLDataType dtype, Device device)
    : shape_(std::move(shape)), strides_(compact_strides(shape_)), dtype_(dtype), device_(device) {
  if (device_.type != kDLCPU) throw std::runtime_error("only CPU allocation is implemented in this prototype");
  auto st = std::make_shared<Storage>();
  st->bytes = compute_numel(shape_) * dtype_nbytes(dtype_);
  st->device = device_;
  st->allocator = default_cpu_allocator();
  st->data = st->allocator->allocate(st->bytes);
  storage_ = std::move(st);
}

GTensor GTensor::from_dlpack(DLManagedTensor* managed) {
  if (!managed) throw std::runtime_error("null DLManagedTensor");
  GTensor t;
  const DLTensor& src = managed->dl_tensor;
  t.dtype_ = src.dtype;
  t.device_ = Device{src.device.device_type, src.device.device_id};
  t.byte_offset_ = src.byte_offset;
  t.shape_.assign(src.shape, src.shape + src.ndim);
  if (src.strides) t.strides_.assign(src.strides, src.strides + src.ndim);
  else t.strides_ = compact_strides(t.shape_);

  auto st = std::make_shared<Storage>();
  st->data = static_cast<char*>(src.data) + src.byte_offset;
  st->bytes = t.nbytes();
  st->device = t.device_;
  st->borrowed_dlmt = managed;
  t.storage_ = std::move(st);
  return t;
}

DLManagedTensor* GTensor::to_dlpack() const {
  if (!storage_) throw std::runtime_error("empty GTensor cannot be exported");
  auto* ctx = new DLPackContext();
  ctx->storage = storage_;
  ctx->shape = shape_;
  ctx->strides = strides_;
  ctx->managed.manager_ctx = ctx;
  ctx->managed.deleter = dlpack_deleter;
  ctx->managed.dl_tensor.data = storage_->data;
  ctx->managed.dl_tensor.device = DLDevice{device_.type, device_.id};
  ctx->managed.dl_tensor.ndim = static_cast<int32_t>(shape_.size());
  ctx->managed.dl_tensor.dtype = dtype_;
  ctx->managed.dl_tensor.shape = ctx->shape.data();
  ctx->managed.dl_tensor.strides = ctx->strides.data();
  ctx->managed.dl_tensor.byte_offset = 0;
  return &ctx->managed;
}

bool GTensor::is_contiguous() const {
  return strides_ == compact_strides(shape_);
}

std::string GTensor::repr() const {
  std::ostringstream os;
  os << "GTensor(shape=[";
  for (size_t i = 0; i < shape_.size(); ++i) {
    if (i) os << ", ";
    os << shape_[i];
  }
  os << "], dtype=(code=" << static_cast<int>(dtype_.code) << ", bits=" << static_cast<int>(dtype_.bits)
     << "), device=" << device_name(device_.type) << ":" << device_.id
     << ", nbytes=" << nbytes() << ")";
  return os.str();
}

}  // namespace gtensor
