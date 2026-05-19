#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtensor/gtensor.h"

namespace py = pybind11;
using namespace gtensor;

namespace {
DLDataType parse_dtype(const std::string& s) {
  if (s == "float32" || s == "f32") return DLDataType{kDLFloat, 32, 1};
  if (s == "float64" || s == "f64") return DLDataType{kDLFloat, 64, 1};
  if (s == "int32" || s == "i32") return DLDataType{kDLInt, 32, 1};
  if (s == "int64" || s == "i64") return DLDataType{kDLInt, 64, 1};
  if (s == "uint8" || s == "u8") return DLDataType{kDLUInt, 8, 1};
  if (s == "bool") return DLDataType{kDLBool, 8, 1};
  throw std::runtime_error("unsupported dtype: " + s);
}

py::capsule make_capsule(DLManagedTensor* dlmt) {
  return py::capsule(dlmt, "dltensor", [](PyObject* cap) {
    if (!PyCapsule_IsValid(cap, "dltensor")) return;
    auto* self = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
    if (self && self->deleter) self->deleter(self);
  });
}
}

PYBIND11_MODULE(_gtensor, m) {
  py::class_<Device>(m, "Device")
      .def(py::init([](std::string type, int id) {
        if (type == "cpu") return Device{kDLCPU, id};
        if (type == "cuda") return Device{kDLCUDA, id};
        throw std::runtime_error("unsupported device type: " + type);
      }), py::arg("type") = "cpu", py::arg("id") = 0)
      .def_readwrite("id", &Device::id)
      .def_property_readonly("type", [](const Device& d) {
        if (d.type == kDLCPU) return std::string("cpu");
        if (d.type == kDLCUDA) return std::string("cuda");
        return std::string("unknown");
      });

  py::class_<MemoryPoolStats>(m, "MemoryPoolStats")
      .def_readonly("current_bytes", &MemoryPoolStats::current_bytes)
      .def_readonly("peak_bytes", &MemoryPoolStats::peak_bytes)
      .def_readonly("total_allocated_bytes", &MemoryPoolStats::total_allocated_bytes)
      .def_readonly("total_freed_bytes", &MemoryPoolStats::total_freed_bytes)
      .def_readonly("live_blocks", &MemoryPoolStats::live_blocks)
      .def_readonly("cached_blocks", &MemoryPoolStats::cached_blocks)
      .def_readonly("cache_bytes", &MemoryPoolStats::cache_bytes);

  py::class_<GTensor>(m, "GTensor")
      .def(py::init([](std::vector<int64_t> shape, const std::string& dtype, Device device) {
        return GTensor(std::move(shape), parse_dtype(dtype), device);
      }), py::arg("shape"), py::arg("dtype") = "float32", py::arg("device") = Device{})
      .def_static("from_dlpack", [](py::capsule cap) {
        if (!PyCapsule_IsValid(cap.ptr(), "dltensor")) {
          throw std::runtime_error("expected a PyCapsule named 'dltensor'");
        }
        auto* dlmt = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap.ptr(), "dltensor"));
        PyCapsule_SetName(cap.ptr(), "used_dltensor");
        return GTensor::from_dlpack(dlmt);
      })
      .def("to_dlpack", [](const GTensor& t) { return make_capsule(t.to_dlpack()); })
      .def_property_readonly("shape", &GTensor::shape)
      .def_property_readonly("strides", &GTensor::strides)
      .def_property_readonly("ndim", &GTensor::ndim)
      .def_property_readonly("nbytes", &GTensor::nbytes)
      .def_property_readonly("numel", &GTensor::numel)
      .def_property_readonly("is_contiguous", &GTensor::is_contiguous)
      .def("__repr__", &GTensor::repr);

  m.def("memory_stats", []() { return default_cpu_allocator()->stats(); });
  m.def("reset_memory_stats", []() { default_cpu_allocator()->reset_stats(); });
}
