#include <pybind11/pybind11.h>
#include <cstdint>
#include <cuda_runtime.h>

namespace py = pybind11;

void multi_afp_convert(
    const float* d_in, float* d_out,
    int* d_m_single, int* d_e_single,
    int* d_count, int* d_m_multi, int* d_e_multi,
    int n,
    int M, int S, int N, int group_up, int mask_bits, int mantissa_min, int mantissa_max,
    cudaStream_t stream
);

void multi_afp_convert_ptr(
    std::uintptr_t d_in,
    std::uintptr_t d_out,

    std::uintptr_t d_m_single,
    std::uintptr_t d_e_single,

    std::uintptr_t d_count,
    std::uintptr_t d_m_multi,
    std::uintptr_t d_e_multi,

    int n,
    int M, int S, int N, int group_up, int mask_bits, int mantissa_min, int mantissa_max,

    std::uintptr_t stream_ptr // 0 => default(legacy) stream
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    multi_afp_convert(
        reinterpret_cast<const float*>(d_in),
        reinterpret_cast<float*>(d_out),

        reinterpret_cast<int*>(d_m_single),
        reinterpret_cast<int*>(d_e_single),

        reinterpret_cast<int*>(d_count),
        reinterpret_cast<int*>(d_m_multi),
        reinterpret_cast<int*>(d_e_multi),

        n,
        M, S, N, group_up, mask_bits, mantissa_min, mantissa_max,
        stream
    );
}

PYBIND11_MODULE(afp_ext, m) {
    m.def(
        "multi_afp_convert",
        &multi_afp_convert_ptr,
        py::arg("d_in"),
        py::arg("d_out"),
        py::arg("d_m_single") = 0,
        py::arg("d_e_single") = 0,
        py::arg("d_count") = 0,
        py::arg("d_m_multi") = 0,
        py::arg("d_e_multi") = 0,
        py::arg("n"),
        py::arg("M"),
        py::arg("S"),
        py::arg("N"),
        py::arg("group_up"),
        py::arg("mask_bits"),
        py::arg("mantissa_min"),
        py::arg("mantissa_max"),
        py::arg("stream_ptr") = 0
    );
}