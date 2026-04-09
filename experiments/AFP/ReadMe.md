# MultiAFP CUDA Extension

A lightweight CUDA implementation of Multi-AFP quantization with a pure pybind11 binding interface.

This project exposes a low-level device-pointer-based API to Python for high-performance GPU execution.

## Project Structure
```
MultiAFP/
├── multi_afp_convert.cu      # CUDA kernel implementation
├── afp_bindings.cpp          # pybind11 binding
├── build.sh                  # build script
├── run_multi_afp.py          # example usage
├── afp_multi.py              # original impl
└── README.md
```

## Requirements

* CUDA Toolkit
* Python > 3.8
* pybind11
* CuPy or Torch (for GPU Memory)

Install Python dependencies:
```
pip install pybind11 cupy
```

## Build
Ensure CUDA_HOME is set correctly:
```
export CUDA_HOME=/usr/local/cuda
```
Then build the extension:
```
chmod +x build.sh
./build.sh
```
After a successful build, you should see something like:
> afp_ext.cpython-310-x86_64-linux-gnu.so

## API
```
multi_afp_convert_ptr(
    d_in: int,
    d_out: int,
    d_m_single: int,
    d_e_single: int,
    d_count: int,
    d_m_multi: int,
    d_e_multi: int,
    n: int,
    M: int,
    S: int,
    N: int,
    group_up: int,
    mask_bits: int,
    mantissa_min: int,
    mantissa_max: int,
    stream_ptr: int = 0
)
```
### Parameters
* d_in, d_out: device pointers (float32)
* d_m_single, d_e_single: output buffers for single-group mode (int32)
* d_count, d_m_multi, d_e_multi: output buffers for multi-group mode (int32)
* n: number of elements
* stream_ptr: CUDA stream pointer (0 for default stream)
All pointer arguments must be GPU device pointers.

## Modes
### 1.mask_bits == 0
Single mantissa/exponent output:
- d_m_single
- d_e_single

### 2.mask_bits > 0
Multi-group output:
* d_count
* d_m_multi
* d_e_multi

For multi-group mode, memory should be allocated with shape: <B>[n, MAX_PAIRS]</B>

## Important

This is a low-level binding:
* No automatic dtype validation
* No automatic device validation
* No memory safety checks

The caller is responsible for:
* Passing correct device pointers
* Ensuring correct data types
*Allocating sufficient GPU memory


# License
Internal / Research use.

----
If you'd like, I can also provide a more production-ready version (with error checking, architecture flags, and version badges).