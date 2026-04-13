import ctypes
import os
from typing import Optional

import torch


# ============================================================
# 配置
# ============================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CPU_SO = os.path.join(_THIS_DIR, "lib/libavx_add.so")
_CUDA_SO = os.path.join(_THIS_DIR, "lib/libcuda_add.so")

# 避免重复注册
_LOADED = False

# 全局 LIB 定义
_LIB_DEF = None

# ctypes 动态库句柄
_CPU_LIB: Optional[ctypes.CDLL] = None
_CUDA_LIB: Optional[ctypes.CDLL] = None


# ============================================================
# 动态库加载
# ============================================================
def _load_cpu_lib() -> ctypes.CDLL:
    global _CPU_LIB
    if _CPU_LIB is None:
        if not os.path.exists(_CPU_SO):
            raise FileNotFoundError(f"CPU library not found: {_CPU_SO}")

        lib = ctypes.CDLL(_CPU_SO)
        lib.sample_add_avx.argtypes = [
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # y
            ctypes.c_void_p,  # out
            ctypes.c_int,     # n
        ]
        lib.sample_add_avx.restype = None
        _CPU_LIB = lib
    return _CPU_LIB


def _load_cuda_lib() -> ctypes.CDLL:
    global _CUDA_LIB
    if _CUDA_LIB is None:
        if not os.path.exists(_CUDA_SO):
            raise FileNotFoundError(f"CUDA library not found: {_CUDA_SO}")

        lib = ctypes.CDLL(_CUDA_SO)

        # int sample_add_cuda(const float* x, const float* y, float* out,
        #                     int n, int device, unsigned long long stream)
        lib.sample_add_cuda.argtypes = [
            ctypes.c_void_p,   # x
            ctypes.c_void_p,   # y
            ctypes.c_void_p,   # out
            ctypes.c_int,      # n
            ctypes.c_int,      # device
            ctypes.c_uint64,   # stream
        ]
        lib.sample_add_cuda.restype = ctypes.c_int

        # int sample_add_cuda_sync(...)
        lib.sample_add_cuda_sync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        lib.sample_add_cuda_sync.restype = ctypes.c_int

        # const char* sample_cuda_get_error_string(int code)
        lib.sample_cuda_get_error_string.argtypes = [ctypes.c_int]
        lib.sample_cuda_get_error_string.restype = ctypes.c_char_p

        # int sample_cuda_runtime_version()
        lib.sample_cuda_runtime_version.argtypes = []
        lib.sample_cuda_runtime_version.restype = ctypes.c_int

        _CUDA_LIB = lib
    return _CUDA_LIB


# ============================================================
# 错误处理
# ============================================================
def _cuda_error_string(err_code: int) -> str:
    try:
        lib = _load_cuda_lib()
        msg = lib.sample_cuda_get_error_string(ctypes.c_int(err_code))
        if msg:
            return msg.decode("utf-8")
    except Exception:
        pass
    return f"unknown CUDA error (code={err_code})"


def _cuda_check(err_code: int, prefix: str = "CUDA call failed") -> None:
    if err_code != 0:
        msg = _cuda_error_string(err_code)
        raise RuntimeError(f"{prefix}: {msg} (code={err_code})")


# ============================================================
# 参数检查
# ============================================================
def _check_common(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise RuntimeError(
            f"only float32 is supported, got x={x.dtype}, y={y.dtype}"
        )
    if x.shape != y.shape:
        raise RuntimeError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if not x.is_contiguous() or not y.is_contiguous():
        raise RuntimeError("x and y must be contiguous")


def _check_out(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    _check_common(x, y)
    if out.dtype != torch.float32:
        raise RuntimeError(f"out must be float32, got {out.dtype}")
    if out.shape != x.shape:
        raise RuntimeError(
            f"out shape mismatch: expected {tuple(x.shape)}, got {tuple(out.shape)}"
        )
    if not out.is_contiguous():
        raise RuntimeError("out must be contiguous")
    if out.device != x.device:
        raise RuntimeError(
            f"out must be on same device as x, got out={out.device}, x={x.device}"
        )


# ============================================================
# CPU 调用
# ============================================================
def _cpu_add_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cpu" or y.device.type != "cpu":
        raise RuntimeError("CPU impl got non-CPU tensor")
    _check_common(x, y)

    out = torch.empty_like(x)
    lib = _load_cpu_lib()

    lib.sample_add_avx(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(x.numel()),
    )
    return out


def _cpu_add_out_impl(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cpu" or y.device.type != "cpu" or out.device.type != "cpu":
        raise RuntimeError("CPU impl got non-CPU tensor")
    _check_out(x, y, out)

    lib = _load_cpu_lib()
    lib.sample_add_avx(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(x.numel()),
    )
    return out


# ============================================================
# CUDA 调用
# ============================================================
def _current_cuda_stream_handle(device_index: int) -> int:
    # torch.cuda.current_stream(device) 返回当前 device 的当前 stream
    stream = torch.cuda.current_stream(device_index)
    return int(stream.cuda_stream)


def _cuda_add_impl(x: torch.Tensor, y: torch.Tensor, sync: bool = False) -> torch.Tensor:
    if x.device.type != "cuda" or y.device.type != "cuda":
        raise RuntimeError("CUDA impl got non-CUDA tensor")
    _check_common(x, y)

    if x.device != y.device:
        raise RuntimeError(f"x and y must be on same CUDA device, got {x.device} vs {y.device}")

    out = torch.empty_like(x)

    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    stream_handle = _current_cuda_stream_handle(device_index)
    lib = _load_cuda_lib()

    if sync:
        err = lib.sample_add_cuda_sync(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(x.numel()),
            ctypes.c_int(device_index),
            ctypes.c_uint64(stream_handle),
        )
        _cuda_check(err, "sample_add_cuda_sync")
    else:
        err = lib.sample_add_cuda(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(x.numel()),
            ctypes.c_int(device_index),
            ctypes.c_uint64(stream_handle),
        )
        _cuda_check(err, "sample_add_cuda")

    return out


def _cuda_add_out_impl(
    x: torch.Tensor,
    y: torch.Tensor,
    out: torch.Tensor,
    sync: bool = False,
) -> torch.Tensor:
    if x.device.type != "cuda" or y.device.type != "cuda" or out.device.type != "cuda":
        raise RuntimeError("CUDA impl got non-CUDA tensor")
    _check_out(x, y, out)

    if x.device != y.device or x.device != out.device:
        raise RuntimeError(
            f"x, y, out must be on same CUDA device, got {x.device}, {y.device}, {out.device}"
        )

    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    stream_handle = _current_cuda_stream_handle(device_index)
    lib = _load_cuda_lib()

    if sync:
        err = lib.sample_add_cuda_sync(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(x.numel()),
            ctypes.c_int(device_index),
            ctypes.c_uint64(stream_handle),
        )
        _cuda_check(err, "sample_add_cuda_sync")
    else:
        err = lib.sample_add_cuda(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(x.numel()),
            ctypes.c_int(device_index),
            ctypes.c_uint64(stream_handle),
        )
        _cuda_check(err, "sample_add_cuda")

    return out


# ============================================================
# 注册逻辑：define + impl + fake
# ============================================================
def load() -> None:
    global _LOADED, _LIB_DEF
    if _LOADED:
        return

    # 先定义 schema
    _LIB_DEF = torch.library.Library("mylib", "DEF")
    _LIB_DEF.define("add(Tensor x, Tensor y) -> Tensor")
    _LIB_DEF.define("add_out(Tensor x, Tensor y, Tensor(a!) out) -> Tensor(a!)")

    # CPU backend
    @torch.library.impl("mylib::add", "cpu")
    def add_cpu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _cpu_add_impl(x, y)

    @torch.library.impl("mylib::add_out", "cpu")
    def add_out_cpu(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return _cpu_add_out_impl(x, y, out)

    # CUDA backend
    @torch.library.impl("mylib::add", "cuda")
    def add_cuda(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _cuda_add_impl(x, y, sync=False)

    @torch.library.impl("mylib::add_out", "cuda")
    def add_out_cuda(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return _cuda_add_out_impl(x, y, out, sync=False)

    # fake 实现：只做 shape / dtype / device 元信息推导
    @torch.library.register_fake("mylib::add")
    def add_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise RuntimeError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
        if x.dtype != torch.float32 or y.dtype != torch.float32:
            raise RuntimeError("fake add only supports float32 in this demo")
        return torch.empty_like(x)

    @torch.library.register_fake("mylib::add_out")
    def add_out_fake(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise RuntimeError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
        if out.shape != x.shape:
            raise RuntimeError(
                f"out shape mismatch: expected {tuple(x.shape)}, got {tuple(out.shape)}"
            )
        if x.dtype != torch.float32 or y.dtype != torch.float32 or out.dtype != torch.float32:
            raise RuntimeError("fake add_out only supports float32 in this demo")
        return out

    _LOADED = True


# ============================================================
# 可选：调试入口
# ============================================================
def add_debug_cuda(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    调试版 CUDA 调用：会同步当前 stream，便于尽早暴露 kernel 错误。
    """
    return _cuda_add_impl(x, y, sync=True)


def add_out_debug_cuda(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """
    调试版 CUDA out 调用：会同步当前 stream，便于尽早暴露 kernel 错误。
    """
    return _cuda_add_out_impl(x, y, out, sync=True)


# ============================================================
# import 时自动注册
# ============================================================
load()


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    print("registered ops:")
    print("  torch.ops.mylib.add")
    print("  torch.ops.mylib.add_out")

    # CPU test
    x = torch.randn(16, dtype=torch.float32)
    y = torch.randn(16, dtype=torch.float32)
    z = torch.ops.mylib.add(x, y)
    print("cpu add ok:", torch.allclose(z, x + y))

    out = torch.empty_like(x)
    z2 = torch.ops.mylib.add_out(x, y, out)
    print("cpu add_out ok:", torch.allclose(out, x + y))
    print("cpu add_out returned out:", z2.data_ptr() == out.data_ptr())

    # CUDA test
    if torch.cuda.is_available():
        xc = torch.randn(16, device="cuda", dtype=torch.float32)
        yc = torch.randn(16, device="cuda", dtype=torch.float32)

        zc = torch.ops.mylib.add(xc, yc)
        print("cuda add ok:", torch.allclose(zc, xc + yc))

        outc = torch.empty_like(xc)
        zc2 = torch.ops.mylib.add_out(xc, yc, outc)
        print("cuda add_out ok:", torch.allclose(outc, xc + yc))
        print("cuda add_out returned out:", zc2.data_ptr() == outc.data_ptr())

        # 调试版
        zd = add_debug_cuda(xc, yc)
        print("cuda debug add ok:", torch.allclose(zd, xc + yc))
    else:
        print("CUDA not available, skip CUDA test")