import ctypes
import os
import torch

_lib = ctypes.CDLL(os.path.abspath("./lib/libavx_add.so"))
_lib.sample_add_avx.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]
_lib.sample_add_avx.restype = None

### torch op checklist
# forward
# fake
# autograd
# device
# dtype
# in-place / out variant
# vmap / batching
# 参考: https://docs.pytorch.org/docs/stable/library.html#


@torch.library.custom_op("myops::avx_add", mutates_args=())
def avx_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cpu" or y.device.type != "cpu":
        raise RuntimeError("CPU only")
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise RuntimeError("float32 only")
    if x.shape != y.shape:
        raise RuntimeError("shape mismatch")
    if not x.is_contiguous() or not y.is_contiguous():
        raise RuntimeError("contiguous only")

    out = torch.empty_like(x)
    _lib.sample_add_avx(x.data_ptr(), y.data_ptr(), out.data_ptr(), x.numel())
    return out


@torch.library.register_fake("myops::avx_add")
def _(x, y):
    if x.shape != y.shape:
        raise RuntimeError("shape mismatch")
    return torch.empty_like(x)

if __name__ == "__main__":
    x = torch.randn(16, dtype=torch.float32)
    y = torch.randn(16, dtype=torch.float32)

    out = torch.ops.myops.avx_add(x, y)
    print(out)
    print(torch.allclose(out, x + y))