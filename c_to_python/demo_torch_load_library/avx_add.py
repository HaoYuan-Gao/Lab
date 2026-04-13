import ctypes
import torch

lib = ctypes.CDLL("./libavx_add.so")

lib.sample_add_avx.argtypes = [
    ctypes.c_void_p,  # x
    ctypes.c_void_p,  # y
    ctypes.c_void_p,  # out
    ctypes.c_int      # n
]
lib.sample_add_avx.restype = None


def avx_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.device.type == "cpu"
    assert y.device.type == "cpu"
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert x.shape == y.shape

    out = torch.empty_like(x)

    lib.sample_add_avx(
        x.data_ptr(),
        y.data_ptr(),
        out.data_ptr(),
        x.numel()
    )
    return out


if __name__ == "__main__":
    x = torch.randn(16, dtype=torch.float32)
    y = torch.randn(16, dtype=torch.float32)

    out = avx_add(x, y)
    print(out)
    print(torch.allclose(out, x + y))