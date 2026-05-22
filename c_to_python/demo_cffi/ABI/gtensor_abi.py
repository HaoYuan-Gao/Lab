from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from cffi import FFI

ffi = FFI()

# ABI mode 也需要 cdef。
#
# 这里的 cdef 只给 cffi/Python 看，用来描述动态库里的 C ABI。
# 不会生成 wrapper.c，也不会编译 Python extension。
ffi.cdef(r"""
typedef struct GTensor GTensor;

typedef struct {
    GTensor* tensors[8];
    int64_t size;
} NoiseResult;

GTensor* gtensor_create_1d(int64_t length, int64_t dtype);
GTensor* gtensor_create_2d(int64_t rows, int64_t cols, int64_t dtype);

void gtensor_retain(GTensor* tensor);
void gtensor_release(GTensor* tensor);

int64_t gtensor_ndim(GTensor* tensor);
int64_t gtensor_shape(GTensor* tensor, int64_t dim);
int64_t gtensor_dtype(GTensor* tensor);
int64_t gtensor_ref_count(GTensor* tensor);
uintptr_t gtensor_data_addr(GTensor* tensor);

NoiseResult gtensor_noise_forward(GTensor* input, int64_t output_count);
void gtensor_noise_result_release(NoiseResult result);
""")

# ABI mode：直接加载已经编译好的普通动态库。
#
# 这个过程更像 ctypes.CDLL("./libgtensor_runtime.so")。
#
# 它不会：
# - 生成 wrapper.c
# - 调用 gcc/clang 编译 Python extension
# - 生成 _gtensor_cffi.cpython-xxx.so
#
# 前提是你已经先执行：
#
#   ./build_shared_lib.sh
#
_lib_path = Path(__file__).with_name("libgtensor_runtime.so")
lib = ffi.dlopen(str(_lib_path))


class Tensor:
    """Python wrapper around opaque GTensor* handle for cffi ABI mode."""

    def __init__(self, handle, *, own: bool = True):
        if handle == ffi.NULL:
            raise ValueError("GTensor handle is NULL.")

        self._handle = handle
        self._own = own

        if own:
            # ffi.gc 会在 Python wrapper 被 GC 时调用 gtensor_release(handle)。
            self._handle = ffi.gc(self._handle, lib.gtensor_release)

    @classmethod
    def empty_1d(cls, length: int, dtype: int = 0) -> "Tensor":
        return cls(lib.gtensor_create_1d(length, dtype), own=True)

    @classmethod
    def empty_2d(cls, rows: int, cols: int, dtype: int = 0) -> "Tensor":
        return cls(lib.gtensor_create_2d(rows, cols, dtype), own=True)

    @property
    def handle(self):
        return self._handle

    @property
    def address(self) -> int:
        return int(ffi.cast("uintptr_t", self._handle))

    @property
    def ndim(self) -> int:
        return int(lib.gtensor_ndim(self._handle))

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(lib.gtensor_shape(self._handle, i)) for i in range(self.ndim))

    @property
    def dtype(self) -> int:
        return int(lib.gtensor_dtype(self._handle))

    @property
    def ref_count(self) -> int:
        return int(lib.gtensor_ref_count(self._handle))

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype}, "
            f"addr=0x{self.address:x}, ref_count={self.ref_count})"
        )


@dataclass
class NoiseResultView:
    raw: object

    def as_tuple(self) -> Tuple[Tensor, ...]:
        # 这里把 C 返回的 GTensor* 包成 Python Tensor，并把 ownership 交给 Python。
        # 因此不再额外调用 gtensor_noise_result_release(raw)，避免重复 release。
        return tuple(
            Tensor(self.raw.tensors[i], own=True)
            for i in range(int(self.raw.size))
            if self.raw.tensors[i] != ffi.NULL
        )


def noise_forward(input_tensor: Tensor, output_count: int = 2) -> Tuple[Tensor, ...]:
    raw = lib.gtensor_noise_forward(input_tensor.handle, output_count)
    return NoiseResultView(raw).as_tuple()
