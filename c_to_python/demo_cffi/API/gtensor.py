from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


# API mode 会把 _gtensor_cffi 扩展模块生成到 API/build 目录。
# 这里把该目录加入 sys.path，方便直接执行：
#
#   python API/demo.py
#
_BUILD_DIR = Path(__file__).resolve().with_name("build")
if str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))

from _gtensor_cffi import ffi, lib  # noqa: E402


class Tensor:
    """Python wrapper around opaque GTensor* handle.

    Python does not know the internal C struct layout. It can only access tensor
    metadata through exported C API functions.
    """

    def __init__(self, handle, *, own: bool = True):
        if handle == ffi.NULL:
            raise ValueError("GTensor handle is NULL.")

        self._handle = handle
        self._own = own

        if own:
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
        return tuple(
            int(lib.gtensor_shape(self._handle, i))
            for i in range(self.ndim)
        )

    @property
    def dtype(self) -> int:
        return int(lib.gtensor_dtype(self._handle))

    @property
    def ref_count(self) -> int:
        return int(lib.gtensor_ref_count(self._handle))

    def retain(self) -> None:
        lib.gtensor_retain(self._handle)

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype}, "
            f"addr=0x{self.address:x}, ref_count={self.ref_count})"
        )


@dataclass
class NoiseResultView:
    """Python view for C NoiseResult.

    This class converts GTensor* outputs into Tensor wrappers. Ownership is
    transferred into Python through ffi.gc(..., gtensor_release).
    """

    raw: object

    def as_tuple(self) -> Tuple[Tensor, ...]:
        return tuple(
            Tensor(self.raw.tensors[i], own=True)
            for i in range(int(self.raw.size))
            if self.raw.tensors[i] != ffi.NULL
        )

    def addresses(self) -> Tuple[int, ...]:
        return tuple(t.address for t in self.as_tuple())


def noise_forward(input_tensor: Tensor, output_count: int = 2) -> Tuple[Tensor, ...]:
    raw = lib.gtensor_noise_forward(input_tensor.handle, output_count)
    return NoiseResultView(raw).as_tuple()
