# GTensor cffi opaque handle demo

This demo shows a GTensor-style runtime binding design using cffi.

It includes:

- opaque handle: `typedef struct GTensor GTensor;`
- multi-output result: `NoiseResult { GTensor* tensors[8]; int64_t size; }`
- Python wrapper around `GTensor*`
- ownership management with `ffi.gc(handle, lib.gtensor_release)`
- shape/dtype/refcount access through C API functions only
- both cffi API mode and cffi ABI mode demos
- unified `build.sh` to build API mode / ABI mode separately

## Directory layout

| Path | Purpose |
|---|---|
| `gtensor_runtime.h` | Public C header. Exposes opaque `GTensor*` and C API. |
| `gtensor_runtime.c` | C implementation. Defines the real internal `struct GTensor`. |
| `build.sh` | Unified build script for API mode and ABI mode. |
| `API/build_cffi.py` | cffi API mode builder. Generates a Python extension under `API/build/`. |
| `API/gtensor.py` | Python wrapper using the generated `_gtensor_cffi` extension. |
| `API/demo.py` | Demo for cffi API mode. |
| `ABI/build_shared_lib.sh` | Builds a plain C ABI shared library: `ABI/libgtensor_runtime.so`. |
| `ABI/gtensor_abi.py` | Python wrapper using cffi ABI mode and `ffi.dlopen(...)`. |
| `ABI/demo_abi.py` | Demo for cffi ABI mode. |

## Build

Install cffi first:

```bash
python -m pip install cffi
```

Build everything:

```bash
./build.sh
```

Equivalent to:

```bash
./build.sh all
```

Build only cffi API mode:

```bash
./build.sh api
```

Build only cffi ABI mode:

```bash
./build.sh abi
```

Clean generated outputs:

```bash
./build.sh clean
```

## Run demos

After building API mode:

```bash
python API/demo.py
```

After building ABI mode:

```bash
python ABI/demo_abi.py
```

## Option 1: cffi API mode

API mode is similar to building a Python extension module.

```bash
./build.sh api
python API/demo.py
```

This generates files under:

```text
API/build/
```

Typical output:

```text
API/build/_gtensor_cffi.c
API/build/_gtensor_cffi.cpython-311-x86_64-linux-gnu.so
```

This extension is tied to the Python version.

## Option 2: cffi ABI mode

ABI mode is closer to `ctypes.CDLL(...)`.

```bash
./build.sh abi
python ABI/demo_abi.py
```

This builds and loads:

```text
ABI/libgtensor_runtime.so
```

through:

```python
lib = ffi.dlopen("ABI/libgtensor_runtime.so")
```

This mode does not generate a Python extension and is closer to the runtime ABI style.

## API mode vs ABI mode

| Mode | Generates wrapper.c | Builds Python extension | Needs existing `.so` | Python-version dependent |
|---|---:|---:|---:|---:|
| cffi API mode | Yes | Yes | No | Yes |
| cffi ABI mode | No | No | Yes | No |

Both modes still need `ffi.cdef(...)` because cffi must know the C function signatures and struct layout from Python side.

## Design notes

Python only sees this:

```c
typedef struct GTensor GTensor;
```

It does not know the real layout of `struct GTensor`. This means the C runtime can later add fields such as device, stride, allocator, stream, storage, or grad metadata without changing the Python-side opaque handle design.

`NoiseResult` uses a fixed-size output array plus `size`:

```c
typedef struct {
    GTensor* tensors[8];
    int64_t size;
} NoiseResult;
```

This keeps the ABI simple and avoids malloc/free protocols for result arrays.

## Chinese note

`API/build_cffi.py` 是 cffi API mode：它会生成一个 C wrapper，再编译成 Python extension，输出到 `API/build/`。

`ABI/gtensor_abi.py` 是 cffi ABI mode：它直接加载已经编译好的 `ABI/libgtensor_runtime.so`，更像 `ctypes.CDLL`。

`build.sh` 是统一构建入口：

```bash
./build.sh api
./build.sh abi
./build.sh all
./build.sh clean
```
