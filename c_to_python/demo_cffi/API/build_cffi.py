from pathlib import Path

from cffi import FFI


ROOT_DIR = Path(__file__).resolve().parents[1]
API_DIR = Path(__file__).resolve().parent
BUILD_DIR = API_DIR / "build"

ffibuilder = FFI()

# cdef 是给 cffi/Python 侧看的：
# 告诉 cffi 有哪些 C 类型、结构体和函数，以及 Python 调用这些函数时
# 应该按照什么 ABI 传参和解析返回值。
#
# 注意：
# - cdef 不会被 gcc/clang 直接看到
# - 它不是 C 编译器的头文件
# - API mode 和 ABI mode 都需要 cdef
ffibuilder.cdef(r"""
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

ffibuilder.set_source(
    # 编译生成的 Python extension 模块名。
    #
    # 最终会生成类似：
    #
    #   _gtensor_cffi.cpython-311-x86_64-linux-gnu.so
    #
    "_gtensor_cffi",

    # 这一段会被插入到 cffi 自动生成的 wrapper.c 顶部。
    #
    # API mode 下，cffi 会先生成一个 C wrapper 文件，大概类似：
    #
    #   _gtensor_cffi.c
    #
    # 然后再调用 gcc/clang 编译这个 wrapper 文件。
    #
    # r"""#include "gtensor_runtime.h"""" 的作用是：
    # 让 C 编译器在编译 wrapper.c 时，能看到真实的 C 类型和函数声明，
    # 例如：
    #
    #   GTensor
    #   NoiseResult
    #   gtensor_create_1d(...)
    #   gtensor_noise_forward(...)
    #
    # 注意：
    # - cdef() 是给 cffi/Python 看的
    # - #include 是给 gcc/clang 看的
    # - 两者看起来重复，但服务于两个不同阶段
    #
    '#include "gtensor_runtime.h"',

    # 参与编译的真实 C 源文件。
    #
    # 最终效果大概类似：
    #
    #   gcc _gtensor_cffi.c gtensor_runtime.c ...
    #
    # 这里用绝对路径，保证无论你从项目根目录执行：
    #
    #   python API/build_cffi.py
    #
    # 还是进入 API 目录执行：
    #
    #   python build_cffi.py
    #
    # 都能正确找到 C 源文件。
    sources=[str(ROOT_DIR / "gtensor_runtime.c")],

    # 头文件搜索路径。
    #
    # 因为上面 include 了：
    #
    #   #include "gtensor_runtime.h"
    #
    # 所以这里告诉 C 编译器去项目根目录查找这个头文件。
    include_dirs=[str(ROOT_DIR)],
)


if __name__ == "__main__":
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    ffibuilder.compile(
        verbose=True,

        # tmpdir 是 cffi 中间构建目录。
        #
        # 会放置自动生成的 wrapper.c、临时 object 文件等。
        tmpdir=str(BUILD_DIR),

        # target 是最终 extension 输出文件模板，不只是目录。
        #
        # 最终会生成到：
        #
        #   API/build/_gtensor_cffi.cpython-xxx.so
        #
        target=str(BUILD_DIR / "_gtensor_cffi.*"),
    )
