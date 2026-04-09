from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="dispatcher",
    ext_modules=[
        CppExtension(
            name="dispatcher",
            sources=["scale_mul.cpp"],
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
