from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="witin",
    ext_modules=[
        CppExtension(
            name="witin",  # Python import 的模块名：import witin
            sources=["dict_test.cpp"],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
