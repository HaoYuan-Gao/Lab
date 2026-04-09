from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='fused_gru',
    ext_modules=[
        CppExtension(
            'fused_gru',
            ['fused_gru.cpp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
