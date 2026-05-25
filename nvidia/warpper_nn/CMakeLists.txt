cmake_minimum_required(VERSION 3.18)

project(cuda_nn_example LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# 如果没找到 CUDNN, 使用 find /home/path -name "cudnn.h" 2>/dev/null 来查找
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    message(STATUS "CONDA_PREFIX = ${CONDA_PREFIX_PATH}")

    file(GLOB CONDA_CUDNN_INCLUDE_HINTS
        "${CONDA_PREFIX_PATH}/lib/python*/site-packages/nvidia/cudnn/include"
    )

    file(GLOB CONDA_CUDNN_LIBRARY_HINTS
        "${CONDA_PREFIX_PATH}/lib/python*/site-packages/nvidia/cudnn/lib"
    )
else()
    set(CONDA_PREFIX_PATH "")
    set(CONDA_CUDNN_INCLUDE_HINTS "")
    set(CONDA_CUDNN_LIBRARY_HINTS "")
    message(WARNING "CONDA_PREFIX is not set")
endif()

find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    HINTS
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CONDA_CUDNN_INCLUDE_HINTS}
        /usr/local/cuda/include
        /usr/include
        /usr/include/x86_64-linux-gnu
    REQUIRED
)

find_library(CUDNN_LIBRARY
    NAMES cudnn libcudnn libcudnn.so.9
    HINTS
        ${CUDAToolkit_LIBRARY_DIR}
        ${CONDA_CUDNN_LIBRARY_HINTS}
        /usr/local/cuda/lib64
        /usr/lib
        /usr/lib/x86_64-linux-gnu
    REQUIRED
)

message(STATUS "CUDNN_INCLUDE_DIR = ${CUDNN_INCLUDE_DIR}")
message(STATUS "CUDNN_LIBRARY     = ${CUDNN_LIBRARY}")

add_executable(cuda_nn_example
    example.cu
    cuda_nn/cublaslt_gemm.cpp
    cuda_nn/cudnn_legacy_conv.cpp
    cuda_nn/cudnn_frontend_conv.cpp
)

target_include_directories(cuda_nn_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDNN_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include
)

target_link_libraries(cuda_nn_example PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cuda_driver
    CUDA::nvrtc
    ${CUDNN_LIBRARY}
    Threads::Threads
)

target_compile_options(cuda_nn_example PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=128>
)
