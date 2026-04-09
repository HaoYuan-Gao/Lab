cmake_minimum_required(VERSION 3.20)
project(CPP_TO_PY LANGUAGES CXX CUDA)

set(PYBIND11_FINDPYTHON ON)

# Set C++/CUDA standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

file(GLOB CU_FILES ./*.cu)
file(GLOB CPP_FILES ./*.cpp)

set(demo demo_pybind11)

add_library(${demo} SHARED ${CU_FILES} ${CPP_FILES})

target_compile_options(${demo} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
)

target_link_libraries(${demo} PRIVATE 
    pybind11::pybind11
    CUDA::cudart
) 

set_target_properties(${demo} PROPERTIES
    PREFIX ""
    OUTPUT_NAME ${demo}
    SUFFIX "${PYTHON_EXT_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
