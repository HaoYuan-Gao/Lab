cmake_minimum_required(VERSION 3.20)
project(gtensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================
# pybind11
# ============================================================

find_package(pybind11 CONFIG REQUIRED)

# ============================================================
# Python Extension
# ============================================================

pybind11_add_module(_gtensor
    src/pybind_module.cpp
    src/gtensor.cpp
    src/memory_pool.cpp
)

target_include_directories(_gtensor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ============================================================
# Install Python extension into gtensor package
# ============================================================

install(TARGETS _gtensor
    LIBRARY DESTINATION gtensor
    RUNTIME DESTINATION gtensor
    ARCHIVE DESTINATION gtensor
)
