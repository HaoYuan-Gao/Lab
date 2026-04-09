#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
BUILD_TYPE="${BUILD_TYPE:-Release}"

# CMake expects a semicolon-separated list, not a bash array
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-70;75;80;86;89;90}"

# Prefer CUDA_HOME, fallback to /usr/local/cuda
CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}"
PYBIND11_DIR=$(python -m pybind11 --cmakedir)

echo "==> Build type:          ${BUILD_TYPE}"
echo "==> Build directory:     ${BUILD_DIR}"
echo "==> CUDA toolkit path:   ${CUDA_PATH}"
echo "==> Python directory:    ${PYBIND11_DIR}"
echo "==> CUDA architectures:  ${CUDA_ARCHITECTURES}"

if [ ! -d "${CUDA_PATH}" ]; then
    echo "Error: CUDA toolkit path does not exist: ${CUDA_PATH}"
    exit 1
fi

cmake -S . -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DCUDAToolkit_ROOT="${CUDA_PATH}" \
    -Dpybind11_DIR="${PYBIND11_DIR}"

ninja -C ${BUILD_DIR}
