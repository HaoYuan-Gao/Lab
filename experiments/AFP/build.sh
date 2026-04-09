#!/usr/bin/env bash
set -euo pipefail

# ===== configurable =====
MOD_NAME="afp_ext"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CXX="${CXX:-g++}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"

# ===== python / pybind11 flags =====
PY_EXT_SUFFIX="$(python3-config --extension-suffix)"
PY_INCLUDES="$(python3-config --includes)"
PYBIND11_INCLUDES="$(python3 -m pybind11 --includes)"   # contains python include too, but fine

# ===== cuda flags =====
CUDA_INC="-I${CUDA_HOME}/include"
CUDA_LIB="-L${CUDA_HOME}/lib64"

# ===== build dir =====
mkdir -p build

echo "[1/3] Compile CUDA (.cu) -> object"
"${NVCC}" -c multi_afp_convert.cu -o build/multi_afp_convert.o \
  -O3 --use_fast_math -lineinfo \
  -Xcompiler -fPIC ${CUDA_INC}

echo "[2/3] Compile pybind11 binding (.cpp) -> object"
${CXX} -c afp_bindings.cpp -o build/afp_bindings.o \
  -O3 -fPIC \
  ${PYBIND11_INCLUDES} ${PY_INCLUDES} ${CUDA_INC}

echo "[3/3] Link -> Python extension"
${CXX} -shared -o "${MOD_NAME}${PY_EXT_SUFFIX}" \
  build/afp_bindings.o build/multi_afp_convert.o \
  ${CUDA_LIB} -lcudart

echo "Built: ${MOD_NAME}${PY_EXT_SUFFIX}"
