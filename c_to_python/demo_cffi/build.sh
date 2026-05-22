#!/usr/bin/env bash
set -euo pipefail

# GTensor cffi demo 统一构建脚本。
#
# 用法：
#
#   ./build.sh api      # 只构建 cffi API mode 的 Python extension
#   ./build.sh abi      # 只构建 cffi ABI mode 使用的普通动态库
#   ./build.sh all      # 同时构建 API mode 和 ABI mode
#   ./build.sh clean    # 清理构建产物
#
# 默认：
#
#   ./build.sh
#
# 等价于：
#
#   ./build.sh all

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-all}"

build_api() {
  echo "==> Building cffi API mode extension"
  echo "    Output: API/build/_gtensor_cffi.*"

  python "${SCRIPT_DIR}/API/build_cffi.py"
}

build_abi() {
  echo "==> Building cffi ABI mode shared library"
  echo "    Output: ABI/libgtensor_runtime.so"

  bash "${SCRIPT_DIR}/ABI/build_shared_lib.sh"
}

clean() {
  echo "==> Cleaning build outputs"

  rm -rf "${SCRIPT_DIR}/API/build"
  rm -f "${SCRIPT_DIR}/ABI/libgtensor_runtime.so"
  rm -rf "${SCRIPT_DIR}/API/__pycache__" "${SCRIPT_DIR}/ABI/__pycache__"
}

case "${MODE}" in
  api)
    build_api
    ;;
  abi)
    build_abi
    ;;
  all)
    build_api
    build_abi
    ;;
  clean)
    clean
    ;;
  *)
    echo "Unknown build mode: ${MODE}" >&2
    echo "Usage: ./build.sh [api|abi|all|clean]" >&2
    exit 1
    ;;
esac
