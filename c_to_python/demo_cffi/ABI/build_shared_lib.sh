#!/usr/bin/env bash
set -euo pipefail

# 构建 cffi ABI mode / ctypes 都可以直接加载的普通动态库。
#
# 这个库不是 Python extension，不会生成：
#
#   xxx.cpython-311-x86_64-linux-gnu.so
#
# 而是生成普通 C ABI 动态库：
#
#   libgtensor_runtime.so
#
# 因此它不绑定具体 Python 版本。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${SCRIPT_DIR}/libgtensor_runtime.so"

cc="${CC:-gcc}"

"${cc}" \
  -shared \
  -fPIC \
  -O2 \
  -I"${ROOT_DIR}" \
  "${ROOT_DIR}/gtensor_runtime.c" \
  -o "${OUT}"

echo "Built ABI shared library: ${OUT}"
