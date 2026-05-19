# GTensor DLPack Prototype

一个最小但可扩展的 GTensor 原型：

- 使用 DLPack 的 `DLManagedTensor` 与 PyTorch Tensor 互转；
- 当前实现 CPU 连续内存分配；
- 提供 CPU memory pool 和内存使用统计；
- 算子层暂不实现，后续可以在 `GTensor` 上添加 dispatcher / kernel registry。

## Build
直接安装

```bash
pip install scikit-build-core pybind11 torch
pip install -e .
```

生成 whl 文件
```bash
pip install build
python -m build --wheel --no-isolation -v
```

删除编译信息
```bash
rm -rf build dist _skbuild
pip uninstall -y gtensor
```

## Example

```bash
python examples/torch_roundtrip.py
```

## 设计说明

- `GTensor::from_dlpack`：消费 PyTorch 导出的 DLPack capsule，GTensor 持有 `DLManagedTensor*`，析构时调用原始 deleter。
- `GTensor::to_dlpack`：导出新的 `DLManagedTensor`，manager context 内保存 `shared_ptr<Storage>`，确保 PyTorch 侧还在使用时底层内存不会提前释放。
- `MemoryPoolStats`：记录 current / peak / total allocated / freed / live blocks / cache blocks。
- 当前 CUDA allocator 只预留 device 抽象，尚未实现。
