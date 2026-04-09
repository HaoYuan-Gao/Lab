#include <torch/extension.h>
#include <ATen/core/Dict.h>
#include <iostream>

static at::Tensor foo_impl(
    const at::Tensor& x,
    const c10::Dict<std::string, int64_t>& meta
) {
    std::cout << "meta size = " << meta.size() << std::endl;
    for (const auto& item : meta) {
        std::cout << item.key() << " : " << item.value() << std::endl;
    }
    return x;
}

// boxed kernel: (OperatorHandle, Stack*) 版本
static void foo_boxed(const c10::OperatorHandle& /*op*/, c10::Stack* stack) {
    // schema: foo(Tensor x, Any meta) -> Tensor
    // Stack 入参顺序：x, meta（top 是最后一个参数 meta）
    auto meta_iv = std::move(stack->back());
    stack->pop_back();

    auto x_iv = std::move(stack->back());
    stack->pop_back();

    auto meta = meta_iv.to<c10::Dict<std::string, int64_t>>();
    auto x = x_iv.toTensor();

    auto y = foo_impl(x, meta);
    stack->push_back(std::move(y));
}

std::string echo_str(c10::string_view msg) {
    return std::string(msg) + " csrc!";
}

TORCH_LIBRARY(witin, m) {
    // 注意：schema 里不要写 Dict (torchscript 不支持)，用 Any 占位
    m.def("foo(Tensor x, Any meta) -> Tensor");
    m.def("echo_str(str msg) -> str");
}

TORCH_LIBRARY_IMPL(witin, CPU, m) {
    m.impl("foo", torch::CppFunction::makeFromBoxedFunction<&foo_boxed>());
}

TORCH_LIBRARY_IMPL(witin, CatchAll, m) {
    m.impl("echo_str", &echo_str);
}

// 确保生成带 python 符号的 so
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}