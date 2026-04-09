#include <torch/extension.h>
#include <torch/torch.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_set>

// 使用 py::object 对象
#include <pybind11/pybind11.h>
// 使用 toIValue 函数
#include <torch/csrc/jit/python/pybind_utils.h>

namespace py = pybind11;

static std::mutex g_mu;
static std::unordered_set<std::string> g_loaded;
static std::vector<void*> g_handles;

static void dlopen_keepalive_once(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_loaded.count(path)) return;

    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
    }
    g_handles.push_back(h);
    g_loaded.insert(path);

    std::cerr << "[consumer] dlopen ok: " << path << "\n";
}

torch::Tensor
infer_by_producer_so(
    std::string path,
    int64_t bias,
    const torch::Tensor& x,
    const torch::Tensor& w
) {
    // 1) load producer.so once
    // python 代码 import 后不需要再调用
    // dlopen_keepalive_once(path);

    auto& disp = c10::Dispatcher::singleton();

    // 2) lookup ops
    auto op_make  = disp.findSchemaOrThrow("producer::make_state", "");
    auto op_infer = disp.findSchemaOrThrow("producer::infer", "");

    auto schema = op_infer.schema();
    size_t nret = schema.returns().size();
    std::cout << "return size is:" << nret << std::endl;

    // 3) make_state(bias)
    std::vector<c10::IValue> st_stack;
    st_stack.emplace_back(bias);
    disp.callBoxed(op_make, &st_stack);

    if (st_stack.empty())
        throw std::runtime_error("make_state returned empty stack");

    c10::IValue st = st_stack.back();

    // 4) infer(state, x, w)
    std::vector<c10::IValue> stack;
    stack.emplace_back(st);
    stack.emplace_back(x);
    stack.emplace_back(w);

    disp.callBoxed(op_infer, &stack);
    std::cout << "callBoxed return size is:" << nret << std::endl;

    if (stack.empty() || !stack.back().isTensor())
        throw std::runtime_error("infer did not return Tensor");

    // 5) return safe tensor
    return stack[0].toTensor().contiguous().clone();
}

at::Tensor infer_by_producer_st(
    const at::Tensor& x,
    const at::Tensor& w,
    const c10::IValue& st // 通过 IValue 接收 st
) {
    auto& disp = c10::Dispatcher::singleton();
    auto op_infer = disp.findSchemaOrThrow("producer::infer", "");

    std::vector<c10::IValue> stack;
    stack.emplace_back(st);
    stack.emplace_back(x);
    stack.emplace_back(w);

    disp.callBoxed(op_infer, &stack);

    if (stack.empty() || !stack.back().isTensor())
        throw std::runtime_error("infer did not return Tensor");

    // 5) return safe tensor
    return stack[0].toTensor().contiguous().clone();
}

at::Tensor infer_by_producer_py(
    const at::Tensor& x,
    const at::Tensor& w,
    py::object st_py  // 接收 producer 的 state
) {
    c10::IValue st = torch::jit::toIValue(st_py, c10::AnyType::get());
    TORCH_CHECK(st.isObject(), "st must be a TorchScript object!");

    auto& disp = c10::Dispatcher::singleton();
    auto op_infer = disp.findSchemaOrThrow("producer::infer", "");

    std::vector<c10::IValue> stack;
    stack.emplace_back(st);
    stack.emplace_back(x);
    stack.emplace_back(w);

    disp.callBoxed(op_infer, &stack);

    if (stack.empty() || !stack.back().isTensor())
        throw std::runtime_error("infer did not return Tensor");

    // 5) return safe tensor
    return stack[0].toTensor().contiguous().clone();
}

TORCH_LIBRARY(consumer, m) {
    m.def("infer_st(Tensor x, Tensor w, Any st) -> Tensor");
}

TORCH_LIBRARY_IMPL(consumer, CPU, m) {
    m.impl("infer_st", TORCH_FN(infer_by_producer_st));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "infer_by_producer_so",
        &infer_by_producer_so,
        py::arg("path"),
        py::arg("bias"),
        py::arg("x"),
        py::arg("w")
    );

    m.def("infer_py", &infer_by_producer_py);
}
