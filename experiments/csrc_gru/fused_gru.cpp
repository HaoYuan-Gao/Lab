#include <torch/extension.h>

// 正向计算 GRU
std::tuple<at::Tensor, at::Tensor> export_fused_gru_cell_forward(
    const at::Tensor& igates,
    const at::Tensor& hgates,
    const at::Tensor& hidden,
    const ::std::optional<at::Tensor>& b_ih,
    const ::std::optional<at::Tensor>& b_hh
) {
    auto result = at::_thnn_fused_gru_cell(igates, hgates, hidden, b_ih, b_hh);

    // 返回 hidden 和 workspace
    return result;
}

// 反向计算 GRU（假设已计算出梯度）
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> export_fused_gru_cell_backward(
    const at::Tensor & grad_hy, 
    const at::Tensor & workspace, 
    bool has_bias
) {
    // 计算 GRU 反向传播所需的梯度
    auto result = at::_thnn_fused_gru_cell_backward(grad_hy, workspace, has_bias);

    // 返回梯度（输入、隐藏状态、权重）
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("export_fused_gru_cell_forward", &export_fused_gru_cell_forward);
    m.def("export_fused_gru_cell_backward", &export_fused_gru_cell_backward);
}
