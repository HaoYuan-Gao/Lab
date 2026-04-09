import torch
import torch.nn as nn

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import fused_gru

# ====== 配置 ======
input_size = 16     # 输入特征维度
hidden_size = 32    # 隐藏层维度
num_layers = 1      # GRU 层数
seq_len = 626         # 序列长度
batch_size = 4      # batch size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUProbe(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, batch_first: bool = False):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first
        )
        self.num_layers = num_layers

        # 这些属性用来存 forward & backward 的中间结果
        self.last_input = None
        self.last_h0 = None
        self.last_output = None
        self.output_grad = None

        self.weights = {}
        self.biases = {}

    def forward(self, x, h0=None):
        # 记录 input / hidden
        self.last_input = x
        self.last_h0 = h0

        # 真正 forward
        y, hn = self.gru(x, h0)

        # 注册 hook，在 backward 时拿到 dy/dLoss
        def save_output_grad(grad):
            # grad 形状和 y 一样: [B, T, H]
            self.output_grad = grad.detach()

        y.register_hook(save_output_grad)
        self.last_output = y

        # 记录当前权重和 bias（只读引用）
        # 对于 num_layers=1, bidirectional=False，只会有 l0 这一组
        self.weights = {
            "weight_ih_l0": self.gru.weight_ih_l0,
            "weight_hh_l0": self.gru.weight_hh_l0,
        }
        self.biases = {
            "bias_ih_l0": self.gru.bias_ih_l0,
            "bias_hh_l0": self.gru.bias_hh_l0,
        }

        return y, hn

# ====== 使用示例 ======
model = GRUProbe(input_size, hidden_size, num_layers, batch_first=True).to(device)
x = torch.randn(batch_size, seq_len, input_size, device=device, requires_grad=True)
h0 = torch.randn(num_layers, batch_size, hidden_size, device=device)
y, hn = model(x, h0)
loss = y.sum()
loss.backward()


def gru_backward_input_fused(
    grad_output,   # [B, T, H]  dL/dy_t
    x,             # [B, T, I]  输入
    h0,            # [1, B, H]  初始 hidden
    weight_ih,     # [3H, I]
    weight_hh,     # [3H, H]
    bias_ih,       # [3H] 或 None
    bias_hh        # [3H] 或 None
):
    """
    使用 fused_gru.export_fused_gru_cell_forward / backward
    手动做一遍 BPTT，返回:
      - dx:        [B, T, I]
      - grad_w_ih: [3H, I]
      - grad_w_hh: [3H, H]
      - grad_b_ih: [3H] 或 None
      - grad_b_hh: [3H] 或 None
    """
    device = x.device
    dtype = x.dtype

    B, T, I = x.shape
    H = weight_hh.shape[1]

    has_bias = (bias_ih is not None) and (bias_hh is not None)

    # ---- 前向：逐时间步调用 fused forward，保存 workspace / x_t / h_prev ----
    h_prev = h0[0]                                  # [B, H]
    hs = []
    workspaces = []
    xs = []          # 每个时间步的 x_t
    h_prevs = []     # 每个时间步对应的 h_{t-1}

    for t in range(T):
        x_t = x[:, t, :]                            # [B, I]

        # 记录下来，用于反向累积参数梯度
        xs.append(x_t)
        h_prevs.append(h_prev)

        # igates = x_t @ W_ih^T
        igates = x_t @ weight_ih.t()                # [B, 3H]
        # hgates = h_prev @ W_hh^T
        hgates = h_prev @ weight_hh.t()             # [B, 3H]

        hidden_t, workspace_t = fused_gru.export_fused_gru_cell_forward(
            igates, hgates, h_prev, bias_ih, bias_hh
        )

        hs.append(hidden_t)                         # [B, H]
        workspaces.append(workspace_t)              # 保存 workspace
        h_prev = hidden_t

    # ---- 反向：从后往前用 fused backward 回传 ----
    dx = torch.zeros_like(x)                        # [B, T, I]
    dh_next = torch.zeros_like(h0[0])               # [B, H]

    grad_w_ih = torch.zeros_like(weight_ih)         # [3H, I]
    grad_w_hh = torch.zeros_like(weight_hh)         # [3H, H]
    grad_b_ih = torch.zeros_like(bias_ih) if has_bias else None
    grad_b_hh = torch.zeros_like(bias_hh) if has_bias else None

    for t in reversed(range(T)):
        workspace_t = workspaces[t]

        # 当前时刻的 dL/dh_t = grad_output_t + 来自未来时间步的 dh_next
        grad_hy = grad_output[:, t, :] + dh_next    # [B, H]

        # fused backward：返回的是这一时刻的局部梯度
        grad_igates, grad_hgates, grad_hidden_prev, grad_b_ih_t, grad_b_hh_t = \
            fused_gru.export_fused_gru_cell_backward(
                grad_hy, workspace_t, has_bias
            )

        # 1) 通过 igates = x_t @ W_ih^T 反推到 x_t：
        #    dL/dx_t = dL/d(igates) @ W_ih
        dx[:, t, :] = grad_igates @ weight_ih       # [B, I]

        # 2) 对 h_{t-1} 的梯度有两部分：
        #    - 来自 h_t 对 h_{t-1} 的显式依赖（grad_hidden_prev）
        #    - 来自 hgates = h_{t-1} @ W_hh^T 这一条：grad_hgates @ W_hh
        dh_prev_from_hgates = grad_hgates @ weight_hh   # [B, H]
        dh_next = grad_hidden_prev + dh_prev_from_hgates

        # 3) 累积权重 / 偏置梯度
        x_t = xs[t]                  # [B, I]
        h_prev_t = h_prevs[t]        # [B, H]

        # igates = x_t @ W_ih^T => dL/dW_ih += grad_igates^T @ x_t
        grad_w_ih = grad_w_ih + grad_igates.transpose(0, 1) @ x_t   # [3H, I]

        # hgates = h_prev_t @ W_hh^T => dL/dW_hh += grad_hgates^T @ h_prev_t
        grad_w_hh = grad_w_hh + grad_hgates.transpose(0, 1) @ h_prev_t  # [3H, H]

        if has_bias:
            grad_b_ih = grad_b_ih + grad_b_ih_t      # [3H]
            grad_b_hh = grad_b_hh + grad_b_hh_t      # [3H]

    return dx, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh



with torch.no_grad():
    dx, gi, gh, bi, bh = gru_backward_input_fused(
        grad_output=model.output_grad,           # [B, T, H]
        x=model.last_input,                     # [B, T, I]
        h0=model.last_h0,                       # [1, B, H]
        weight_ih=model.weights["weight_ih_l0"],
        weight_hh=model.weights["weight_hh_l0"],
        bias_ih=model.biases["bias_ih_l0"],
        bias_hh=model.biases["bias_hh_l0"],
    )

print("max dx =", (dx - x.grad).abs().max().item())
print("max dw_i =", (gi - model.gru.weight_ih_l0.grad).abs().max().item())
print("max dw_h =", (gh - model.gru.weight_hh_l0.grad).abs().max().item())
print("max db_i =", (bi - model.gru.bias_ih_l0.grad).abs().max().item())
print("max db_h =", (bh - model.gru.bias_hh_l0.grad).abs().max().item())
