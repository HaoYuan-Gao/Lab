import os

import torch
from torch import nn
from typing import Optional, Tuple

import fused_gru
from util.context import Debug

class FusedGRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, alias: str=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        self.alias = alias

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first)
        # 在加载 state_dict 前，把旧 key 自动改成新 key
        self._register_load_state_dict_pre_hook(self._prehook_remap_old_keys, with_module=True)

    @staticmethod
    def _prehook_remap_old_keys(module, state_dict, prefix, local_metadata,
                               strict, missing_keys, unexpected_keys, error_msgs):
        """
        prefix 例子：'enc_seqs.0.seq_t.'
        我们要做的是：如果 ckpt 里出现 prefix + 'weight_ih_l0'
        就把它搬到 prefix + 'gru.weight_ih_l0'
        """
        # 只处理属于这个模块前缀的 keys，避免误伤别的层
        # 目标 old: prefix + 'weight_ih_l0'   new: prefix + 'gru.weight_ih_l0'
        for base in ("weight_ih_l", "weight_hh_l", "bias_ih_l", "bias_hh_l"):
            layer = 0
            while True:
                old_k = f"{prefix}{base}{layer}"
                new_k = f"{prefix}gru.{base}{layer}"
                if old_k in state_dict and new_k not in state_dict:
                    state_dict[new_k] = state_dict.pop(old_k)
                # 如果这个 layer 不存在就停止（兼容不同 num_layers）
                if (f"{prefix}{base}{layer}" not in state_dict) and (f"{prefix}gru.{base}{layer}" not in state_dict):
                    # 继续探测下一个 layer 是否存在没有意义了，直接 break
                    break
                layer += 1

    def _save_path(self, name):
        save_path = f"./gru_grad/{self.alias}/epoch_{Debug.epoch}/step_{Debug.step}"
        os.makedirs(save_path, exist_ok=True)

        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            name = name + f"_rank{dist.get_rank()}"

        return os.path.join(save_path, f"{name}.pt")
    
    def save_forward_tensor(self, tensor: torch.Tensor, name: str):
        torch.save(tensor.detach().cpu(), self._save_path(name))

    def register_tensor_grad(self, tensor: torch.Tensor, name: str):
        def _hook(grad):
            t: torch.Tensor = grad.detach().cpu()
            if self.batch_first:
                t = t.transpose(0, 1)
            torch.save(t, self._save_path(name + "_grad"))
        tensor.register_hook(_hook)
    
    def save_parameter_snapshots(self):
        for name, p in self.named_parameters():
            torch.save(p.detach().cpu(), self._save_path(name))

    def forward(self, x: torch.Tensor, hx: torch.Tensor=None):
        x.retain_grad()
        # 保存一份引用，方便 backward 后读 .grad
        self.last_x = x

        out, h_n = self.gru(x, hx)

        out.retain_grad()
        self.last_out = out
        self.last_h_n = h_n

        self.save_forward_tensor(x, "x")
        if hx is not None:
            self.save_forward_tensor(hx, "h0")

        self.save_parameter_snapshots()

        self.register_tensor_grad(self.last_x, "x")
        self.register_tensor_grad(self.last_out, "h")

        return out, h_n


class TBPTTGRU(nn.Module):
    """
    Drop-in replacement for nn.GRU that performs TBPTT inside forward:
      - splits sequence into chunks of length K
      - carries hidden across chunks
      - detaches hidden between chunks to truncate gradients
      - concatenates outputs so downstream loss can be computed once

    Usage:
      rnn = TBPTTGRU(input_size, hidden_size, num_layers=..., batch_first=True, K=100)
      y, hT = rnn(x, h0)
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        K: int = 100,
        detach_between_chunks: bool = True,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.K = int(K)
        self.detach_between_chunks = bool(detach_between_chunks)

        # 在加载 state_dict 前，把旧 key 自动改成新 key
        self._register_load_state_dict_pre_hook(self._prehook_remap_old_keys, with_module=True)

    @staticmethod
    def _prehook_remap_old_keys(module, state_dict, prefix, local_metadata,
                               strict, missing_keys, unexpected_keys, error_msgs):
        """
        prefix 例子：'enc_seqs.0.seq_t.'
        我们要做的是：如果 ckpt 里出现 prefix + 'weight_ih_l0'
        就把它搬到 prefix + 'gru.weight_ih_l0'
        """
        # 只处理属于这个模块前缀的 keys，避免误伤别的层
        # 目标 old: prefix + 'weight_ih_l0'   new: prefix + 'gru.weight_ih_l0'
        for base in ("weight_ih_l", "weight_hh_l", "bias_ih_l", "bias_hh_l"):
            layer = 0
            while True:
                old_k = f"{prefix}{base}{layer}"
                new_k = f"{prefix}gru.{base}{layer}"
                if old_k in state_dict and new_k not in state_dict:
                    state_dict[new_k] = state_dict.pop(old_k)
                # 如果这个 layer 不存在就停止（兼容不同 num_layers）
                if (f"{prefix}{base}{layer}" not in state_dict) and (f"{prefix}gru.{base}{layer}" not in state_dict):
                    # 继续探测下一个 layer 是否存在没有意义了，直接 break
                    break
                layer += 1

    @property
    def batch_first(self) -> bool:
        return self.gru.batch_first

    @property
    def hidden_size(self) -> int:
        return self.gru.hidden_size

    @property
    def num_layers(self) -> int:
        return self.gru.num_layers

    @property
    def bidirectional(self) -> bool:
        return self.gru.bidirectional

    def extra_repr(self) -> str:
        return f"K={self.K}, detach_between_chunks={self.detach_between_chunks}, batch_first={self.batch_first}"

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:  [B, T, I] if batch_first else [T, B, I]
        h0: [num_layers * num_directions, B, H] or None
        returns:
          y:  [B, T, H*num_directions] if batch_first else [T, B, ...]
          hT: [num_layers * num_directions, B, H]
        """
        K = self.K
        if K <= 0:
            # behave like a normal GRU
            return self.gru(x, h0)

        bf = self.batch_first
        T = x.shape[1] if bf else x.shape[0]

        h = h0
        outs = []

        for t0 in range(0, T, K):
            if bf:
                x_chunk = x[:, t0:t0+K, :]
            else:
                x_chunk = x[t0:t0+K, :, :]

            out_chunk, h = self.gru(x_chunk, h)
            outs.append(out_chunk)

            # TBPTT: truncate gradients through time between chunks
            if self.detach_between_chunks and h is not None:
                h = h.detach()

        y = torch.cat(outs, dim=1 if bf else 0)
        return y, h



def gru_backward_fused(
    grad_output,   # [T, B, H]  dL/dy_t
    x,             # [B, T, I]  输入
    h0,            # [1, B, H]  初始 hidden
    weight_ih,     # [3H, I]
    weight_hh,     # [3H, H]
    bias_ih,       # [3H] 或 None
    bias_hh,       # [3H] 或 None
    path: str,
    rank: str      # GPU ID
):
    """
    使用 fused_gru.export_fused_gru_cell_forward / backward
    手动做一遍 BPTT, 返回:
      - dx:        [B, T, I]
      - grad_w_ih: [3H, I]
      - grad_w_hh: [3H, H]
      - grad_b_ih: [3H] 或 None
      - grad_b_hh: [3H] 或 None
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    B, T, I = x.shape
    H = weight_hh.shape[1]

    has_bias = (bias_ih is not None) and (bias_hh is not None)

    # ---- 前向：逐时间步调用 fused forward，保存 workspace / x_t / h_prev ----
    h_prev = h0[0] if h0 is not None else x.new_zeros(B, H)  # [B, H]
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

    # torch.save(torch.stack(xs, dim=0), os.path.join(path, f"xs.pt")) # x 经过 norm、clamp 已经到了正常范围
    # torch.save(torch.stack(xgates, dim=0), os.path.join(path, f"xgates_{rank}.pt")) # 中间结果看输入就有判断
    torch.save(torch.stack(h_prevs, dim=0), os.path.join(path, f"h_{rank}.pt"))

    # ---- 反向：从后往前用 fused backward 回传 ----
    dx = torch.zeros_like(x)                        # [B, T, I]
    dh_next = torch.zeros_like(h_prev)              # [B, H]

    grad_w_ih = torch.zeros_like(weight_ih)         # [3H, I]
    grad_w_hh = torch.zeros_like(weight_hh)         # [3H, H]
    grad_b_ih = torch.zeros_like(bias_ih) if has_bias else None
    grad_b_hh = torch.zeros_like(bias_hh) if has_bias else None

    grad_wi = []
    grad_wh = []
    grad_bi = []
    grad_bh = []
    grad_dh = [dh_next]

    for t in reversed(range(T)):
        workspace_t = workspaces[t]

        # 当前时刻的 dL/dh_t = grad_output_t + 来自未来时间步的 dh_next
        # 输出的时候，调整了 B T 方向
        grad_hy = grad_output[t, :, :] + dh_next    # [B, H]

        # fused backward：返回的是这一时刻的局部梯度
        grad_igates, grad_hgates, grad_hidden_prev, grad_b_ih_t, grad_b_hh_t = \
            fused_gru.export_fused_gru_cell_backward(
                grad_hy, workspace_t, has_bias
            )

        #    dL/dx_t = dL/d(igates) @ W_ih
        dx[:, t, :] = grad_igates @ weight_ih       # [B, I]

        # 2) 对 h_{t-1} 的梯度有两部分：
        #    - 来自 h_t 对 h_{t-1} 的显式依赖（grad_hidden_prev）
        #    - 来自 hgates = h_{t-1} @ W_hh^T 这一条：grad_hgates @ W_hh
        dh_prev_from_hgates = grad_hgates @ weight_hh   # [B, H]
        dh_next = grad_hidden_prev + dh_prev_from_hgates
        grad_dh.insert(0, dh_next)

        # 3) 累积权重 / 偏置梯度
        x_t = xs[t]                  # [B, I]
        h_prev_t = h_prevs[t]        # [B, H]

        # igates = x_t @ W_ih^T => dL/dW_ih += grad_igates^T @ x_t
        grad_w_ih = grad_w_ih + grad_igates.transpose(0, 1) @ x_t   # [3H, I]
        grad_wi.insert(0, grad_w_ih)

        # hgates = h_prev_t @ W_hh^T => dL/dW_hh += grad_hgates^T @ h_prev_t
        grad_w_hh = grad_w_hh + grad_hgates.transpose(0, 1) @ h_prev_t  # [3H, H]
        grad_wh.insert(0, grad_w_hh)

        if has_bias:
            grad_b_ih = grad_b_ih + grad_b_ih_t      # [3H]
            grad_bi.insert(0, grad_b_ih)

            grad_b_hh = grad_b_hh + grad_b_hh_t      # [3H]
            grad_bh.insert(0, grad_b_hh)

    torch.save(torch.stack(grad_wi, dim=0), os.path.join(path, f"grad_wi_{rank}.pt"))
    torch.save(torch.stack(grad_wh, dim=0), os.path.join(path, f"grad_wh_{rank}.pt"))
    torch.save(torch.stack(grad_bi, dim=0), os.path.join(path, f"grad_bi_{rank}.pt"))
    torch.save(torch.stack(grad_bh, dim=0), os.path.join(path, f"grad_bh_{rank}.pt"))
    torch.save(torch.stack(grad_dh, dim=0), os.path.join(path, f"grad_dh_{rank}.pt"))

    return dx, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh