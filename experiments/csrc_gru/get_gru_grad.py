import os
import re

import torch
from multiprocessing import get_context

# worker 全局内存空间
_worker_idx_counter = None
_worker_idx_lock = None

_devices = None
_my_device = None # 当前 worker 绑定的 device

def init_worker(devices, counter, lock):
    """
    在 Pool 里，每个 worker 进程启动时会调用一次这个函数。
    我们在这里：
    1) 分配一个 worker_id
    2) 用 worker_id % len(devices) 选出 device
    3) torch.cuda.set_device(device)
    """
    import torch # 确保在子进程里也 import 了
    global _worker_idx_counter, _worker_idx_lock, _devices, _my_device
    _devices = devices
    _worker_idx_counter = counter
    _worker_idx_lock = lock

    # 下面两个只能在主进程里创建，再通过 initializer_args 传进来，
    # 所以这里假设主进程已经创建好了它们并传进来。
    if _worker_idx_counter is None or _worker_idx_lock is None:
        raise RuntimeError("worker counter / lock not initialized")
    
    with _worker_idx_lock:
        worker_id = _worker_idx_counter.value
        _worker_idx_counter.value += 1

    device = _devices[worker_id % len(_devices)]
    _my_device = device
    torch.cuda.set_device(device)

def load_tensor(path, device):
    return torch.load(path, weights_only=False).to(device)

def worker_job(job):
    global _my_device
    step_index, step_dir, ranks = job
    device = _my_device

    for rank in ranks:
        try:
            # 1. 路径
            x_path      = os.path.join(step_dir, f"x_{rank}.pt")
            h_grad_path = os.path.join(step_dir, f"h_grad_{rank}.pt")
            w_ih_path   = os.path.join(step_dir, f"weight_ih_l0_{rank}.pt")
            w_hh_path   = os.path.join(step_dir, f"weight_hh_l0_{rank}.pt")
            b_ih_path   = os.path.join(step_dir, f"bias_ih_l0_{rank}.pt")
            b_hh_path   = os.path.join(step_dir, f"bias_hh_l0_{rank}.pt")

            if not os.path.exists(x_path) or not os.path.exists(h_grad_path):
                return (step_index, rank, "no_data")
            
            # 2. load 到 CPU 再转到 GPU
            x           = load_tensor(x_path, device)
            grad_h_out  = load_tensor(h_grad_path, device)
            weight_ih   = load_tensor(w_ih_path, device)
            weight_hh   = load_tensor(w_hh_path, device)
            bias_ih     = load_tensor(b_ih_path, device) if os.path.exists(b_ih_path) else None
            bias_hh     = load_tensor(b_hh_path, device) if os.path.exists(b_hh_path) else None

            h0_path = os.path.join(step_dir, "h0.pt")
            h0 = load_tensor(h0_path, device) if os.path.exists(h0_path) else None

            # 3. 调你的 fused backward
            import sys
            sys.path.append(r'./csrc')
            from csrc.gru import gru_backward_fused

            with torch.inference_mode():
                dx, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh = gru_backward_fused(
                    grad_output = grad_h_out,
                    x      = x,
                    h0     = h0,
                    weight_ih  = weight_ih,
                    weight_hh  = weight_hh,
                    bias_ih   = bias_ih,
                    bias_hh   = bias_hh,
                    path    = step_dir,
                    rank    = rank,
                )
            torch.cuda.empty_cache()
        except Exception as e:
            return (step_index, e)
    
    return (step_index, "OK")

def recompute_all_with_mp(tasks, ranks=None, devices=None, num_workers=None):
    """
    tasks: [(step_index, step_dir), ...]
    ranks: ['rank0', 'rank1', ...]
    devices: GPU 列表，例如 [0,1,2,3]
    num_workers: 进程数（可以 >= len(devices)，但显存要自己评估）
    """
    ctx = get_context("spawn")

    # 这两个对象在多个 worker 之间共享，用来分配 worker_id
    idx_counter = ctx.Value('i', 0)     # 共享计数器
    idx_lock  = ctx.Lock()              # 共享锁

    # 把它们塞进全局
    global _worker_idx_counter, _worker_idx_lock
    _worker_idx_counter = idx_counter
    _worker_idx_lock  = idx_lock

    # 构造 job 列表
    jobs = [ (step_index, step_dir, ranks) for step_index, step_dir in tasks ]

    with ctx.Pool(
        processes=min(num_workers, len(jobs)),
        initializer=init_worker,
        initargs=(devices, idx_counter, idx_lock)
    ) as pool:
        from tqdm import tqdm
        for result in tqdm(
            pool.imap_unordered(worker_job, jobs),
            total=len(jobs),
            desc="Recomputing GRU Gradients (mp)",
            ncols=120,
        ):
            step_index, status = result
            if isinstance(status, Exception):
                print(f"Worker failed at step={step_index} : {status}")

if __name__ == "__main__":
    RANKS = [f"rank{i}" for i in range(8)]
    DEVICES = [i for i in range(8)]
    moduleList = [ "dec_seqs1" ] # ["enc_seqs_0", "enc_seqs_1", "neck_seqs0", "neck_seqs1", "dec_seqs1", "dec_seqs0"]

    START = 246
    END = None

    for module in moduleList:
        ROOT_DIR = f"gru_grad/{module}/epoch_286"
        print(f"======= process:{ROOT_DIR} =======")
        
        # ---- 1. 收集 step_x 目录 ----
        step_dirs = [
            d for d in os.listdir(ROOT_DIR)
            if re.match(r"step_(\d+)$", d)
        ]
        if not step_dirs:
            raise RuntimeError(f"ERROR: {ROOT_DIR} 下没找到 step_x 目录")

        # 按 step 编号排序
        step_dirs_sorted = sorted(step_dirs, key=lambda x: int(x.split("_")[1]))

        # ---- 2. 构造任务列表（带过滤）----
        tasks = []
        for step_name in step_dirs_sorted:
            step_index = int(step_name.split("_")[1])

            # 过滤 step 范围
            if START is not None and step_index < START:
                continue
            if END is not None and step_index > END:
                continue

            step_dir = os.path.join(ROOT_DIR, step_name)
            tasks.append((step_index, step_dir))

        if not tasks:
            raise RuntimeError(f"没有找到 step_x 目录")

        recompute_all_with_mp(tasks, ranks=RANKS, devices=DEVICES, num_workers=64)