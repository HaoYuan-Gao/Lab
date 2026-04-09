import os
import re
import torch
import numpy as np

from multiprocessing import get_context
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from itertools import product

from visualize.curve import plot_heatmap_sparse_with_focus, plot_line_norm
from visualize.load_tensor import _load_single_step, _load_single_file


def get_paths_for_figure(fig_idx):
    # TODO: 你自己的逻辑：返回这一张图要用到的 256 个文件路径
    paths = [...]
    return paths

def plot_curve_job(args):
    try:
        param, rank, load_func, module, thread_num, task_steps = args

        SAVE_PATH = f"./grad/{module}/{rank}"
        os.makedirs(SAVE_PATH, exist_ok=True)

        files = [ f"{param}_{rank}.pt" ]

        load_task = []
        step_indices = np.empty(len(task_steps), dtype=np.int32)

        for idx, (step_id, step_dir) in enumerate(task_steps):
            step_indices[idx] = step_id
            load_task.append((step_id, step_dir, files))

        # ---- 预分配结果（按任务顺序）----
        norms = np.empty(len(load_task), dtype=np.float32)

        # ---- 内层多线程：负责 IO tensor ----
        max_workers = min(thread_num, len(load_task))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_func, task) for task in load_task]

            for fut in as_completed(futures):
                step_id, norm = fut.result()
                norms[step_id] = norm

        plot_line_norm(
            norms=norms,
            step_indices=step_indices,
            save_path=f"{SAVE_PATH}/{param}.png"
        )

        return (param, rank, "OK")
    except Exception as e:
            return (param, rank, e)

def plot_heat_job(args):
    try:
        grads, rank, load_func, module, thread_num, task_steps = args

        SAVE_PATH = f"./grad/{module}/{rank}"
        os.makedirs(SAVE_PATH, exist_ok=True)

        files = [ f"{grads}_{rank}.pt" ]

        load_task = []
        step_indices = np.empty(len(task_steps), dtype=np.int32)

        for idx, (step_id, step_dir) in enumerate(task_steps):
            step_indices[idx] = step_id
            load_task.append((step_id, step_dir, files))

        # ---- 预分配结果（按任务顺序）----
        norms = [None] * len(load_task)

        # ---- 内层多线程：负责 IO tensor ----
        max_workers = min(thread_num, len(load_task))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_func, task) for task in load_task]

            for fut in as_completed(futures):
                step_id, norm = fut.result()
                norms[step_id] = norm

        norms = torch.vstack(norms).transpose(0, 1).contiguous().numpy()
        seq_indices = np.arange(norms.shape[0])

        plot_heatmap_sparse_with_focus(
            norms,
            seq_indices,
            step_indices,
            save_path=f"{SAVE_PATH}/{grads}.png",
            base_xtick_div=20,   # 横轴稀疏程度，数值越小越稀疏
            base_ytick_div=20,   # 纵轴稀疏程度
            grid_line_width=0.4,
            grid_line_alpha=0.5
        )

        return (grads, rank, "OK")
    except Exception as e:
            return (grads, rank, e)


def draw_img_with_mp(task_steps, tasks: list[tuple], load_func, curve_func, module, num_workers=32, num_threads=64):
    ctx = get_context("spawn")

    jobs = [ t + (load_func, module, num_threads, task_steps, ) for t in tasks ]

    # 多进程：每个进程负责画若干张图
    with ctx.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(curve_func, jobs):
            param, rank, status = result
            print(f"Worker Status at {param}-{rank} : {status}")


if __name__ == "__main__":
    moduleList = ["dec_seqs1"] # ["enc_seqs_0", "enc_seqs_1", "neck_seqs0", "neck_seqs1", "dec_seqs1", "dec_seqs0"]
    gradList = ["grad_dh"] # ["x_grad", "h_grad", "grad_dh", "grad_wi", "grad_wh", "grad_bi", "grad_bh", "h"]
    paramList = [ "bias_hh_l0", "bias_ih_l0", "weight_hh_l0", "weight_ih_l0" ]
    
    RANKS = [f"rank{i}" for i in range(8)]

    grads_task = [(g, r) for g, r in product(gradList, RANKS)]
    params_task = [(p, r) for p, r in product(paramList, RANKS)]
    
    START = 0
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
        step_dirs = sorted(step_dirs, key=lambda x: int(x.split("_")[1]))

        # ---- 2. 构造任务列表（带过滤）----
        task_steps = []
        for step_name in step_dirs:
            step_index = int(step_name.split("_")[1])

            # 过滤 step 范围
            if START is not None and step_index < START:
                continue
            if END is not None and step_index > END:
                continue

            step_dir = os.path.join(ROOT_DIR, step_name)
            task_steps.append((step_index, step_dir))

        if not task_steps:
            raise RuntimeError(f"没有找到 step_x 目录")
        
        # draw_img_with_mp(task_steps, params_task, _load_single_file, plot_curve_job, module, num_workers=32)
        draw_img_with_mp(task_steps, grads_task, _load_single_step, plot_heat_job, module, num_workers=32)
