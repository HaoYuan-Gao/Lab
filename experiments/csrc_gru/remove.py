#!/usr/bin/env python3
import os
import argparse

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def _rm_file(path: str) -> bool:
    """删除单个文件，返回是否成功。"""
    try:
        os.remove(path)
        return True
    except Exception:
        return False

def fast_rm_tree(root: str, workers: int | None = None, chunksize: int = 256):
    if not os.path.exists(root):
        print(f"[WARN] {root} 不存在，跳过")
        return

    # root 是文件的话，直接删掉就行
    if os.path.isfile(root) or os.path.islink(root):
        try:
            os.remove(root)
            print(f"[INFO] 已删除文件: {root}")
        except Exception as e:
            print(f"[ERROR] 无法删除文件 {root}: {e}")
        return

    workers = workers or cpu_count()
    print(f"[INFO] 多进程删除目录: {root}")
    print(f"[INFO] 使用进程数: {workers}")

    # 收集所有文件路径
    file_paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for f in filenames:
            file_paths.append(os.path.join(dirpath, f))

    total_files = len(file_paths)
    print(f"[INFO] 文件数量: {total_files}")

    # 多进程删除文件（带进度条）
    if total_files > 0:
        with Pool(processes=workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(_rm_file, file_paths, chunksize=chunksize),
                total=total_files,
                desc="Deleting files",
            ):
                pass
    print("[INFO] 所有文件已删除，开始删除空目录...")

    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for d in dirnames:
            full = os.path.join(dirpath, d)
            try:
                os.rmdir(full)
            except Exception:
                pass

    try:
        os.rmdir(root)
        print(f"[INFO] 根目录已删除: {root}")
    except Exception:
        print(f"[WARN] 根目录 {root} 非空或无法删除")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast multi-process rm -rf with progress bar"
    )
    parser.add_argument(
        "target",
        type=str,
        help="Directory to remove"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=cpu_count(),
        help="Number of worker processes (default: CPU cores)"
    )
    parser.add_argument(
        "-c",
        "--chunksize",
        type=int,
        default=256,
        help="Chunk size for multiprocessing imap_unordered"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fast_rm_tree(args.target, workers=args.jobs, chunksize=args.chunksize)

