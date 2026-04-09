import time
import numpy as np
import cupy as cp

from Benchmark.pybind11.afp_multi import AFP

def time_cpu_scalar(afp: AFP, x_np: np.ndarray, iters: int = 1):
    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = np.array([afp.convert(float(v), True) for v in x_np], dtype=np.float32)
    t1 = time.perf_counter()
    return out, (t1 - t0) / iters


def time_gpu_kernel(afp: AFP, x_cp: cp.ndarray, warmup: int = 10, iters: int = 100):
    # warmup
    for _ in range(warmup):
        y = afp.convert_cuda(x_cp, return_extra=False)
    cp.cuda.runtime.deviceSynchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(iters):
        y = afp.convert_cuda(x_cp, return_extra=False)
    end.record()
    end.synchronize()

    ms = cp.cuda.get_elapsed_time(start, end) / iters  # per-iter ms
    return y, ms


def main():
    # ====== params (adjust to yours) ======
    M, S, N = 4, 2, 5
    group_up = True
    mask_bits = 0          # try 0
    max_pairs = 8

    afp = AFP(M, S, N, group_up=group_up, mask_bits=mask_bits)

    # ====== data ======
    n = 1_000_000
    x_np = (np.random.uniform(-1.2, 1.2, size=n)).astype(np.float32)
    x_cp = cp.asarray(x_np)

    print(f"Config: M={M}, S={S}, N={N}, group_up={group_up}, mask_bits={mask_bits}")
    print(f"Input: n={n}")

    # ====== GPU perf ======
    y_cp, gpu_ms = time_gpu_kernel(afp, x_cp, warmup=10, iters=200)
    gpu_s = gpu_ms / 1000.0
    gpu_throughput = n / gpu_s / 1e6  # M elems/s

    # ====== correctness check ======
    # CPU scalar is very slow for 1e6; we sample for correctness
    sample = 20000
    idx = np.random.choice(n, size=sample, replace=False)
    x_sample = x_np[idx]

    y_cpu_sample, cpu_sec = time_cpu_scalar(afp, x_sample, iters=1)
    y_gpu_sample = cp.asnumpy(y_cp[idx])

    diff = y_gpu_sample.astype(np.float64) - y_cpu_sample.astype(np.float64)
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    max_rel = np.max(np.abs(diff) / (np.abs(y_cpu_sample) + 1e-12))

    print("\n=== Correctness (sampled) ===")
    print(f"sample size: {sample}")
    print(f"max_abs_error : {max_abs:.6e}")
    print(f"mean_abs_error: {mean_abs:.6e}")
    print(f"max_rel_error : {max_rel:.6e}")

    max_idx = np.argmax(np.abs(diff))
    print("\n=== Worst Case Detail ===")
    print(f"sample index        : {max_idx}")
    print(f"input value         : {x_sample[max_idx]:+.10f}")
    print(f"cpu output          : {y_cpu_sample[max_idx]:+.10f}")
    print(f"gpu output          : {y_gpu_sample[max_idx]:+.10f}")
    print(f"abs error           : {abs(diff[max_idx]):.10e}")
    print(f"rel error           : {abs(diff[max_idx]) / (abs(y_cpu_sample[max_idx]) + 1e-12):.10e}")

    print("\n=== Performance ===")
    print(f"GPU: {gpu_ms:.3f} ms / call  |  throughput: {gpu_throughput:.2f} M elems/s")

    # optional: show CPU scalar speed on small sample
    cpu_throughput = sample / cpu_sec / 1e6
    print(f"CPU(scalar, sample only): {cpu_sec*1e3:.3f} ms / {sample} elems  |  {cpu_throughput:.4f} M elems/s")

    # print a few values
    print("\n=== Preview ===")
    for i in range(5):
        vi = float(x_sample[i])
        print(f"x={vi:+.6f} | cpu={y_cpu_sample[i]:+.6f} | gpu={y_gpu_sample[i]:+.6f} | diff={diff[i]:+.3e}")

if __name__ == "__main__":
    main()