#include <cuda_runtime.h>

__device__ __forceinline__
long comb(int n, int r) {
    if (r > n || r < 0) return 0;
    if (r == 0 || r == n) return 1;

    r = min(r, n - r);  // C(n, r) == C(n, n-r)
    long result = 1;

    for (int i = 1; i <= r; ++i) {
        result = result * (n - i + 1) / i;
    }
    return result;
}

/**
 * @brief 根据字典序序号 remain 计算组合索引 index[]
 * 
 * 字典序的顺序是固定且 *严格* 的， 可以理解为他的顺序是从小到底依次排序的，当确定了一个 idx 其数据的组合就确定了
 * 以 C(100, 3) 为例，给定一个 idx，先随机从前一部分样本中选取两个，然后从后部分选一个
 * 比如从前 99 个中选两个 step=C(99, 2), 如果 step 小于 idx, 说明该组合还在 idx 之前
 * 如果 step 大于 idx, 说明该组合到达了 idx 处
 * 
 * @param frames 总帧数
 * @param use_frames 选择的帧数
 * @param remain 组合序号
 * @param index 输出组合序号，数组长度为 use_frames
 */
__device__ __host__ 
void get_combination_index(int frames, int use_frames, uint64_t remain, int* index) {
    int x = 1;

    for (int i = 0; i < use_frames - 1; ++i) {
        int r = use_frames - 1 - i;

        uint64_t step = comb(frames - x, r);
        while (step <= remain) {
            remain -= step;
            x++;
            step = comb(frames - x, r);
        }

        index[i] = x - 1;
        x++;
    }

    index[use_frames - 1] = x + remain - 1;
}
