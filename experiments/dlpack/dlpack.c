#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "dlpack.h"

/*
  为了方便管理，把真实需要释放的资源挂到 manager_ctx 上
*/
typedef struct {
    void* data;
    int64_t* shape;
    int64_t* strides;
} MyTensorCtx;

static void my_dlpack_deleter(DLManagedTensor* self) {
    if (self == NULL) return;

    MyTensorCtx* ctx = (MyTensorCtx*)self->manager_ctx;
    if (ctx != NULL) {
        printf("[deleter] free data/shape/strides/ctx\n");
        free(ctx->data);
        free(ctx->shape);
        free(ctx->strides);
        free(ctx);
    }

    free(self);
}

static DLManagedTensor* producer_create_tensor_2x3_float32(void) {
    DLManagedTensor* m = (DLManagedTensor*)malloc(sizeof(DLManagedTensor));
    if (!m) return NULL;
    memset(m, 0, sizeof(*m));

    MyTensorCtx* ctx = (MyTensorCtx*)malloc(sizeof(MyTensorCtx));
    if (!ctx) {
        free(m);
        return NULL;
    }
    memset(ctx, 0, sizeof(*ctx));

    // 分配实际数据：2x3 float32
    float* data = (float*)malloc(2 * 3 * sizeof(float));
    int64_t* shape = (int64_t*)malloc(2 * sizeof(int64_t));
    int64_t* strides = (int64_t*)malloc(2 * sizeof(int64_t));

    if (!data || !shape || !strides) {
        free(data);
        free(shape);
        free(strides);
        free(ctx);
        free(m);
        return NULL;
    }

    // 初始化数据
    for (int i = 0; i < 6; ++i) {
        data[i] = (float)(i + 1);   // 1,2,3,4,5,6
    }

    // shape = [2, 3]
    shape[0] = 2;
    shape[1] = 3;

    // contiguous row-major strides（单位：元素个数，不是字节）
    strides[0] = 3;
    strides[1] = 1;

    ctx->data = data;
    ctx->shape = shape;
    ctx->strides = strides;

    // 填充 DLTensor
    m->dl_tensor.data = data;
    m->dl_tensor.device.device_type = kDLCPU;
    m->dl_tensor.device.device_id = 0;
    m->dl_tensor.ndim = 2;
    m->dl_tensor.dtype.code = kDLFloat;
    m->dl_tensor.dtype.bits = 32;
    m->dl_tensor.dtype.lanes = 1;
    m->dl_tensor.shape = shape;
    m->dl_tensor.strides = strides;
    m->dl_tensor.byte_offset = 0;

    // 管理信息
    m->manager_ctx = ctx;
    m->deleter = my_dlpack_deleter;

    return m;
}

static void print_dl_tensor(const DLTensor* t) {
    printf("device_type = %d, device_id = %d\n",
           (int)t->device.device_type, (int)t->device.device_id);
    printf("ndim = %d\n", (int)t->ndim);
    printf("dtype = {code=%u, bits=%u, lanes=%u}\n",
           (unsigned)t->dtype.code,
           (unsigned)t->dtype.bits,
           (unsigned)t->dtype.lanes);
    printf("shape = [");
    for (int i = 0; i < t->ndim; ++i) {
        printf("%lld%s", (long long)t->shape[i], (i + 1 == t->ndim) ? "" : ", ");
    }
    printf("]\n");

    printf("strides = [");
    for (int i = 0; i < t->ndim; ++i) {
        printf("%lld%s", (long long)t->strides[i], (i + 1 == t->ndim) ? "" : ", ");
    }
    printf("]\n");

    printf("byte_offset = %llu\n", (unsigned long long)t->byte_offset);
}

static void consumer_print_and_modify(DLManagedTensor* m) {
    DLTensor* t = &m->dl_tensor;

    print_dl_tensor(t);

    // 这里只处理 CPU + float32 + 2D contiguous 的简单情况
    if (t->device.device_type != kDLCPU) {
        printf("unsupported device\n");
        return;
    }
    if (!(t->dtype.code == kDLFloat && t->dtype.bits == 32 && t->dtype.lanes == 1)) {
        printf("unsupported dtype\n");
        return;
    }
    if (t->ndim != 2) {
        printf("unsupported ndim\n");
        return;
    }

    float* base = (float*)((char*)t->data + t->byte_offset);
    int64_t rows = t->shape[0];
    int64_t cols = t->shape[1];

    printf("tensor values before modify:\n");
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            float v = base[i * t->strides[0] + j * t->strides[1]];
            printf("%6.1f ", v);
        }
        printf("\n");
    }

    // 原地修改一个元素，模拟 consumer 对共享内存的操作
    base[1 * t->strides[0] + 2 * t->strides[1]] = 999.0f;

    printf("tensor values after modify:\n");
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            float v = base[i * t->strides[0] + j * t->strides[1]];
            printf("%6.1f ", v);
        }
        printf("\n");
    }
}

int main(void) {
    DLManagedTensor* m = producer_create_tensor_2x3_float32();
    if (!m) {
        fprintf(stderr, "failed to create tensor\n");
        return 1;
    }

    consumer_print_and_modify(m);

    // consumer 用完后，按协议调用 producer 提供的 deleter
    if (m->deleter) {
        m->deleter(m);
    }

    return 0;
}