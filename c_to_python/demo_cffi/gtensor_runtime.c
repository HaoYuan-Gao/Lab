#include "gtensor_runtime.h"

#include <stdlib.h>
#include <string.h>

struct GTensor {
    void* data;
    int64_t shape[GTENSOR_MAX_DIMS];
    int64_t ndim;
    int64_t dtype;
    int64_t ref_count;
};

static GTensor* gtensor_alloc_empty(void) {
    GTensor* tensor = (GTensor*)calloc(1, sizeof(GTensor));
    if (!tensor) {
        return NULL;
    }

    tensor->data = NULL;
    tensor->ndim = 0;
    tensor->dtype = 0;
    tensor->ref_count = 1;
    return tensor;
}

GTensor* gtensor_create_1d(int64_t length, int64_t dtype) {
    GTensor* tensor = gtensor_alloc_empty();
    if (!tensor) {
        return NULL;
    }

    tensor->ndim = 1;
    tensor->shape[0] = length;
    tensor->dtype = dtype;
    return tensor;
}

GTensor* gtensor_create_2d(int64_t rows, int64_t cols, int64_t dtype) {
    GTensor* tensor = gtensor_alloc_empty();
    if (!tensor) {
        return NULL;
    }

    tensor->ndim = 2;
    tensor->shape[0] = rows;
    tensor->shape[1] = cols;
    tensor->dtype = dtype;
    return tensor;
}

void gtensor_retain(GTensor* tensor) {
    if (!tensor) {
        return;
    }

    tensor->ref_count += 1;
}

void gtensor_release(GTensor* tensor) {
    if (!tensor) {
        return;
    }

    tensor->ref_count -= 1;
    if (tensor->ref_count <= 0) {
        free(tensor->data);
        free(tensor);
    }
}

int64_t gtensor_ndim(GTensor* tensor) {
    if (!tensor) {
        return 0;
    }

    return tensor->ndim;
}

int64_t gtensor_shape(GTensor* tensor, int64_t dim) {
    if (!tensor || dim < 0 || dim >= tensor->ndim || dim >= GTENSOR_MAX_DIMS) {
        return 0;
    }

    return tensor->shape[dim];
}

int64_t gtensor_dtype(GTensor* tensor) {
    if (!tensor) {
        return -1;
    }

    return tensor->dtype;
}

int64_t gtensor_ref_count(GTensor* tensor) {
    if (!tensor) {
        return 0;
    }

    return tensor->ref_count;
}

uintptr_t gtensor_data_addr(GTensor* tensor) {
    if (!tensor || !tensor->data) {
        return 0;
    }

    return (uintptr_t)tensor->data;
}

NoiseResult gtensor_noise_forward(GTensor* input, int64_t output_count) {
    NoiseResult result;
    memset(&result, 0, sizeof(NoiseResult));

    if (!input) {
        return result;
    }

    if (output_count < 0) {
        output_count = 0;
    }
    if (output_count > GTENSOR_MAX_RESULTS) {
        output_count = GTENSOR_MAX_RESULTS;
    }

    result.size = output_count;

    for (int64_t i = 0; i < output_count; ++i) {
        GTensor* out = gtensor_alloc_empty();
        if (!out) {
            result.size = i;
            return result;
        }

        out->ndim = input->ndim;
        out->dtype = input->dtype;
        for (int64_t d = 0; d < input->ndim && d < GTENSOR_MAX_DIMS; ++d) {
            out->shape[d] = input->shape[d];
        }

        result.tensors[i] = out;
    }

    return result;
}

void gtensor_noise_result_release(NoiseResult result) {
    for (int64_t i = 0; i < result.size && i < GTENSOR_MAX_RESULTS; ++i) {
        gtensor_release(result.tensors[i]);
    }
}
