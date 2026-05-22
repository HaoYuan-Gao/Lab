#ifndef GTENSOR_RUNTIME_H
#define GTENSOR_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GTENSOR_MAX_DIMS 5
#define GTENSOR_MAX_RESULTS 8

/*
 * Opaque handle.
 * Python/cffi only knows GTensor* exists, but does not know internal fields.
 */
typedef struct GTensor GTensor;

typedef struct {
    GTensor* tensors[GTENSOR_MAX_RESULTS];
    int64_t size;
} NoiseResult;

GTensor* gtensor_create_1d(int64_t length, int64_t dtype);
GTensor* gtensor_create_2d(int64_t rows, int64_t cols, int64_t dtype);

void gtensor_retain(GTensor* tensor);
void gtensor_release(GTensor* tensor);

int64_t gtensor_ndim(GTensor* tensor);
int64_t gtensor_shape(GTensor* tensor, int64_t dim);
int64_t gtensor_dtype(GTensor* tensor);
int64_t gtensor_ref_count(GTensor* tensor);
uintptr_t gtensor_data_addr(GTensor* tensor);

NoiseResult gtensor_noise_forward(GTensor* input, int64_t output_count);
void gtensor_noise_result_release(NoiseResult result);

#ifdef __cplusplus
}
#endif

#endif
