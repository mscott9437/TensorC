#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

#define TENSOR_MAX_DIMS 4

typedef struct {
  float *data;
  size_t ndim;
  size_t shape[TENSOR_MAX_DIMS];
  size_t size;
} Tensor;

typedef enum {
  TENSOR_SUCCESS = 0,
  TENSOR_ERROR_INVALID_ARGUMENT,
  TENSOR_ERROR_SHAPE_MISMATCH,
  TENSOR_ERROR_SIZE_MISMATCH,
  TENSOR_ERROR_EMPTY_TENSOR
} TensorStatus;

/* Validation */
int tensor_same_shape(const Tensor *a, const Tensor *b);
TensorStatus tensor_validate(const Tensor *tensor);

const char *tensor_status_string(TensorStatus status);

/* Elementwise operations */
TensorStatus tensor_add(const Tensor *a, const Tensor *b, Tensor *out);

TensorStatus tensor_sub(const Tensor *a, const Tensor *b, Tensor *out);

TensorStatus tensor_mul(const Tensor *a, const Tensor *b, Tensor *out);

TensorStatus tensor_div(const Tensor *a, const Tensor *b, Tensor *out);

/* Scalar operations */
TensorStatus tensor_scale(const Tensor *a, float scalar, Tensor *out);

TensorStatus tensor_neg(const Tensor *a, Tensor *out);

TensorStatus tensor_abs(const Tensor *a, Tensor *out);

/* Reduction operations */
TensorStatus tensor_dot(const Tensor *a, const Tensor *b, float *result);

TensorStatus tensor_sum(const Tensor *a, float *result);

TensorStatus tensor_min(const Tensor *a, float *result);

TensorStatus tensor_max(const Tensor *a, float *result);

/* Utility operations */
TensorStatus tensor_fill(Tensor *tensor, float value);

#endif
