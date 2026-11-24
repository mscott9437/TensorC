#include "tensor.h"

#include <math.h>
#include <stdint.h>

/* ---------------------------------------------------------
 * Internal validation
 * --------------------------------------------------------- */

static TensorStatus tensor_validate_metadata(const Tensor *t) {
  if (!t)
    return TENSOR_ERROR_INVALID_ARGUMENT;

  if (t->ndim > TENSOR_MAX_DIMS)
    return TENSOR_ERROR_INVALID_ARGUMENT;

  /*
   * A zero-dimensional tensor represents a scalar.
   * Therefore, its expected size is 1.
   */
  size_t expected_size = 1;

  for (size_t i = 0; i < t->ndim; ++i) {
    size_t dim = t->shape[i];

    if (expected_size != 0 && dim > SIZE_MAX / expected_size)
      return TENSOR_ERROR_INVALID_ARGUMENT;

    expected_size *= dim;
  }

  if (t->size != expected_size)
    return TENSOR_ERROR_SIZE_MISMATCH;

  return TENSOR_SUCCESS;
}

TensorStatus tensor_validate(const Tensor *t) {
  TensorStatus status = tensor_validate_metadata(t);

  if (status != TENSOR_SUCCESS)
    return status;

  if (t->size > 0 && !t->data)
    return TENSOR_ERROR_INVALID_ARGUMENT;

  return TENSOR_SUCCESS;
}

int tensor_same_shape(const Tensor *a, const Tensor *b) {
  if (tensor_validate_metadata(a) != TENSOR_SUCCESS ||
      tensor_validate_metadata(b) != TENSOR_SUCCESS)
    return 0;

  if (a->ndim != b->ndim)
    return 0;

  for (size_t i = 0; i < a->ndim; ++i) {
    if (a->shape[i] != b->shape[i])
      return 0;
  }

  return 1;
}

static TensorStatus tensor_validate_binary(const Tensor *a, const Tensor *b,
                                           Tensor *out) {
  TensorStatus status;

  status = tensor_validate(a);
  if (status != TENSOR_SUCCESS)
    return status;

  status = tensor_validate(b);
  if (status != TENSOR_SUCCESS)
    return status;

  status = tensor_validate(out);
  if (status != TENSOR_SUCCESS)
    return status;

  if (!tensor_same_shape(a, b))
    return TENSOR_ERROR_SHAPE_MISMATCH;

  if (!tensor_same_shape(a, out))
    return TENSOR_ERROR_SHAPE_MISMATCH;

  return TENSOR_SUCCESS;
}

static TensorStatus tensor_validate_unary(const Tensor *a, Tensor *out) {
  TensorStatus status;

  status = tensor_validate(a);
  if (status != TENSOR_SUCCESS)
    return status;

  status = tensor_validate(out);
  if (status != TENSOR_SUCCESS)
    return status;

  if (!tensor_same_shape(a, out))
    return TENSOR_ERROR_SHAPE_MISMATCH;

  return TENSOR_SUCCESS;
}

/* ---------------------------------------------------------
 * Status messages
 * --------------------------------------------------------- */

const char *tensor_status_string(TensorStatus status) {
  switch (status) {
  case TENSOR_SUCCESS:
    return "Success";

  case TENSOR_ERROR_INVALID_ARGUMENT:
    return "Invalid argument";

  case TENSOR_ERROR_SHAPE_MISMATCH:
    return "Tensor shape mismatch";

  case TENSOR_ERROR_SIZE_MISMATCH:
    return "Tensor size does not match shape";

  case TENSOR_ERROR_EMPTY_TENSOR:
    return "Operation requires a non-empty tensor";

  default:
    return "Unknown tensor error";
  }
}

/* ---------------------------------------------------------
 * Elementwise operations
 * --------------------------------------------------------- */

TensorStatus tensor_add(const Tensor *a, const Tensor *b, Tensor *out) {
  TensorStatus status = tensor_validate_binary(a, b, out);

  if (status != TENSOR_SUCCESS)
    return status;

  for (size_t i = 0; i < a->size; ++i)
    out->data[i] = a->data[i] + b->data[i];

  return TENSOR_SUCCESS;
}

TensorStatus tensor_sub(const Tensor *a, const Tensor *b, Tensor *out) {
  TensorStatus status = tensor_validate_binary(a, b, out);

  if (status != TENSOR_SUCCESS)
    return status;

  for (size_t i = 0; i < a->size; ++i)
    out->data[i] = a->data[i] - b->data[i];

  return TENSOR_SUCCESS;
}

TensorStatus tensor_mul(const Tensor *a, const Tensor *b, Tensor *out) {
  TensorStatus status = tensor_validate_binary(a, b, out);

  if (status != TENSOR_SUCCESS)
    return status;

  for (size_t i = 0; i < a->size; ++i)
    out->data[i] = a->data[i] * b->data[i];

  return TENSOR_SUCCESS;
}

TensorStatus tensor_div(const Tensor *a, const Tensor *b, Tensor *out) {
  TensorStatus status = tensor_validate_binary(a, b, out);

  if (status != TENSOR_SUCCESS)
    return status;

  for (size_t i = 0; i < a->size; ++i)
    out->data[i] = a->data[i] / b->data[i];

  return TENSOR_SUCCESS;
}

/* ---------------------------------------------------------
 * Scalar operations
 * --------------------------------------------------------- */

TensorStatus tensor_scale(const Tensor *a, float scalar, Tensor *out) {
  TensorStatus status = tensor_validate_unary(a, out);

  if (status != TENSOR_SUCCESS)
    return status;

  for (size_t i = 0; i < a->size; ++i)
    out->data[i] = a->data[i] * scalar;

  return TENSOR_SUCCESS;
}

TensorStatus tensor_neg(const Tensor *a, Tensor *out) {
  TensorStatus status = tensor_validate_unary(a, out);

  if (status != TENSOR_SUCCESS)
    return status;

  for (size_t i = 0; i < a->size; ++i)
    out->data[i] = -a->data[i];

  return TENSOR_SUCCESS;
}

TensorStatus tensor_abs(const Tensor *a, Tensor *out) {
  TensorStatus status = tensor_validate_unary(a, out);

  if (status != TENSOR_SUCCESS)
    return status;

  for (size_t i = 0; i < a->size; ++i)
    out->data[i] = fabsf(a->data[i]);

  return TENSOR_SUCCESS;
}

/* ---------------------------------------------------------
 * Reduction operations
 * --------------------------------------------------------- */

TensorStatus tensor_dot(const Tensor *a, const Tensor *b, float *result) {
  if (!result)
    return TENSOR_ERROR_INVALID_ARGUMENT;

  TensorStatus status = tensor_validate(a);
  if (status != TENSOR_SUCCESS)
    return status;

  status = tensor_validate(b);
  if (status != TENSOR_SUCCESS)
    return status;

  if (a->size != b->size)
    return TENSOR_ERROR_SIZE_MISMATCH;

  float sum = 0.0f;

  for (size_t i = 0; i < a->size; ++i)
    sum += a->data[i] * b->data[i];

  *result = sum;

  return TENSOR_SUCCESS;
}

TensorStatus tensor_sum(const Tensor *a, float *result) {
  if (!result)
    return TENSOR_ERROR_INVALID_ARGUMENT;

  TensorStatus status = tensor_validate(a);

  if (status != TENSOR_SUCCESS)
    return status;

  float sum = 0.0f;

  for (size_t i = 0; i < a->size; ++i)
    sum += a->data[i];

  *result = sum;

  return TENSOR_SUCCESS;
}

TensorStatus tensor_min(const Tensor *a, float *result) {
  if (!result)
    return TENSOR_ERROR_INVALID_ARGUMENT;

  TensorStatus status = tensor_validate(a);

  if (status != TENSOR_SUCCESS)
    return status;

  if (a->size == 0)
    return TENSOR_ERROR_EMPTY_TENSOR;

  float min = a->data[0];

  for (size_t i = 1; i < a->size; ++i) {
    if (a->data[i] < min)
      min = a->data[i];
  }

  *result = min;

  return TENSOR_SUCCESS;
}

TensorStatus tensor_max(const Tensor *a, float *result) {
  if (!result)
    return TENSOR_ERROR_INVALID_ARGUMENT;

  TensorStatus status = tensor_validate(a);

  if (status != TENSOR_SUCCESS)
    return status;

  if (a->size == 0)
    return TENSOR_ERROR_EMPTY_TENSOR;

  float max = a->data[0];

  for (size_t i = 1; i < a->size; ++i) {
    if (a->data[i] > max)
      max = a->data[i];
  }

  *result = max;

  return TENSOR_SUCCESS;
}

/* ---------------------------------------------------------
 * Utility operations
 * --------------------------------------------------------- */

TensorStatus tensor_fill(Tensor *t, float value) {
  TensorStatus status = tensor_validate(t);

  if (status != TENSOR_SUCCESS)
    return status;

  for (size_t i = 0; i < t->size; ++i)
    t->data[i] = value;

  return TENSOR_SUCCESS;
}
