#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_failed = 0;

#define TEST_CHECK(condition, description)                                     \
  do {                                                                         \
    ++tests_run;                                                               \
    if (condition) {                                                           \
      printf("[PASS] %s\n", description);                                      \
    } else {                                                                   \
      ++tests_failed;                                                          \
      printf("[FAIL] %s (line %d)\n", description, __LINE__);                  \
    }                                                                          \
  } while (0)

#define TEST_STATUS(expression, expected, description)                         \
  do {                                                                         \
    TensorStatus actual_status = (expression);                                 \
    TEST_CHECK(actual_status == (expected), description);                      \
    if (actual_status != (expected)) {                                         \
      printf("       expected: %s; got: %s\n", tensor_status_string(expected), \
             tensor_status_string(actual_status));                             \
    }                                                                          \
  } while (0)

static int nearly_equal(float actual, float expected) {
  const float epsilon = 1.0e-5f;
  return fabsf(actual - expected) <= epsilon * fmaxf(1.0f, fabsf(expected));
}

static int array_equal(const float *actual, const float *expected,
                       size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!nearly_equal(actual[i], expected[i]))
      return 0;
  }

  return 1;
}

static void test_validation(void) {
  printf("\n== Validation ==\n");

  float values[6] = {1, 2, 3, 4, 5, 6};
  size_t shape_2x3[2] = {2, 3};

  Tensor valid = {.data = values, .ndim = 2, .shape = {2, 3}, .size = 6};

  Tensor null_data = {.data = NULL, .ndim = 2, .shape = {2, 3}, .size = 6};

  Tensor wrong_size = {.data = values, .ndim = 2, .shape = {2, 3}, .size = 5};

  Tensor too_many_dims = {.data = values,
                          .ndim = TENSOR_MAX_DIMS + 1,
                          .shape = {1, 1, 1, 1},
                          .size = 1};

  Tensor empty = {.data = NULL, .ndim = 1, .shape = {0}, .size = 0};

  Tensor scalar = {.data = values, .ndim = 0, .shape = {0}, .size = 1};

  (void)shape_2x3;

  TEST_STATUS(tensor_validate(&valid), TENSOR_SUCCESS,
              "valid 2D tensor is accepted");
  TEST_STATUS(tensor_validate(NULL), TENSOR_ERROR_INVALID_ARGUMENT,
              "NULL tensor is rejected");
  TEST_STATUS(tensor_validate(&null_data), TENSOR_ERROR_INVALID_ARGUMENT,
              "non-empty tensor with NULL data is rejected");
  TEST_STATUS(tensor_validate(&wrong_size), TENSOR_ERROR_SIZE_MISMATCH,
              "size inconsistent with shape is rejected");
  TEST_STATUS(tensor_validate(&too_many_dims), TENSOR_ERROR_INVALID_ARGUMENT,
              "dimension count above TENSOR_MAX_DIMS is rejected");
  TEST_STATUS(tensor_validate(&empty), TENSOR_SUCCESS,
              "zero-element tensor with NULL data is accepted");
  TEST_STATUS(tensor_validate(&scalar), TENSOR_SUCCESS,
              "zero-dimensional scalar with one element is accepted");

  TEST_CHECK(tensor_same_shape(&valid, &valid),
             "tensor_same_shape accepts matching tensors");
  TEST_CHECK(!tensor_same_shape(&valid, NULL),
             "tensor_same_shape rejects NULL tensor");
  TEST_CHECK(!tensor_same_shape(&valid, &wrong_size),
             "tensor_same_shape rejects invalid metadata");

  TEST_CHECK(strcmp(tensor_status_string(TENSOR_SUCCESS), "Success") == 0,
             "success status has a readable message");
  TEST_CHECK(tensor_status_string((TensorStatus)999) != NULL,
             "unknown status has a fallback message");
}

static void test_elementwise_operations(void) {
  printf("\n== Elementwise operations ==\n");

  float a_data[4] = {1, 2, 3, 4};
  float b_data[4] = {5, 6, 7, 8};
  float out_data[4] = {0};
  const float add_expected[4] = {6, 8, 10, 12};
  const float sub_expected[4] = {-4, -4, -4, -4};
  const float mul_expected[4] = {5, 12, 21, 32};
  const float div_expected[4] = {0.2f, 1.0f / 3.0f, 3.0f / 7.0f, 0.5f};

  Tensor a = {.data = a_data, .ndim = 1, .shape = {4}, .size = 4};
  Tensor b = {.data = b_data, .ndim = 1, .shape = {4}, .size = 4};
  Tensor out = {.data = out_data, .ndim = 1, .shape = {4}, .size = 4};

  TEST_STATUS(tensor_add(&a, &b, &out), TENSOR_SUCCESS,
              "tensor_add returns success");
  TEST_CHECK(array_equal(out_data, add_expected, 4),
             "tensor_add computes elementwise sums");

  TEST_STATUS(tensor_sub(&a, &b, &out), TENSOR_SUCCESS,
              "tensor_sub returns success");
  TEST_CHECK(array_equal(out_data, sub_expected, 4),
             "tensor_sub computes elementwise differences");

  TEST_STATUS(tensor_mul(&a, &b, &out), TENSOR_SUCCESS,
              "tensor_mul returns success");
  TEST_CHECK(array_equal(out_data, mul_expected, 4),
             "tensor_mul computes elementwise products");

  TEST_STATUS(tensor_div(&a, &b, &out), TENSOR_SUCCESS,
              "tensor_div returns success");
  TEST_CHECK(array_equal(out_data, div_expected, 4),
             "tensor_div computes elementwise quotients");

  TEST_STATUS(tensor_add(NULL, &b, &out), TENSOR_ERROR_INVALID_ARGUMENT,
              "tensor_add rejects NULL input");
  TEST_STATUS(tensor_add(&a, &b, NULL), TENSOR_ERROR_INVALID_ARGUMENT,
              "tensor_add rejects NULL output");

  {
    float other_data[2] = {10, 20};
    Tensor other = {.data = other_data, .ndim = 1, .shape = {2}, .size = 2};
    TEST_STATUS(tensor_add(&a, &other, &out), TENSOR_ERROR_SHAPE_MISMATCH,
                "tensor_add rejects mismatched input shapes");
    TEST_STATUS(tensor_mul(&a, &b, &other), TENSOR_ERROR_SHAPE_MISMATCH,
                "tensor_mul rejects mismatched output shape");
  }

  {
    Tensor bad_input = {.data = NULL, .ndim = 1, .shape = {4}, .size = 4};
    TEST_STATUS(tensor_mul(&bad_input, &b, &out), TENSOR_ERROR_INVALID_ARGUMENT,
                "tensor_mul rejects input with NULL data");
  }

  /* Exact in-place output aliasing is supported by the implementation. */
  {
    float in_place_data[4] = {1, 2, 3, 4};
    const float expected[4] = {6, 8, 10, 12};
    Tensor in_place = {
        .data = in_place_data, .ndim = 1, .shape = {4}, .size = 4};

    TEST_STATUS(tensor_add(&in_place, &b, &in_place), TENSOR_SUCCESS,
                "tensor_add supports output aliasing its first input");
    TEST_CHECK(array_equal(in_place_data, expected, 4),
               "in-place tensor_add produces correct values");
  }

  {
    float in_place_data[4] = {2, 3, 4, 5};
    const float expected[4] = {4, 9, 16, 25};
    Tensor in_place = {
        .data = in_place_data, .ndim = 1, .shape = {4}, .size = 4};

    TEST_STATUS(tensor_mul(&in_place, &in_place, &in_place), TENSOR_SUCCESS,
                "tensor_mul supports identical input/output tensor");
    TEST_CHECK(array_equal(in_place_data, expected, 4),
               "in-place tensor_mul produces correct values");
  }
}

static void test_scalar_operations(void) {
  printf("\n== Scalar operations ==\n");

  float input_data[5] = {-2, -1, 0, 1, 2};
  float output_data[5] = {0};
  Tensor input = {.data = input_data, .ndim = 1, .shape = {5}, .size = 5};
  Tensor output = {.data = output_data, .ndim = 1, .shape = {5}, .size = 5};

  const float scale_expected[5] = {-6, -3, 0, 3, 6};
  const float neg_expected[5] = {2, 1, 0, -1, -2};
  const float abs_expected[5] = {2, 1, 0, 1, 2};

  TEST_STATUS(tensor_scale(&input, 3.0f, &output), TENSOR_SUCCESS,
              "tensor_scale returns success");
  TEST_CHECK(array_equal(output_data, scale_expected, 5),
             "tensor_scale multiplies every element by scalar");

  TEST_STATUS(tensor_neg(&input, &output), TENSOR_SUCCESS,
              "tensor_neg returns success");
  TEST_CHECK(array_equal(output_data, neg_expected, 5),
             "tensor_neg negates every element");

  TEST_STATUS(tensor_abs(&input, &output), TENSOR_SUCCESS,
              "tensor_abs returns success");
  TEST_CHECK(array_equal(output_data, abs_expected, 5),
             "tensor_abs computes absolute values");

  TEST_STATUS(tensor_scale(NULL, 2.0f, &output), TENSOR_ERROR_INVALID_ARGUMENT,
              "tensor_scale rejects NULL input");
  TEST_STATUS(tensor_neg(&input, NULL), TENSOR_ERROR_INVALID_ARGUMENT,
              "tensor_neg rejects NULL output");

  {
    float wrong_output_data[2] = {0};
    Tensor wrong_output = {
        .data = wrong_output_data, .ndim = 1, .shape = {2}, .size = 2};

    TEST_STATUS(tensor_abs(&input, &wrong_output), TENSOR_ERROR_SHAPE_MISMATCH,
                "tensor_abs rejects mismatched output shape");
  }

  {
    float in_place_data[3] = {-1, 0, 2};
    const float expected[3] = {1, 0, 2};
    Tensor in_place = {
        .data = in_place_data, .ndim = 1, .shape = {3}, .size = 3};

    TEST_STATUS(tensor_abs(&in_place, &in_place), TENSOR_SUCCESS,
                "tensor_abs supports in-place operation");
    TEST_CHECK(array_equal(in_place_data, expected, 3),
               "in-place tensor_abs produces correct values");
  }
}

static void test_reductions(void) {
  printf("\n== Reduction operations ==\n");

  float a_data[4] = {1, 2, 3, 4};
  float b_data[4] = {5, 6, 7, 8};
  Tensor a = {.data = a_data, .ndim = 1, .shape = {4}, .size = 4};
  Tensor b = {.data = b_data, .ndim = 1, .shape = {4}, .size = 4};

  float result = -123.0f;

  TEST_STATUS(tensor_dot(&a, &b, &result), TENSOR_SUCCESS,
              "tensor_dot returns success");
  TEST_CHECK(nearly_equal(result, 70.0f),
             "tensor_dot computes flattened dot product");

  TEST_STATUS(tensor_sum(&a, &result), TENSOR_SUCCESS,
              "tensor_sum returns success");
  TEST_CHECK(nearly_equal(result, 10.0f), "tensor_sum computes element sum");

  TEST_STATUS(tensor_min(&a, &result), TENSOR_SUCCESS,
              "tensor_min returns success");
  TEST_CHECK(nearly_equal(result, 1.0f), "tensor_min finds minimum element");

  TEST_STATUS(tensor_max(&a, &result), TENSOR_SUCCESS,
              "tensor_max returns success");
  TEST_CHECK(nearly_equal(result, 4.0f), "tensor_max finds maximum element");

  TEST_STATUS(tensor_dot(&a, &b, NULL), TENSOR_ERROR_INVALID_ARGUMENT,
              "tensor_dot rejects NULL result pointer");
  TEST_STATUS(tensor_sum(&a, NULL), TENSOR_ERROR_INVALID_ARGUMENT,
              "tensor_sum rejects NULL result pointer");
  TEST_STATUS(tensor_min(&a, NULL), TENSOR_ERROR_INVALID_ARGUMENT,
              "tensor_min rejects NULL result pointer");
  TEST_STATUS(tensor_max(&a, NULL), TENSOR_ERROR_INVALID_ARGUMENT,
              "tensor_max rejects NULL result pointer");

  {
    float other_data[3] = {1, 2, 3};
    Tensor other = {.data = other_data, .ndim = 1, .shape = {3}, .size = 3};

    TEST_STATUS(tensor_dot(&a, &other, &result), TENSOR_ERROR_SIZE_MISMATCH,
                "tensor_dot rejects different element counts");
  }

  {
    Tensor empty = {.data = NULL, .ndim = 1, .shape = {0}, .size = 0};

    TEST_STATUS(tensor_sum(&empty, &result), TENSOR_SUCCESS,
                "tensor_sum accepts empty tensor");
    TEST_CHECK(nearly_equal(result, 0.0f),
               "tensor_sum of empty tensor is zero");

    TEST_STATUS(tensor_dot(&empty, &empty, &result), TENSOR_SUCCESS,
                "tensor_dot accepts two empty tensors");
    TEST_CHECK(nearly_equal(result, 0.0f),
               "tensor_dot of empty tensors is zero");

    TEST_STATUS(tensor_min(&empty, &result), TENSOR_ERROR_EMPTY_TENSOR,
                "tensor_min rejects empty tensor");
    TEST_STATUS(tensor_max(&empty, &result), TENSOR_ERROR_EMPTY_TENSOR,
                "tensor_max rejects empty tensor");
  }

  {
    float scalar_data[1] = {42.0f};
    Tensor scalar = {.data = scalar_data, .ndim = 0, .shape = {0}, .size = 1};

    TEST_STATUS(tensor_sum(&scalar, &result), TENSOR_SUCCESS,
                "tensor_sum accepts zero-dimensional scalar");
    TEST_CHECK(nearly_equal(result, 42.0f),
               "sum of scalar tensor equals scalar value");
  }
}

static void test_utility_operations(void) {
  printf("\n== Utility operations ==\n");

  float data[4] = {1, 2, 3, 4};
  const float expected[4] = {-7, -7, -7, -7};
  Tensor tensor = {.data = data, .ndim = 1, .shape = {4}, .size = 4};

  TEST_STATUS(tensor_fill(&tensor, -7.0f), TENSOR_SUCCESS,
              "tensor_fill returns success");
  TEST_CHECK(array_equal(data, expected, 4),
             "tensor_fill writes value to every element");

  TEST_STATUS(tensor_fill(NULL, 1.0f), TENSOR_ERROR_INVALID_ARGUMENT,
              "tensor_fill rejects NULL tensor");

  {
    Tensor empty = {.data = NULL, .ndim = 1, .shape = {0}, .size = 0};

    TEST_STATUS(tensor_fill(&empty, 3.0f), TENSOR_SUCCESS,
                "tensor_fill accepts empty tensor");
  }
}

static void test_multidimensional_tensors(void) {
  printf("\n== Multidimensional tensors ==\n");

  float a_data[6] = {1, 2, 3, 4, 5, 6};
  float b_data[6] = {10, 20, 30, 40, 50, 60};
  float out_data[6] = {0};
  const float expected[6] = {11, 22, 33, 44, 55, 66};

  Tensor a = {.data = a_data, .ndim = 2, .shape = {2, 3}, .size = 6};
  Tensor b = {.data = b_data, .ndim = 2, .shape = {2, 3}, .size = 6};
  Tensor out = {.data = out_data, .ndim = 2, .shape = {2, 3}, .size = 6};

  TEST_STATUS(tensor_add(&a, &b, &out), TENSOR_SUCCESS,
              "tensor_add accepts matching 2D tensors");
  TEST_CHECK(array_equal(out_data, expected, 6),
             "2D tensor_add processes contiguous elements correctly");
}

int main(void) {
  printf("========================================\n");
  printf(" Tensor Library Unit Tests\n");
  printf("========================================\n");

  test_validation();
  test_elementwise_operations();
  test_scalar_operations();
  test_reductions();
  test_utility_operations();
  test_multidimensional_tensors();

  printf("\n========================================\n");
  printf(" Tests run:    %d\n", tests_run);
  printf(" Tests passed: %d\n", tests_run - tests_failed);
  printf(" Tests failed: %d\n", tests_failed);
  printf("========================================\n");

  if (tests_failed != 0) {
    printf("RESULT: FAIL\n");
    return EXIT_FAILURE;
  }

  printf("RESULT: PASS\n");
  return EXIT_SUCCESS;
}
