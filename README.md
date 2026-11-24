# TensorC

A lightweight, dependency-free tensor library written in C, designed for efficient numerical and field processing in embedded applications.

**TensorC** provides a simple API for working with contiguous, multidimensional `float` tensors, with a focus on portability, predictable memory usage, and SIMD acceleration.

## Features

- Up to 4 tensor dimensions (`TENSOR_MAX_DIMS`)
- Contiguous `float32` data storage
- Tensor shape and metadata validation
- Consistent operation status codes
- Elementwise arithmetic:
  - Addition (`tensor_add`)
  - Subtraction (`tensor_sub`)
  - Multiplication (`tensor_mul`)
  - Division (`tensor_div`)
- Scalar operations:
  - Scaling (`tensor_scale`)
  - Negation (`tensor_neg`)
  - Absolute value (`tensor_abs`)
- Reduction operations:
  - Dot product (`tensor_dot`)
  - Sum (`tensor_sum`)
  - Minimum (`tensor_min`)
  - Maximum (`tensor_max`)
- Tensor utilities:
  - Fill (`tensor_fill`)
  - Shape comparison (`tensor_same_shape`)
  - Tensor validation (`tensor_validate`)
- Supports exact in-place operations

## Project Status

The current implementation provides the foundational tensor operations using standard C loops.

Next update will include SIMD acceleration, additional tensor operations, and performance optimizations.

## Building

Compile the library using Clang:

```bash
clang -std=c11 -Wall -Wextra -Wpedantic \
    -c tensor.c -o tensor.o
```

Link against the math library where required:

```bash
clang -std=c11 tensor.c test.c -lm -o test
```

## Testing

TensorC includes a standalone unit test suite with no external testing framework dependencies.

Run the tests:

```bash
./test
```

The test suite currently covers tensor validation, arithmetic, scalar operations, reductions, error handling, empty tensors, in-place operations, and multidimensional tensors.

## Design Goals

- Minimal external dependencies
- Portable C implementation
- Simple, predictable API
- Explicit memory management by the application
- SIMD acceleration without changing the public API
- Suitable for embedding into other applications

## License

MIT
