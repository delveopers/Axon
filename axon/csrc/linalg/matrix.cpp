#include <stdio.h>
#include <stdlib.h>
#include "../cpu/ops_matrix.h"
#include "matrix.h"

Array* inv_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Input array cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim < 2) {
    fprintf(stderr, "Input array must be at least 2D for matrix inverse\n");
    exit(EXIT_FAILURE);
  }

  int last_dim = a->shape[a->ndim - 1];
  int second_last_dim = a->shape[a->ndim - 2];
  if (last_dim != second_last_dim) {
    fprintf(stderr, "Matrix must be square for inverse: %d != %d\n", second_last_dim, last_dim);
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  int* result_shape = (int*)malloc(a->ndim * sizeof(int));
  
  for (size_t i = 0; i < a->ndim; i++) result_shape[i] = a->shape[i];
  size_t result_size = a->size;
  float* out = (float*)malloc(result_size * sizeof(float));
  if (a->ndim == 2) inv_ops(a_float, out, a->shape);
  else batched_inv_ops(a_float, out, a->shape, a->ndim);
  Array* result = create_array(out, a->ndim, result_shape, result_size, a->dtype);
  free(a_float);
  free(out);
  free(result_shape);
  return result;
}

Array* matrix_rank_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Input array cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim < 2) {
    fprintf(stderr, "Input array must be at least 2D for matrix rank\n");
    exit(EXIT_FAILURE);
  }
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  int* result_shape = NULL;
  size_t result_ndim = 0, result_size = 1;
  if (a->ndim == 2) result_ndim = 0, result_size = 1;
  else {
    result_ndim = a->ndim - 2;
    result_shape = (int*)malloc(result_ndim * sizeof(int));
    for (size_t i = 0; i < result_ndim; i++) {
      result_shape[i] = a->shape[i];
      result_size *= result_shape[i];
    }
  }
  float* out = (float*)malloc(result_size * sizeof(float));
  if (a->ndim == 2) matrix_rank_ops(a_float, out, a->shape);
  else batched_matrix_rank_ops(a_float, out, a->shape, a->ndim);
  Array* result = create_array(out, result_ndim, result_shape, result_size, a->dtype);
  free(a_float);
  free(out);
  if (result_shape) free(result_shape);
  return result;
}

Array* solve_array(Array* a, Array* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Input arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim < 2 || b->ndim < 1) {
    fprintf(stderr, "Matrix 'a' must be at least 2D and vector 'b' must be at least 1D\n");
    exit(EXIT_FAILURE);
  }
  
  int a_rows = a->shape[a->ndim - 2], a_cols = a->shape[a->ndim - 1], b_rows = b->shape[b->ndim - 1];
  if (a_rows != a_cols) {
    fprintf(stderr, "Matrix 'a' must be square for solve: %d != %d\n", a_rows, a_cols);
    exit(EXIT_FAILURE);
  }
  if (a_rows != b_rows) {
    fprintf(stderr, "Matrix 'a' rows must match vector 'b' size: %d != %d\n", a_rows, b_rows);
    exit(EXIT_FAILURE);
  }

  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *b_float = convert_to_float32(b->data, b->dtype, b->size);
  int* result_shape = (int*)malloc(b->ndim * sizeof(int));

  for (size_t i = 0; i < b->ndim; i++) result_shape[i] = b->shape[i];
  size_t result_size = b->size;
  float* out = (float*)malloc(result_size * sizeof(float));
  if (a->ndim == 2 && b->ndim <= 2) {
    int shape_b[2] = {b->shape[b->ndim - 1], (b->ndim == 2) ? b->shape[1] : 1};
    solve_ops(a_float, b_float, out, a->shape + (a->ndim - 2), shape_b);
  } else {
    int shape_a_2d[2] = {a->shape[a->ndim - 2], a->shape[a->ndim - 1]};
    int shape_b_2d[2] = {b->shape[b->ndim - 1], (b->ndim >= 2) ? b->shape[b->ndim - 1] : 1};
    batched_solve_ops(a_float, b_float, out, shape_a_2d, shape_b_2d, a->ndim);
  }

  dtype_t result_dtype = promote_dtypes(a->dtype, b->dtype);
  Array* result = create_array(out, b->ndim, result_shape, result_size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);
  free(result_shape);
  return result;
}

Array* lstsq_array(Array* a, Array* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Input arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim < 2 || b->ndim < 1) {
    fprintf(stderr, "Matrix 'a' must be at least 2D and vector 'b' must be at least 1D\n");
    exit(EXIT_FAILURE);
  }
  int a_rows = a->shape[a->ndim - 2], a_cols = a->shape[a->ndim - 1], b_rows = b->shape[b->ndim - 1];
  if (a_rows != b_rows) {
    fprintf(stderr, "Matrix 'a' rows must match vector 'b' size: %d != %d\n", a_rows, b_rows);
    exit(EXIT_FAILURE);
  }

  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *b_float = convert_to_float32(b->data, b->dtype, b->size);
  size_t result_ndim = b->ndim;
  int* result_shape = (int*)malloc(result_ndim * sizeof(int));
  for (size_t i = 0; i < result_ndim - 1; i++) result_shape[i] = b->shape[i];
  result_shape[result_ndim - 1] = a_cols;

  size_t result_size = 1;
  for (size_t i = 0; i < result_ndim; i++) result_size *= result_shape[i];
  float* out = (float*)malloc(result_size * sizeof(float));
  if (a->ndim == 2 && b->ndim <= 2) {
    int shape_a_2d[2] = {a_rows, a_cols};
    int shape_b_2d[2] = {b_rows, (b->ndim == 2) ? b->shape[1] : 1};
    lstsq_ops(a_float, b_float, out, shape_a_2d, shape_b_2d);
  } else {
    int shape_a_2d[2] = {a_rows, a_cols};
    int shape_b_2d[2] = {b_rows, (b->ndim >= 2) ? b->shape[b->ndim - 1] : 1};
    batched_lstsq_ops(a_float, b_float, out, shape_a_2d, shape_b_2d, a->ndim);
  }

  dtype_t result_dtype = promote_dtypes(a->dtype, b->dtype);
  Array* result = create_array(out, result_ndim, result_shape, result_size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);
  free(result_shape);
  return result;
}