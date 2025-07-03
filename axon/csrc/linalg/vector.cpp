#include <stdio.h>
#include <stdlib.h>
#include "../cpu/ops_vector.h"
#include "vector.h"

Array* vector_dot(Array* a, Array* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 1 || b->ndim != 1) {
    fprintf(stderr, "Only 1D arrays supported for dot product\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != b->shape[0]) {
    fprintf(stderr, "Arrays must have same size for dot product. size_a '%d' != size_b '%d'\n", a->shape[0], b->shape[0]);
    exit(EXIT_FAILURE);
  }
  int* shape = (int*)malloc(1 * sizeof(int));
  shape[0] = 1;
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *b_float = convert_to_float32(b->data, b->dtype, b->size);
  float* out = (float*)malloc(1 * sizeof(float));
  if (a_float == NULL || b_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    if (out) free(out);
    exit(EXIT_FAILURE);
  }

  vector_dot_ops(a_float, b_float, out, a->size);
  Array* result = create_array(out, 1, shape, 1, a->dtype);
  free(a_float); free(b_float); free(out); free(shape);
  return result;
}

Array* vector_matrix_dot(Array* vec, Array* mat) {
  if (vec == NULL || mat == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (vec->ndim != 1 || mat->ndim != 2) {
    fprintf(stderr, "Vector must be 1D and matrix must be 2D for vector-matrix dot product\n");
    exit(EXIT_FAILURE);
  }
  if (vec->shape[0] != mat->shape[0]) {
    fprintf(stderr, "Vector size must match matrix rows. vec_size '%d' != mat_rows '%d'\n", vec->shape[0], mat->shape[0]);
    exit(EXIT_FAILURE);
  }
  int* shape = (int*)malloc(1 * sizeof(int));
  shape[0] = mat->shape[1];
  float *vec_float = convert_to_float32(vec->data, vec->dtype, vec->size), *mat_float = convert_to_float32(mat->data, mat->dtype, mat->size);
  float* out = (float*)malloc(mat->shape[1] * sizeof(float));
  if (vec_float == NULL || mat_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion");
    if (vec_float) free(vec_float);
    if (mat_float) free(mat_float);
    if (out) free(out);
    exit(EXIT_FAILURE);
  }

  vector_matrix_dot_ops(vec_float, mat_float, out, vec->size, mat->size);
  Array* result = create_array(out, 1, shape, mat->shape[1], vec->dtype);
  free(vec_float); free(mat_float); free(out); free(shape);
  return result;
}

Array* vector_inner(Array* a, Array* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 1 || b->ndim != 1) {
    fprintf(stderr, "Only 1D arrays supported for inner product\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != b->shape[0]) {
    fprintf(stderr, "Arrays must have same size for inner product. size_a '%d' != size_b '%d'\n", a->shape[0], b->shape[0]);
    exit(EXIT_FAILURE);
  }
  int* shape = (int*)malloc(1 * sizeof(int));
  shape[0] = 1;
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *b_float = convert_to_float32(b->data, b->dtype, b->size);
  float* out = (float*)malloc(1 * sizeof(float));
  if (a_float == NULL || b_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    if (out) free(out);
    exit(EXIT_FAILURE);
  }

  vector_inner_product_ops(a_float, b_float, out, a->size);
  Array* result = create_array(out, 1, shape, 1, a->dtype);
  free(a_float); free(b_float); free(out); free(shape);
  return result;
}

Array* vector_outer(Array* a, Array* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 1 || b->ndim != 1) {
    fprintf(stderr, "Only 1D arrays supported for outer product\n");
    exit(EXIT_FAILURE);
  }
  int* shape = (int*)malloc(2 * sizeof(int));
  shape[0] = a->shape[0]; shape[1] = b->shape[0];
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *b_float = convert_to_float32(b->data, b->dtype, b->size);
  float* out = (float*)malloc(a->shape[0] * b->shape[0] * sizeof(float));
  if (a_float == NULL || b_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    if (out) free(out);
    exit(EXIT_FAILURE);
  }

  vector_outer_product_ops(a_float, b_float, out, a->shape[0], b->shape[0]);
  Array* result = create_array(out, 2, shape, a->shape[0] * b->shape[0], a->dtype);
  free(a_float); free(b_float); free(out); free(shape);
  return result;
}

Array* vector_cross(Array* a, Array* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Arrays must have same number of dimensions for cross product\n");
    exit(EXIT_FAILURE);
  }
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *b_float = convert_to_float32(b->data, b->dtype, b->size);
  float* out = (float*)malloc(a->size * sizeof(float));
  if (a_float == NULL || b_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    if (out) free(out);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(a->ndim * sizeof(int));
  for (int i = 0; i < a->ndim; i++) { shape[i] = a->shape[i]; }
  if (a->ndim == 1) {
    if (a->shape[0] != b->shape[0]) {
      fprintf(stderr, "Arrays must have same size for cross product. size_a '%d' != size_b '%d'\n", a->shape[0], b->shape[0]);
      exit(EXIT_FAILURE);
    }
    cross_1d_ops(a_float, b_float, out, a->size);
  } else if (a->ndim == 2) {
    if (a->shape[0] != b->shape[0] || a->shape[1] != b->shape[1]) {
      fprintf(stderr, "Arrays must have same shape for cross product\n");
      exit(EXIT_FAILURE);
    }
    cross_2d_ops(a_float, b_float, out, a->shape[0], a->shape[1], -1);
  } else if (a->ndim == 3) {
    if (a->shape[0] != b->shape[0] || a->shape[1] != b->shape[1] || a->shape[2] != b->shape[2]) {
      fprintf(stderr, "Arrays must have same shape for cross product\n");
      exit(EXIT_FAILURE);
    }
    cross_3d_ops(a_float, b_float, out, a->shape[0], a->shape[1], a->shape[2], -1);
  } else {
    fprintf(stderr, "Only 1D, 2D, and 3D arrays supported for cross product\n");
    exit(EXIT_FAILURE);
  }

  Array* result = create_array(out, a->ndim, shape, a->size, a->dtype);
  free(a_float); free(b_float); free(out); free(shape);
  return result;
}

Array* vector_cross_axis(Array* a, Array* b, int axis) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "Arrays must have same number of dimensions for cross product\n");
    exit(EXIT_FAILURE);
  }
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *b_float = convert_to_float32(b->data, b->dtype, b->size);
  float* out = (float*)malloc(a->size * sizeof(float));
  if (a_float == NULL || b_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    if (out) free(out);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(a->ndim * sizeof(int));
  for (int i = 0; i < a->ndim; i++) { shape[i] = a->shape[i]; }
  if (a->ndim == 2) {
    if (a->shape[0] != b->shape[0] || a->shape[1] != b->shape[1]) {
      fprintf(stderr, "Arrays must have same shape for cross product\n");
      exit(EXIT_FAILURE);
    }
    cross_2d_ops(a_float, b_float, out, a->shape[0], a->shape[1], axis);
  } else if (a->ndim == 3) {
    if (a->shape[0] != b->shape[0] || a->shape[1] != b->shape[1] || a->shape[2] != b->shape[2]) {
      fprintf(stderr, "Arrays must have same shape for cross product\n");
      exit(EXIT_FAILURE);
    }
    cross_3d_ops(a_float, b_float, out, a->shape[0], a->shape[1], a->shape[2], axis);
  } else {
    fprintf(stderr, "Only 2D and 3D arrays supported for cross product with axis\n");
    exit(EXIT_FAILURE);
  }

  Array* result = create_array(out, a->ndim, shape, a->size, a->dtype);
  free(a_float); free(b_float); free(out); free(shape);
  return result;
}