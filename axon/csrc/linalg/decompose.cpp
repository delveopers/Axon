#include <stdio.h>
#include <stdlib.h>
#include "../cpu/ops_decomp.h"
#include "decompose.h"

Array* det_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D array supported for det()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Array must be square to compute det(). dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(1 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = 1;
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* out = (float*)malloc(1 * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }

  // Passing matrix dimension (shape[0]), not total size
  det_ops_array(a_float, out, a->shape[0]);
  Array* result = create_array(out, 1, shape, 1, a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Array* batched_det_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 3) {
    fprintf(stderr, "Only 3D array supported for batched det()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != a->shape[2]) {
    fprintf(stderr, "Array must be square to compute det(). dim1 '%d' != dim2 '%d'\n", a->shape[1], a->shape[2]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(1 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0]; // Output should have batch size
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* out = (float*)malloc(a->shape[0] * sizeof(float)); // allocating for batch size
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }

  // Pass matrix dimension (shape[1])
  batched_det_ops(a_float, out, a->shape[1], a->shape[0]);
  Array* result = create_array(out, 1, shape, a->shape[0], a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}