#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "core/core.h"
#include "array.h"
#include "cpu/maths_ops.h"
#include "cpu/utils.h"

Array* add_array(Array* a, Array* b) {
  if (a == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "arrays must have the same no of dims %d and %d for addition\n", a->ndim, b->ndim);
    exit(1);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  add_ops(a->data, b->data, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* add_scalar_array(Array* a, float b) {
  if (a == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  add_scalar_ops(a->data, b, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* add_broadcasted_array(Array* a, Array* b) {
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i < a->ndim ? a->shape[a->ndim - 1 -i] : 1, dim2 = i < b->ndim ? b->shape[b->ndim - 1 -i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 2) {
      fprintf(stderr, "shapes are not compatible for broadcasting\n");
      exit(1);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }
  int broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  float* out = (float*)malloc(broadcasted_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  add_broadcasted_array_ops(a->data, b->data, out, broadcasted_shape, broadcasted_size, a->ndim, b->ndim, a->shape, b->shape);
  return create_array(out, max_ndim, broadcasted_shape, broadcasted_size);
}

Array* sub_array(Array* a, Array* b) {
  if (a == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "arrays must have the same no of dims %d and %d for subtraction\n", a->ndim, b->ndim);
    exit(1);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  sub_ops(a->data, b->data, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* sub_scalar_array(Array* a, float b) {
  if (a == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  sub_scalar_ops(a->data, b, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* sub_broadcasted_array(Array* a, Array* b) {
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i < a->ndim ? a->shape[a->ndim - 1 -i] : 1, dim2 = i < b->ndim ? b->shape[b->ndim - 1 -i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 2) {
      fprintf(stderr, "shapes are not compatible for broadcasting\n");
      exit(1);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }
  int broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  float* out = (float*)malloc(broadcasted_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  sub_broadcasted_array_ops(a->data, b->data, out, broadcasted_shape, broadcasted_size, a->ndim, b->ndim, a->shape, b->shape);
  return create_array(out, max_ndim, broadcasted_shape, broadcasted_size);
}

Array* mul_array(Array* a, Array* b) {
  if (a == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "arrays must have the same no of dims %d and %d for multiplication\n", a->ndim, b->ndim);
    exit(1);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  mul_ops(a->data, b->data, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* mul_scalar_array(Array* a, float b) {
  if (a == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  mul_scalar_ops(a->data, b, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* mul_broadcasted_array(Array* a, Array* b) {
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i < a->ndim ? a->shape[a->ndim - 1 -i] : 1, dim2 = i < b->ndim ? b->shape[b->ndim - 1 -i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 2) {
      fprintf(stderr, "shapes are not compatible for broadcasting\n");
      exit(1);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }
  int broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  float* out = (float*)malloc(broadcasted_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  mul_broadcasted_array_ops(a->data, b->data, out, broadcasted_shape, broadcasted_size, a->ndim, b->ndim, a->shape, b->shape);
  return create_array(out, max_ndim, broadcasted_shape, broadcasted_size);
}

Array* div_array(Array* a, Array* b) {
  if (a == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "arrays must have the same no of dims %d and %d for divison\n", a->ndim, b->ndim);
    exit(1);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  div_ops(a->data, b->data, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* div_scalar_array(Array* a, float b) {
  if (a == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  div_scalar_ops(a->data, b, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* div_broadcasted_array(Array* a, Array* b) {
  int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
  int* broadcasted_shape = (int*)malloc(max_ndim * sizeof(int));
  if (broadcasted_shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < max_ndim; i++) {
    int dim1 = i < a->ndim ? a->shape[a->ndim - 1 -i] : 1, dim2 = i < b->ndim ? b->shape[b->ndim - 1 -i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 2) {
      fprintf(stderr, "shapes are not compatible for broadcasting\n");
      exit(1);
    }
    broadcasted_shape[max_ndim - 1 - i] = dim1 > dim2 ? dim1 : dim2;
  }
  int broadcasted_size = 1;
  for (int i = 0; i < max_ndim; i++) {
    broadcasted_size *= broadcasted_shape[i];
  }
  float* out = (float*)malloc(broadcasted_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  div_broadcasted_array_ops(a->data, b->data, out, broadcasted_shape, broadcasted_size, a->ndim, b->ndim, a->shape, b->shape);
  return create_array(out, max_ndim, broadcasted_shape, broadcasted_size);
}

Array* sin_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  sin_ops(a->data, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* sinh_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  sinh_ops(a->data, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* cos_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  cos_ops(a->data, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* cosh_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  cosh_ops(a->data, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* tan_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  sin_ops(a->data, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* tanh_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  tanh_ops(a->data, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* pow_array(Array* a, float exp) {
  if (a == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  pow_array_ops(a->data, exp, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}

Array* pow_scalar(float a, Array* exp) {
  if (exp == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  float* out = (float*)malloc(exp->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  pow_scalar_ops(a, exp->data, out, exp->size);
  return create_array(out, exp->ndim, exp->shape, exp->size);
}

Array* transpose_array(Array* a) {
  int ndim = a->ndim, *shape = (int*)malloc(ndim * sizeof(int)), size = a->size;
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < ndim; i++) {
    shape[i] = a->shape[ndim - 1 - i];
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed!");
    exit(1);
  }
  switch(ndim) {
    case 1:
      transpose_1d_array_ops(a->data, out, shape);
      break;
    case 2:
      transpose_2d_array_ops(a->data, out, shape);
      break;
    case 3:
      transpose_3d_array_ops(a->data, out, shape);
      break;
    default:
      fprintf(stderr, "Transpose supported only for 3-dim array");
      exit(1);
  }
  return create_array(out, ndim, shape, size);
}

Array* reshape_array(Array* a, int* new_shape, int new_ndim) {
  int ndim = new_ndim, *shape = (int*)malloc(ndim * sizeof(int));
  if (shape == NULL) {
    fprintf(stderr, "Memory allocation failed");
    exit(1);
  }
  for (int i = 0; i < ndim; i++) { shape[i] = new_shape[i]; }
  int size = 1;
  for (int i = 0; i < new_ndim; i++) { size *= shape[i];}
  if (size != a->size) {
    fprintf(stderr, "Can't reshape the array. array's size doesn't match the target size: %d != %d", a->size, size);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed!");
    exit(1);
  }
  reassign_array_ops(a->data, out, size);
  return create_array(out, ndim, shape, size);
}

Array* equal_array(Array* a, Array* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != b->ndim) {
    fprintf(stderr, "arrays must have same dimensions %d and %d for equal", a->ndim, b->ndim);
    exit(1);
  }
  float* out = (float*)malloc(a->size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed!");
    exit(1);
  }
  equal_array_ops(a->data, b->data, out, a->size);
  return create_array(out, a->ndim, a->shape, a->size);
}