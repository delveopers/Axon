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

Array* eig_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D array supported for eig()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Array must be square to compute eigenvalues. dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(1 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0]; // eigenvalues count equals matrix dimension
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->shape[0] * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }
  eigenvals_ops_array(a_float, out, a->shape[0]);
  Array* result = create_array(out, 1, shape, a->shape[0], a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Array* eigv_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D array supported for eigv()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Array must be square to compute eigenvectors. dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(2 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0]; shape[1] = a->shape[1]; // same dimensions as input matrix
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->size * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }
  eigenvecs_ops_array(a_float, out, a->shape[0]);
  Array* result = create_array(out, 2, shape, a->size, a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Array* eigh_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D array supported for eigh()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Array must be square to compute hermitian eigenvalues. dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(1 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0]; // eigenvalues count equals matrix dimension
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->shape[0] * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }

  eigenvals_h_ops_array(a_float, out, a->shape[0]);
  Array* result = create_array(out, 1, shape, a->shape[0], a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Array* eighv_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D array supported for eighv()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Array must be square to compute hermitian eigenvectors. dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(2 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0]; shape[1] = a->shape[1]; // same dimensions as input matrix
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->size * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }
  eigenvecs_h_ops_array(a_float, out, a->shape[0]);
  Array* result = create_array(out, 2, shape, a->size, a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Array* batched_eig_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 3) {
    fprintf(stderr, "Only 3D array supported for batched eig()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != a->shape[2]) {
    fprintf(stderr, "Array must be square to compute eigenvalues. dim1 '%d' != dim2 '%d'\n", a->shape[1], a->shape[2]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(2 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0], shape[1] = a->shape[1];
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->shape[0] * a->shape[1] * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }

  batched_eigenvals_ops(a_float, out, a->shape[1], a->shape[0]);
  Array* result = create_array(out, 2, shape, a->shape[0] * a->shape[1], a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Array* batched_eigv_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 3) {
    fprintf(stderr, "Only 3D array supported for batched eigv()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != a->shape[2]) {
    fprintf(stderr, "Array must be square to compute eigenvectors. dim1 '%d' != dim2 '%d'\n", a->shape[1], a->shape[2]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(3 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0], shape[1] = a->shape[1], shape[2] = a->shape[2];
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->size * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }
  batched_eigenvecs_ops(a_float, out, a->shape[1], a->shape[0]);
  Array* result = create_array(out, 3, shape, a->size, a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Array* batched_eigh_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 3) {
    fprintf(stderr, "Only 3D array supported for batched eigh()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != a->shape[2]) {
    fprintf(stderr, "Array must be square to compute hermitian eigenvalues. dim1 '%d' != dim2 '%d'\n", a->shape[1], a->shape[2]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(2 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0], shape[1] = a->shape[1];
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->shape[0] * a->shape[1] * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }
  batched_eigenvals_h_ops(a_float, out, a->shape[1], a->shape[0]);
  Array* result = create_array(out, 2, shape, a->shape[0] * a->shape[1], a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Array* batched_eighv_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 3) {
    fprintf(stderr, "Only 3D array supported for batched eighv()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != a->shape[2]) {
    fprintf(stderr, "Array must be square to compute hermitian eigenvectors. dim1 '%d' != dim2 '%d'\n", a->shape[1], a->shape[2]);
    exit(EXIT_FAILURE);
  }

  int* shape = (int*)malloc(3 * sizeof(int));
  if (!shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  shape[0] = a->shape[0], shape[1] = a->shape[1], shape[2] = a->shape[2];
  float *a_float = convert_to_float32(a->data, a->dtype, a->size), *out = (float*)malloc(a->size * sizeof(float));
  if (a_float == NULL || out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (out) free(out);
    if (shape) free(shape);
    exit(EXIT_FAILURE);
  }

  batched_eigenvecs_h_ops(a_float, out, a->shape[1], a->shape[0]);
  Array* result = create_array(out, 3, shape, a->size, a->dtype);
  free(a_float); 
  free(out); 
  free(shape);
  return result;
}

Array** qr_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D array supported for qr()\n");
    exit(EXIT_FAILURE);
  }

  int m = a->shape[0], n = a->shape[1];
  int* q_shape = (int*)malloc(2 * sizeof(int));
  int* r_shape = (int*)malloc(2 * sizeof(int));
  if (!q_shape || !r_shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  q_shape[0] = m; q_shape[1] = m;
  r_shape[0] = m; r_shape[1] = n;

  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* q_out = (float*)malloc(m * m * sizeof(float));
  float* r_out = (float*)malloc(m * n * sizeof(float));
  if (a_float == NULL || q_out == NULL || r_out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (q_out) free(q_out);
    if (r_out) free(r_out);
    if (q_shape) free(q_shape);
    if (r_shape) free(r_shape);
    exit(EXIT_FAILURE);
  }

  qr_decomp_ops(a_float, q_out, r_out, a->shape);
  Array** result = (Array**)malloc(2 * sizeof(Array*));
  result[0] = create_array(q_out, 2, q_shape, m * m, a->dtype);
  result[1] = create_array(r_out, 2, r_shape, m * n, a->dtype);
  free(a_float);
  free(q_out);
  free(r_out);
  free(q_shape);
  free(r_shape);
  return result;
}

Array** batched_qr_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim < 2) {
    fprintf(stderr, "Array must have at least 2 dimensions for batched qr()\n");
    exit(EXIT_FAILURE);
  }

  int m = a->shape[a->ndim - 2], n = a->shape[a->ndim - 1];
  int batch_size = 1;
  for (int i = 0; i < a->ndim - 2; i++) { batch_size *= a->shape[i]; }

  int* q_shape = (int*)malloc(a->ndim * sizeof(int));
  int* r_shape = (int*)malloc(a->ndim * sizeof(int));
  if (!q_shape || !r_shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < a->ndim - 2; i++) {
    q_shape[i] = a->shape[i];
    r_shape[i] = a->shape[i];
  }
  q_shape[a->ndim - 2] = m; q_shape[a->ndim - 1] = m;
  r_shape[a->ndim - 2] = m; r_shape[a->ndim - 1] = n;

  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* q_out = (float*)malloc(batch_size * m * m * sizeof(float));
  float* r_out = (float*)malloc(batch_size * m * n * sizeof(float));
  if (a_float == NULL || q_out == NULL || r_out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (q_out) free(q_out);
    if (r_out) free(r_out);
    if (q_shape) free(q_shape);
    if (r_shape) free(r_shape);
    exit(EXIT_FAILURE);
  }

  batched_qr_decomp_ops(a_float, q_out, r_out, a->shape, a->ndim);
  Array** result = (Array**)malloc(2 * sizeof(Array*));
  result[0] = create_array(q_out, a->ndim, q_shape, batch_size * m * m, a->dtype);
  result[1] = create_array(r_out, a->ndim, r_shape, batch_size * m * n, a->dtype);
  free(a_float);
  free(q_out);
  free(r_out);
  free(q_shape);
  free(r_shape);
  return result;
}

Array** lu_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D array supported for lu()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[0] != a->shape[1]) {
    fprintf(stderr, "Array must be square for lu(). dim0 '%d' != dim1 '%d'\n", a->shape[0], a->shape[1]);
    exit(EXIT_FAILURE);
  }

  int n = a->shape[0];
  int* l_shape = (int*)malloc(2 * sizeof(int));
  int* u_shape = (int*)malloc(2 * sizeof(int));
  int* p_shape = (int*)malloc(1 * sizeof(int));
  if (!l_shape || !u_shape || !p_shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  l_shape[0] = n; l_shape[1] = n;
  u_shape[0] = n; u_shape[1] = n;
  p_shape[0] = n;

  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* l_out = (float*)malloc(n * n * sizeof(float));
  float* u_out = (float*)malloc(n * n * sizeof(float));
  int* p_out = (int*)malloc(n * sizeof(int));
  if (a_float == NULL || l_out == NULL || u_out == NULL || p_out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (l_out) free(l_out);
    if (u_out) free(u_out);
    if (p_out) free(p_out);
    if (l_shape) free(l_shape);
    if (u_shape) free(u_shape);
    if (p_shape) free(p_shape);
    exit(EXIT_FAILURE);
  }

  lu_decomp_ops(a_float, l_out, u_out, p_out, a->shape);
  float* p_float = (float*)malloc(n * sizeof(float));
  for (int i = 0; i < n; i++) { p_float[i] = (float)p_out[i]; }
  
  Array** result = (Array**)malloc(3 * sizeof(Array*));
  result[0] = create_array(l_out, 2, l_shape, n * n, a->dtype);
  result[1] = create_array(u_out, 2, u_shape, n * n, a->dtype);
  result[2] = create_array(p_float, 1, p_shape, n, a->dtype);
  free(a_float);
  free(l_out);
  free(u_out);
  free(p_out);
  free(p_float);
  free(l_shape);
  free(u_shape);
  free(p_shape);
  return result;
}

Array** batched_lu_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim < 2) {
    fprintf(stderr, "Array must have at least 2 dimensions for batched lu()\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[a->ndim - 1] != a->shape[a->ndim - 2]) {
    fprintf(stderr, "Array must be square for lu(). dim%d '%d' != dim%d '%d'\n", a->ndim - 2, a->shape[a->ndim - 2], a->ndim - 1, a->shape[a->ndim - 1]);
    exit(EXIT_FAILURE);
  }

  int n = a->shape[a->ndim - 1];
  int batch_size = 1;
  for (int i = 0; i < a->ndim - 2; i++) { batch_size *= a->shape[i]; }

  int* l_shape = (int*)malloc(a->ndim * sizeof(int));
  int* u_shape = (int*)malloc(a->ndim * sizeof(int));
  int* p_shape = (int*)malloc(a->ndim - 1 * sizeof(int));
  if (!l_shape || !u_shape || !p_shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < a->ndim; i++) {
    l_shape[i] = a->shape[i];
    u_shape[i] = a->shape[i];
  }
  for (int i = 0; i < a->ndim - 1; i++) { p_shape[i] = a->shape[i]; }

  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* l_out = (float*)malloc(a->size * sizeof(float));
  float* u_out = (float*)malloc(a->size * sizeof(float));
  int* p_out = (int*)malloc(batch_size * n * sizeof(int));
  if (a_float == NULL || l_out == NULL || u_out == NULL || p_out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (l_out) free(l_out);
    if (u_out) free(u_out);
    if (p_out) free(p_out);
    if (l_shape) free(l_shape);
    if (u_shape) free(u_shape);
    if (p_shape) free(p_shape);
    exit(EXIT_FAILURE);
  }

  batched_lu_decomp_ops(a_float, l_out, u_out, p_out, a->shape, a->ndim);
  float* p_float = (float*)malloc(batch_size * n * sizeof(float));
  for (int i = 0; i < batch_size * n; i++) { p_float[i] = (float)p_out[i]; }

  Array** result = (Array**)malloc(3 * sizeof(Array*));
  result[0] = create_array(l_out, a->ndim, l_shape, a->size, a->dtype);
  result[1] = create_array(u_out, a->ndim, u_shape, a->size, a->dtype);
  result[2] = create_array(p_float, a->ndim - 1, p_shape, batch_size * n, a->dtype);
  free(a_float);
  free(l_out);
  free(u_out);
  free(p_out);
  free(p_float);
  free(l_shape);
  free(u_shape);
  free(p_shape);
  return result;
}

Array** lq_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 2) {
    fprintf(stderr, "Only 2D array supported for lq()\n");
    exit(EXIT_FAILURE);
  }

  int m = a->shape[0], n = a->shape[1];
  int* l_shape = (int*)malloc(2 * sizeof(int));
  int* q_shape = (int*)malloc(2 * sizeof(int));
  if (!l_shape || !q_shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  l_shape[0] = m; l_shape[1] = m;
  q_shape[0] = m; q_shape[1] = n;

  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* l_out = (float*)malloc(m * m * sizeof(float));
  float* q_out = (float*)malloc(m * n * sizeof(float));
  if (a_float == NULL || l_out == NULL || q_out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (l_out) free(l_out);
    if (q_out) free(q_out);
    if (l_shape) free(l_shape);
    if (q_shape) free(q_shape);
    exit(EXIT_FAILURE);
  }

  lq_decomp_ops(a_float, l_out, q_out, a->shape);
  Array** result = (Array**)malloc(2 * sizeof(Array*));
  result[0] = create_array(l_out, 2, l_shape, m * m, a->dtype);
  result[1] = create_array(q_out, 2, q_shape, m * n, a->dtype);
  free(a_float);
  free(l_out);
  free(q_out);
  free(l_shape);
  free(q_shape);
  return result;
}

Array** batched_lq_array(Array* a) {
  if (a == NULL) {
    fprintf(stderr, "Arrays cannot be null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim < 2) {
    fprintf(stderr, "Array must have at least 2 dimensions for batched lq()\n");
    exit(EXIT_FAILURE);
  }

  int m = a->shape[a->ndim - 2], n = a->shape[a->ndim - 1];
  int batch_size = 1;
  for (int i = 0; i < a->ndim - 2; i++) { batch_size *= a->shape[i]; }

  int* l_shape = (int*)malloc(a->ndim * sizeof(int));
  int* q_shape = (int*)malloc(a->ndim * sizeof(int));
  if (!l_shape || !q_shape) {
    fprintf(stderr, "Memory allocation failed for shape\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < a->ndim - 2; i++) {
    l_shape[i] = a->shape[i];
    q_shape[i] = a->shape[i];
  }
  l_shape[a->ndim - 2] = m; l_shape[a->ndim - 1] = m;
  q_shape[a->ndim - 2] = m; q_shape[a->ndim - 1] = n;

  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* l_out = (float*)malloc(batch_size * m * m * sizeof(float));
  float* q_out = (float*)malloc(batch_size * m * n * sizeof(float));
  if (a_float == NULL || l_out == NULL || q_out == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (l_out) free(l_out);
    if (q_out) free(q_out);
    if (l_shape) free(l_shape);
    if (q_shape) free(q_shape);
    exit(EXIT_FAILURE);
  }

  batched_lq_decomp_ops(a_float, l_out, q_out, a->shape, a->ndim);
  Array** result = (Array**)malloc(2 * sizeof(Array*));
  result[0] = create_array(l_out, a->ndim, l_shape, batch_size * m * m, a->dtype);
  result[1] = create_array(q_out, a->ndim, q_shape, batch_size * m * n, a->dtype);
  free(a_float);
  free(l_out);
  free(q_out);
  free(l_shape);
  free(q_shape);
  return result;
}