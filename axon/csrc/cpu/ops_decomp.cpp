#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "ops_decomp.h"

void det_ops_array(float* a, float* out, size_t size) {
  float det = 1.0f;
  // creating a copy to avoid modifying original matrix
  float* temp = (float*)malloc(size * size * sizeof(float));
  if (!temp) { *out = 0.0f; return; }
  for (size_t i = 0; i < size * size; ++i) { temp[i] = a[i]; }
  // Gaussian elimination with partial pivoting
  for (size_t i = 0; i < size; ++i) {
    // finding pivot
    size_t pivot_row = i;
    float max_val = fabsf(temp[i * size + i]);
    for (size_t row = i + 1; row < size; ++row) {
      float val = fabsf(temp[row * size + i]);
      if (val > max_val) {
        max_val = val;
        pivot_row = row;
      }
    }
    // swapping rows if needed
    if (pivot_row != i) {
      for (size_t col = 0; col < size; ++col) {
        float tmp = temp[i * size + col];
        temp[i * size + col] = temp[pivot_row * size + col];
        temp[pivot_row * size + col] = tmp;
      }
      det = -det; // Row swap changes sign
    }
    float pivot = temp[i * size + i];
    if (fabsf(pivot) < 1e-6f) { det = 0.0f; break; }
    det *= pivot;
    // eliminating column
    for (size_t j = i + 1; j < size; ++j) {
      float factor = temp[j * size + i] / pivot;
      for (size_t k = i; k < size; ++k) { temp[j * size + k] -= factor * temp[i * size + k]; }
    }
  }
  free(temp);
  *out = det;
}

void batched_det_ops(float* a, float* out, size_t size, size_t batch) {
  size_t mat_size = size * size;
  for (size_t b = 0; b < batch; ++b) {
    float* mat = &a[b * mat_size];
    det_ops_array(mat, &out[b], size);
  }
}