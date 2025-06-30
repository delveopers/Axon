#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "ops_decomp.h"

void det_ops_array(float* a, float* out, size_t size) {
  float det = 1.0f;
  for (size_t i = 0; i < size; ++i) {
    float pivot = a[i * size + i];
    if (fabsf(pivot) < 1e-6) { det = 0.0f; break; }  // singular or close to 0
    for (size_t j = i + 1; j < size; ++j) {
      float factor = a[j * size + i] / pivot;
      for (size_t k = i; k < size; ++k) a[j * size + k] -= factor * a[i * size + k];
    }
    det *= pivot;
  }
  free(a);
  *out = det;
}

void batched_det_ops(float* a, float* out, size_t size, size_t batch) {
  size_t mat_size = size * size;
  for (size_t b = 0; b < batch; ++b) {
    float* mat = &a[b * mat_size], det = 1.0f;
    for (size_t i = 0; i < size; ++i) {
      float pivot = mat[i * size + i];
      if (fabsf(pivot) < 1e-6f) { det = 0.0f; break; }
      for (size_t j = i + 1; j < size; ++j) {
        float factor = mat[j * size + i] / pivot;
        for (size_t k = i; k < size; ++k) mat[j * size + k] -= factor * mat[i * size + k];
      }
      det *= pivot;
    }
    out[b] = det;
  }
}