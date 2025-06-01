#include <stdlib.h>
#include <stddef.h>
#include "utils.h"

void reassign_array_ops(float* a, float* out, size_t size) {
  for (int i = 0; i < size; i++) { out[i] = a[i]; }
}

void equal_array_ops(float* a, float* b, float* out, size_t size) {
  for (int i = 0; i < size; i++) { out[i] = (a[i] == b[i]) ? 1 : 0;}
}

void transpose_1d_array_ops(float* a, float* out, int* shape) {
  for (int i = 0; i < shape[0]; i++) { out[i] = a[i]; }
}

void transpose_2d_array_ops(float* a, float* out, int* shape) {
  int rows = shape[0], cols = shape[1];
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      out[j * rows + i] = a[i * cols + j];
    }
  }
}

void transpose_3d_array_ops(float* a, float* out, int* shape) {
  int batch = shape[0], rows = shape[1], cols = shape[2];
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < rows; j++) {
      for (int k = 0; k < cols; k++) {
        out[k * rows * cols + j * batch + i] = a[i * rows * cols + cols * j + k];
      }
    }
  }
}