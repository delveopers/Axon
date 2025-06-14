#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include "binary_ops.h"

// standard 2D matrix multiplication: C = A @ B
// A: shape1[0] x shape1[1], B: shape1[1] x shape2[1], C: shape1[0] x shape2[1]
void matmul_array_ops(float* a, float* b, float* out, int* shape1, int* shape2) {
  for (int i = 0; i < shape1[0]; i++) {
    for (int j = 0; j < shape2[1]; j++) {
      float sum = 0.0f;
      for (int k = 0; k < shape1[1]; k++) {
        sum += a[i * shape1[1] + k] * b[k * shape2[1] + j];
      }
      out[i * shape2[1] + j] = sum;
    }
  }
}

void matmul_t_array_ops(float* a, float* b, float* out, size_t size) {
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < size; ++k) {
        sum += a[i * size + k] * b[j * size + k];
      }
      out[i * size + j] = sum;
    }
  }
}

// batch matrix multiplication: batched A @ batched B
// A: shape1[0] x shape1[1] x shape1[2], B: shape2[0] x shape2[1] x shape2[2]
// output: shape1[0] x shape1[1] x shape2[2] (assuming shape1[0] == shape2[0])
void batch_matmul_array_ops(float* a, float* b, float* out, int* shape1, int* shape2, int* strides1, int* strides2) {
  int batch_size = shape1[0];
  int out_stride = shape1[1] * shape2[2];
  
  for (int batch = 0; batch < batch_size; batch++) {
    for (int i = 0; i < shape1[1]; i++) {
      for (int j = 0; j < shape2[2]; j++) {
        float sum = 0.0f;
        for (int k = 0; k < shape1[2]; k++) {
          float a_val = a[batch * strides1[0] + i * shape1[2] + k];
          float b_val = b[batch * strides2[0] + k * shape2[2] + j];
          sum += a_val * b_val;
        }
        out[batch * out_stride + i * shape2[2] + j] = sum;
      }
    }
  }
}

// broadcasted matrix multiplication: single A * batched B
// A: shape1[0] x shape1[1], B: shape2[0] x shape2[1] x shape2[2]
// output: shape2[0] x shape1[0] x shape2[2]
void broadcasted_matmul_array_ops(float* a, float* b, float* out, int* shape1, int* shape2, int* strides1, int* strides2) {
  int out_stride = shape1[0] * shape2[2];
  
  for (int batch = 0; batch < shape2[0]; batch++) {
    for (int i = 0; i < shape1[0]; i++) {
      for (int j = 0; j < shape2[2]; j++) {
        float sum = 0.0f;
        for (int k = 0; k < shape1[1]; k++) {
          // A is broadcasted across batches, B is batched
          float a_val = a[i * shape1[1] + k];
          float b_val = b[batch * strides2[0] + k * shape2[2] + j];
          sum += a_val * b_val;
        }
        out[batch * out_stride + i * shape2[2] + j] = sum;
      }
    }
  }
}

// Dot product of two 1D vectors
// computes sum(a[i] * b[i]) for i = 0 to size-1
void dot_array_ops(float* a, float* b, float* out, size_t size) {
  float sum = 0.0f;
  for (size_t i = 0; i < size; i++) {
    sum += a[i] * b[i];
  }
  *out = sum;
}

// batch dot product of multiple pairs of 1D vectors
// a: batch_count x vector_size (flattened), b: batch_count x vector_size (flattened)
// out: batch_count (output array of dot products)
void batch_dot_array_ops(float* a, float* b, float* out, size_t batch_count, size_t vector_size) {
  for (size_t batch = 0; batch < batch_count; batch++) {
    float sum = 0.0f;
    size_t batch_offset = batch * vector_size;
    
    for (size_t i = 0; i < vector_size; i++) {
      sum += a[batch_offset + i] * b[batch_offset + i];
    }
    out[batch] = sum;
  }
}