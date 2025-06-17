// to compile:
// --> g++ -O3 -std=c++11 -o run matmul_test.cpp array.cpp core/core.cpp core/dtype.cpp cpu/maths_ops.cpp cpu/utils.cpp cpu/helpers.cpp cpu/red_ops.cpp cpu/binary_ops.cpp core/contiguous.cpp
// --> run
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include "array.h"
#include "core/core.h"

using namespace std;
using namespace std::chrono;

// implementating naive matmul here, away from main codebase, just for benchmarking
// standard 2D matrix multiplication: C = A @ B
// A: shape1[0] x shape1[1], B: shape1[1] x shape2[1], C: shape1[0] x shape2[1]
void matmul_n_array_ops(float* a, float* b, float* out, int* shape1, int* shape2) {
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

// main array function
Array* matmul_n_array(Array* a, Array* b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Array value pointers are null!\n");
    exit(EXIT_FAILURE);
  }
  if (a->ndim != 2 || b->ndim != 2) {
    fprintf(stderr, "Both arrays must be 2D for matrix multiplication\n");
    exit(EXIT_FAILURE);
  }
  if (a->shape[1] != b->shape[0]) {
    fprintf(stderr, "Inner dimensions must match for matrix multiplication: %d != %d\n", a->shape[1], b->shape[0]);
    exit(EXIT_FAILURE);
  }

  // converting both arrays to float32 for computation
  float* a_float = convert_to_float32(a->data, a->dtype, a->size);
  float* b_float = convert_to_float32(b->data, b->dtype, b->size);

  if (a_float == NULL || b_float == NULL) {
    fprintf(stderr, "Memory allocation failed during dtype conversion\n");
    if (a_float) free(a_float);
    if (b_float) free(b_float);
    exit(EXIT_FAILURE);
  }

  // result shape: [a->shape[0], b->shape[1]]
  int* result_shape = (int*)malloc(2 * sizeof(int));
  if (result_shape == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(b_float);
    exit(EXIT_FAILURE);
  }
  result_shape[0] = a->shape[0];
  result_shape[1] = b->shape[1];
  size_t result_size = result_shape[0] * result_shape[1];

  float* out = (float*)malloc(result_size * sizeof(float));
  if (out == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    free(a_float);
    free(b_float);
    free(result_shape);
    exit(EXIT_FAILURE);
  }

  matmul_n_array_ops(a_float, b_float, out, a->shape, b->shape);
  dtype_t result_dtype = promote_dtypes(a->dtype, b->dtype);
  Array* result = create_array(out, 2, result_shape, result_size, result_dtype);
  free(a_float);
  free(b_float);
  free(out);
  free(result_shape);
  return result;
}

void benchmark_matmul(int size1, int size2, int size3, int size4) {
  printf("\n====== Benchmarking size: (%d x %d) @ (%d x %d) ======\n", size1, size2, size3, size4);
  
  int* shape1 = (int*)malloc(2 * sizeof(int));
  int* shape2 = (int*)malloc(2 * sizeof(int));
  shape1[0] = size1, shape1[1] = size2;
  shape2[0] = size3, shape2[1] = size4;  // FIXED: was shape2[0] = size3 twice
  size_t total_size1 = size1 * size2, total_size2 = size3 * size4;
  size_t ndim = 2;

  Array* a = randn_array(shape1, total_size1, ndim, DTYPE_FLOAT32);
  Array* b = randn_array(shape2, total_size2, ndim, DTYPE_FLOAT32);

  // Benchmark naive matmul_array
  auto start1 = high_resolution_clock::now();
  Array* c = matmul_n_array(a, b);
  auto end1 = high_resolution_clock::now();
  auto duration1 = duration_cast<milliseconds>(end1 - start1);
  printf("Naive matmul time: %lld ms\n", duration1.count());

  // Benchmark optimized matmul_t_array
  auto start2 = high_resolution_clock::now();
  Array* d = matmul_array(a, b);
  auto end2 = high_resolution_clock::now();
  auto duration2 = duration_cast<milliseconds>(end2 - start2);
  printf("Transposed matmul time: %lld ms\n", duration2.count());

  printf("array1: (%d x %d)\n", c->shape[0], c->shape[1]);
  printf("array2: (%d x %d)\n", d->shape[0], d->shape[1]);

  Array* eq = equal_array(c, d);
  printf("checking if equal: \n");
  print_array(eq);

  delete_array(a);
  delete_array(b);
  delete_array(c);
  delete_array(d);
  delete_array(eq);  // ADDED: delete the equality result array too
  free(shape1);
  free(shape2);
}

int main() {
  benchmark_matmul(256, 256, 256, 100);
  benchmark_matmul(512, 256, 256, 100);
  benchmark_matmul(1024, 512, 512, 768);
  return 0;
}