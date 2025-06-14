// to compile:
// --> g++ -O3 -std=c++11 -o run test.cpp array.cpp core/core.cpp core/dtype.cpp cpu/maths_ops.cpp cpu/utils.cpp cpu/helpers.cpp cpu/red_ops.cpp cpu/binary_ops.cpp
// --> run
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include "array.h"

using namespace std;
using namespace std::chrono;

void benchmark_matmul(int size) {
  printf("\n====== Benchmarking size: %d x %d ======\n", size, size);
  
  int* shape = (int*)malloc(2 * sizeof(int));
  shape[0] = size;
  shape[1] = size;
  size_t total_size = size * size;
  size_t ndim = 2;

  Array* a = randn_array(shape, total_size, ndim, DTYPE_FLOAT32);
  Array* b = randn_array(shape, total_size, ndim, DTYPE_FLOAT32);
  // Benchmark optimized matmul_t_array
  auto start2 = high_resolution_clock::now();
  Array* d = matmul_t_array(a, b);
  auto end2 = high_resolution_clock::now();
  auto duration2 = duration_cast<milliseconds>(end2 - start2);
  printf("Transposed matmul time: %lld ms\n", duration2.count());

  // Benchmark naive matmul_array
  auto start1 = high_resolution_clock::now();
  Array* c = matmul_array(a, b);
  auto end1 = high_resolution_clock::now();
  auto duration1 = duration_cast<milliseconds>(end1 - start1);
  printf("Naive matmul time: %lld ms\n", duration1.count());

  delete_array(a);
  delete_array(b);
  delete_array(c);
  delete_array(d);
  free(shape);
}

int main() {
  benchmark_matmul(256);
  benchmark_matmul(512);
  benchmark_matmul(1024);
  return 0;
}
