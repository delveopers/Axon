#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include "ops.h"

void add_ops(float* a, float* b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = a[i] + b[i];
  }
}

void add_scalar_ops(float* a, float b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = a[i] + b;
  }
}

void sub_ops(float* a, float* b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = a[i] - b[i];
  }
}

void sub_scalar_ops(float* a, float b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = a[i] - b;
  }
}

void mul_ops(float* a, float* b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = a[i] * b[i];
  }
}

void mul_scalar_ops(float* a, float b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = a[i] * b;
  }
}

void div_ops(float* a, float* b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = a[i] / b[i];
  }
}

void div_scalar_ops(float* a, float b, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = a[i] / b;
  }
}