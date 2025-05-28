#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include "maths_ops.h"

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

void sin_ops(float* a, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = sinf(a[i]);
  }
}

void cos_ops(float* a, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = cosf(a[i]);
  }
}

void tan_ops(float* a, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = tanf(a[i]);
  }
}

void sinh_ops(float* a, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = sinhf(a[i]);
  }
}

void cosh_ops(float* a, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = coshf(a[i]);
  }
}

void tanh_ops(float* a, float* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    out[i] = tanhf(a[i]);
  }
}

void ones_like_array_ops(float* out, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = 1.0f;
  }
}

void zeros_like_array_ops(float* out, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = 0.0f;
  }
}

void ones_array_ops(float* out, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = 1.0f;
  }
}

void zeros_array_ops(float* out, size_t size) {
  for (int i = 0; i < size; i++) {
    out[i] = 0.0f;
  }
}