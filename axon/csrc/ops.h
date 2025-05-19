#ifndef __OPS__H__
#define __OPS__H__

#include <stddef.h>

extern "C" {
  void add_ops(float* a, float* b, float* out, size_t size);
  void add_scalar_ops(float* a, float b, float* out, size_t size);
  void sub_ops(float* a, float* b, float* out, size_t size);
  void sub_scalar_ops(float* a, float b, float* out, size_t size);
  void mul_ops(float* a, float* b, float* out, size_t size);
  void mul_scalar_ops(float* a, float b, float* out, size_t size);
  void div_ops(float* a, float* b, float* out, size_t size);
  void div_scalar_ops(float* a, float b, float* out, size_t size);
}

#endif  //!__OPS__H__