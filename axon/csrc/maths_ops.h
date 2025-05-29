#ifndef __MATHS_OPS__H__
#define __MATHS_OPS__H__

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
  void pow_array_ops(float* a, float exp, float* out, size_t size);
  void pow_scalar_ops(float a, float* exp, float* out, size_t size);

  void sin_ops(float* a, float* out, size_t size);
  void cos_ops(float* a, float* out, size_t size);
  void tan_ops(float* a, float* out, size_t size);
  void sinh_ops(float* a, float* out, size_t size);
  void cosh_ops(float* a, float* out, size_t size);
  void tanh_ops(float* a, float* out, size_t size);
}

#endif  //!__OPS__H__