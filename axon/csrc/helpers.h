#ifndef __HELPER__H__
#define __HELPER__H__

#include "inc/random.h"

extern "C" {
  void fill_randn(float* out, size_t size);
  void fill_uniform(float* out, float low, float high, size_t size);
  void fill_randint(float* out, int low, int high, size_t size);

  void zeros_like_array_ops(float* out, size_t size);
  void zeros_array_ops(float* out, size_t size);
  void ones_like_array_ops(float* out, size_t size);
  void ones_array_ops(float* out, size_t size);

  void fill_array_ops(float* out, float value, size_t size);
  void linspace_array_ops(float* out, float start, float step_size, size_t size);
}

#endif  //!__HELPER__H__