#ifndef __HELPER__H__
#define __HELPER__H__

#include "inc/random.h"

extern "C" {
  void fill_randn(float* out, size_t size);
  void fill_uniform(float* out, float low, float high, size_t size);
  void fill_randint(float* out, int low, int high, size_t size);
}

#endif  //!__HELPER__H__