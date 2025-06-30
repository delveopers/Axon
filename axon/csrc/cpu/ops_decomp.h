#ifndef __OPS_DECOMP__H__
#define __OPS_DECOMP__H__

#include <stddef.h>

extern "C" {
  void det_ops_array(float* a, float* out, size_t size);
  void batched_det_ops(float* a, float* out, size_t size, size_t batch);
}

#endif  //!__OPS_DECOMP__H__