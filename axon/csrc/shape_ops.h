#ifndef __SHAPE_OPS__H__
#define __SHAPE_OPS__H__

#include "core/core.h"
#include "core/dtype.h"

extern "C" {
  // shaping ops
  Array* transpose_array(Array* a);
  Array* equal_array(Array* a, Array* b);
  Array* reshape_array(Array* a, int* new_shape, int new_ndim);
  Array* squeeze_array(Array* a, int axis);
  Array* expand_dims_array(Array* a, int axis);
  Array* flatten_array(Array* a);
}

#endif  //!__SHAPE_OPS__H__