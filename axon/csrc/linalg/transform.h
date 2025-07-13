#ifndef __TRANSFORM__H__
#define __TRANSFORM__H__

#include "../core/core.h"

extern "C" {
  Array* linear_1d_array(Array* weight, Array* input, Array* bias);
  Array* linear_2d_array(Array* weight, Array* input, Array* bias);
  Array* linear_transform_array(Array* weights, Array* input, Array* bias);
}

#endif  //!__TRANSFORM__H__