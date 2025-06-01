/**
  @file array.h header file for array.cpp & array
  * contains all the array related ops
  * imports basic core & basic functionalities from core.h
  * compile it as:
    *- '.so': g++ -shared -fPIC -o libarray.so core.cpp array.cpp maths_ops.cpp
    *- '.dll': g++ -shared -o libarray.dll core.cpp array.cpp maths_ops.cpp
    *- '.dylib': g++ -dynamiclib -o libarray.dylib core.cpp array.cpp maths_ops.cpp
*/

#ifndef __ARRAY__H__
#define __ARRAY__H__

#include <stdlib.h>
#include "core/core.h"

extern "C" {
  // array ops
  Array* add_array(Array* a, Array* b);
  Array* add_scalar_array(Array* a, float b);
  Array* add_broadcasted_array(Array* a, Array* b);
  Array* sub_array(Array* a, Array* b);
  Array* sub_scalar_array(Array* a, float b);
  Array* sub_broadcasted_array(Array* a, Array* b);
  Array* mul_array(Array* a, Array* b);
  Array* mul_scalar_array(Array* a, float b);
  Array* mul_broadcasted_array(Array* a, Array* b);
  Array* div_array(Array* a, Array* b);
  Array* div_scalar_array(Array* a, float b);
  Array* div_broadcasted_array(Array* a, Array* b);
  Array* sin_array(Array* a);
  Array* sinh_array(Array* a);
  Array* cos_array(Array* a);
  Array* cosh_array(Array* a);
  Array* tan_array(Array* a);
  Array* tanh_array(Array* a);
  Array* pow_array(Array* a, float exp);
  Array* pow_scalar(float a, Array* exp);
  Array* transpose_array(Array* a);
  Array* equal_array(Array* a, Array* b);
  Array* reshape_array(Array* a, int* new_shape, int new_ndim);
}

#endif  //!__ARRAY__H__