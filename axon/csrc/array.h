/**
  @file array.h header file for array.cpp & array
  * compile it as:
    *- '.so': g++ -shared -fPIC -o libarray.so array.cpp ops.cpp
    *- '.dll': g++ -shared -o libarray.dll array.cpp ops.cpp
*/

#ifndef __ARRAY__H__
#define __ARRAY__H__

#include <stdlib.h>

typedef struct Array {
  float* data;
  int* strides;
  int* backstrides;
  int* shape;
  size_t size;
  size_t ndim;
} Array;

extern "C" {
  // array initialization & deletion related function
  Array* create_array(float* data, size_t ndim, int* shape, size_t size);
  void delete_array(Array* self);
  void delete_shape(Array* self);
  void delete_data(Array* self);
  void delete_strides(Array* self);

  Array* add_array(Array* a, Array* b);
  Array* add_array_array(Array* a, float b);
  Array* sub_array(Array* a, Array* b);
  Array* sub_array_array(Array* a, float b);
  Array* mul_array(Array* a, Array* b);
  Array* mul_array_array(Array* a, float b);
  Array* div_array(Array* a, Array* b);
  Array* div_array_array(Array* a, float b);

  void print_tensor(Array* self);
}

#endif  //!__ARRAY__H__